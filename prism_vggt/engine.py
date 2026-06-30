import gc
import os
import time
import hashlib
from typing import Optional, List, Tuple, Dict, Iterator, Any
import torch
import numpy as np
import open3d as o3d
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

from .tsdf import NvbloxPanoTSDF
from .perception_base import BasePerceptionExtractor
from .frames import FrameInput, WorldFrame, ProcessingMode
from .coloring import BlockColorCache

from .utils.geometry import register_camera_poses_sim3, rotation_aligning_vectors
from .utils.metricfication import estimate_metric_scale_from_floor
from .utils.profiler import VRAMProfiler

class StreamingWindowEngine:
    def __init__(self, perception: BasePerceptionExtractor, voxel_size: float = 0.02,
                 max_depth: float = 4.5, face_size: int = 512, crop_margin: int = 24,
                 device: str = "cuda"):
        self.perception = perception
        self.device = device

        self.max_depth = max_depth
        self.voxel_size = voxel_size
        # NOTE: camera height is no longer an engine setting -- each FrameInput carries
        # its own instantaneous camera_height, used for that window's metric scaling.

        # --- Cubemap reprojection resolution -----------------------------------
        # The equirectangular panorama can't be fed to nvblox's pinhole projective
        # integrator directly, so tsdf.py reprojects the 360 sphere onto 6 virtual
        # 90-deg-FOV pinhole faces. ``face_size`` is the pixel resolution of each
        # cube face; ``crop_margin`` trims the distorted/overlapping seam pixels at
        # each face edge.
        #
        # Sizing: the panorama's angular resolution is ~W/360 px/deg (e.g.
        # 1036/360 ~= 2.9 px/deg). A face of size F over 90 deg gives F/90 px/deg at
        # the face *center* (denser toward corners). Matching the pano at center
        # wants F ~= 2.9*90 ~= 260; F=512 keeps a ~2x safety margin so thin
        # structures near face edges survive. F=1024 is ~4x linear oversampling of a
        # 1036-wide pano -> pure wasted grid_sample + VRAM, no extra real detail.
        self.face_size = face_size
        self.crop_margin = crop_margin

        # Colorizer memory budget. Coloring streams over ALL keyframes (no data is
        # dropped), but processes them in bounded camera batches x point chunks so
        # VRAM no longer scales with sequence length. Tune down if you still OOM.
        self.color_cam_batch = 8
        self.color_point_chunk = 500_000
        # Coarse grid (multiples of voxel_size) used for the incremental color
        # cache: one cached color per block, propagated to all dense vertices in it.
        self.color_block_mult = 1.75

        # Dump nvblox's internal C++ stage timers every N submaps (0 = off). These
        # give the library's own per-stage breakdown (integration, meshing, ...).
        self.nvblox_timing_every = 0

        # --- Mesh extraction cadence ------------------------------------------
        # Depth is integrated every submap, but pulling + rebuilding the full
        # Open3D mesh scales with TOTAL map size and dominates once the map grows.
        # Only rebuild the display mesh every N submaps; nvblox's incremental
        # marching cubes catches up on the next extraction. 1 = every submap.
        self.mesh_extract_every = 1

        # --- Point-cloud-only streaming ---------------------------------------
        # The downstream client consumes the point cloud, not the triangle mesh, and
        # building a full Open3D TriangleMesh every submap means ~6 Vector3dVector
        # copies of the whole growing mesh. When True we skip the mesh entirely on
        # the per-submap path: compute vertex normals on the GPU, color the vertices,
        # and emit just the point cloud (2 copies). The triangle mesh / .glb is still
        # built once at sequence end. Set False to restore the per-submap mesh.
        self.point_cloud_only = True

        # --- World representation ---------------------------------------------
        # Single knob for how the output map is framed. All options are Z-up,
        # right-handed (ROS REP-103 / nvblox). It controls only where the origin sits:
        #   "floor"  -> Z=0 on the detected floor (camera ends up at ~+camera height)
        #   "camera" -> the first camera is the origin (0,0,0), as the legacy default
        self.world_frame: WorldFrame = "floor"

        # --- ESDF (Euclidean Signed Distance Field) ---------------------------
        # When True, the ESDF is recomputed each submap so callers can query collision
        # distances (get_esdf_slice / tsdf.query_esdf) for planning. Cheap in practice.
        self.compute_esdf = True
        self.esdf_slice_resolution = 0.05  # meters between ESDF query samples

        # --- nvblox VRAM bounding ---------------------------------------------
        # The nvblox TSDF volume grows with explored area; on a streaming run it
        # eventually exhausts VRAM when its block hash doubles. To keep ALL the
        # geometry while capping GPU memory, we periodically "flush": extract +
        # color the current volume, append it to a persistent CPU accumulator, then
        # clear the GPU volume (and the per-window keyframe buffers). The full map
        # therefore lives on the CPU; the GPU only ever holds one flush-window.
        self.map_accumulate = False
        self.map_flush_every_n = 3          # flush after this many submaps...
        self.map_flush_min_free_gb = 3.0    # ...or sooner if free VRAM drops below this
        self.accum_downsample = True        # voxel-downsample the accumulated cloud (dedup overlaps)


        # Minimum |cos(normal . up)| required to trust a floor detection for the
        # one-time gravity leveling of the world frame.
        self.level_min_confidence = 0.4

        # --- Perception/mapping pipeline --------------------------------------
        # "parallel": the NEXT window's GPU inference runs on a background thread while
        # THIS window's mapping (integrate -> mesh -> color) runs on the main thread,
        # overlapping the two ~equal costs (A/B double buffer; no window is skipped).
        # "sequential": inference then mapping, one after the other (lower peak VRAM,
        # slower). See the process_sequence docstring for the buffering contract.
        self.processing_mode: ProcessingMode = "parallel"

        self.vram_tracker = VRAMProfiler()
        self.tsdf_executor = ThreadPoolExecutor(max_workers=1)
        self.tsdf_future = None
        self.perc_executor = ThreadPoolExecutor(max_workers=1)
        # Perception (VGGT forward) memo, keyed by a window's frame identities. The
        # network output is deterministic for the same frames, and consecutive batches
        # (especially reset mode, which re-runs the WHOLE window) re-request mostly the
        # same windows — so caching turns ~7 inferences/batch into ~1. Deliberately NOT
        # cleared in reset(), so a fresh rebuild reuses already-computed perception.
        # ~77 MB/window (12×518×1036×3 f32); tune PERC_CACHE_WINDOWS for RAM.
        self._perc_cache_max = int(os.environ.get("PERC_CACHE_WINDOWS", "16"))
        self._perc_cache = OrderedDict() if self._perc_cache_max > 0 else None
        self._perc_hits = 0
        self._perc_misses = 0
        self.perc_future = None
        self.tsdf = None

        print("[Engine] Initializing PRISM-VGGT Engine...")
        self.reset()
        
    def reset(self):
        if self.tsdf_future is not None:
            self.tsdf_future.result()
            self.tsdf_future = None
        if self.perc_future is not None:
            try:
                self.perc_future.result()
            except Exception:
                pass
            self.perc_future = None

        self.last_mesh = None
        self.last_pcd = None

        # Persistent CPU-side accumulation of the full colored map (see map flush).
        # accum_mesh keeps the full mesh; accum_pcd_cache is its (downsampled) point
        # cloud, rebuilt only on flush so per-submap display stays cheap.
        self.accum_mesh = o3d.geometry.TriangleMesh()
        self.accum_pcd_cache = o3d.geometry.PointCloud()
        self.last_flush_submap = 0

        self.tsdf = NvbloxPanoTSDF(
            voxel_size_m=self.voxel_size,
            max_depth=self.max_depth,
            face_size=self.face_size,
            crop_margin=self.crop_margin,
            device=self.device
        )
        
        self.prev_overlap_raw_pts = []
        self.prev_overlap_global_poses = []
        self.trajectory = []
        self.full_poses = []
        self._last_kf_pose = None        # keyframe-gating: last integrated camera pose
        self._last_kf_t = None           # capture time of last integrated keyframe
        self.pose_timestamps = []      # parallel to full_poses (from FrameInput.timestamp)
        self.processed_indices = []
        self._done_starts = set()      # window-start indices already processed (online mode)

        # Block color cache + best-view colorizer + streaming API (see coloring.py).
        # Rebuilt here so per-run config (voxel_size, max_depth, ...) is picked up.
        self.colorizer = BlockColorCache(
            voxel_size=self.voxel_size,
            color_block_mult=self.color_block_mult,
            max_depth=self.max_depth,
            device=self.device,
            cam_batch=self.color_cam_batch,
            point_chunk=self.color_point_chunk,
        )

        self.submap_count = 0
        self.is_first_window = True
        self.current_metric_scale = 1.0

        # --- Robust initial metric-scale anchoring (scale warm-up) -------------
        # The metric scale is read from the floor on the FIRST window, but a single
        # window's floor RANSAC is noisy and the very first frames (camera settling,
        # an oblique/partial floor view) often give an OUTLIER scale. Locking the
        # whole map to that one estimate inflates/shrinks the entire trajectory:
        # the cameras end up at one scale while later windows' geometry sits at the
        # true (steady-state) scale, so walls/objects no longer register and the
        # live map "drifts"/ghosts as the robot moves (the offline gradio run only
        # looked clean because its curated first frames happened to anchor well).
        # Fix: for the first ``scale_warmup_windows`` confident windows, re-anchor
        # the scale to the running MEDIAN of their floor estimates instead of hard-
        # locking to window 0; once the median has stabilised it locks. Set
        # SCALE_WARMUP_WINDOWS=1 to restore the old lock-on-first-window behaviour.
        self.scale_warmup_windows = int(os.environ.get("SCALE_WARMUP_WINDOWS", "3"))
        self.floor_scale_samples = []
        self._scale_committed = False

        # Gravity / floor leveling. ``world_align`` is a one-time rotation applied to
        # the world frame so the detected floor becomes horizontal; everything fed to
        # nvblox is built in this leveled frame. ``last_floor_plane`` holds the most
        # recent detected plane expressed in (leveled) world coordinates, for display.
        self.world_align = np.eye(4)
        self.is_leveled = False
        self.last_floor_plane = None

    def _async_tsdf_task(self, depth_maps, rgb_frames, masks, poses):
        for j in range(len(poses)):
            self.tsdf.integrate(depth_maps[j], rgb_frames[j], masks[j], poses[j])
        torch.cuda.synchronize()

    def _timed_perception(self, window: List[FrameInput]) -> Dict[str, Any]:
        """Run perception on a window of FrameInputs and stash its true wall-clock
        cost in the result. Memoised by frame identity (see _perc_cache): the same
        window of frames never re-runs the VGGT forward (the reset-mode latency fix)."""
        cache = self._perc_cache
        key = None
        if cache is not None:
            key = tuple(int(round(float(getattr(f, "timestamp", i)) * 1e9))
                        for i, f in enumerate(window))
            hit = cache.get(key)
            if hit is not None:
                cache.move_to_end(key)
                self._perc_hits += 1
                out = dict(hit)
                out["_infer_time"] = 0.0      # served from cache → no inference cost
                return out
        t = time.time()
        preds = self.perception.process_sequence([f.image for f in window])
        preds["_infer_time"] = time.time() - t
        self._perc_misses += 1
        if cache is not None and key is not None:
            cache[key] = {k: v for k, v in preds.items() if k != "_infer_time"}
            while len(cache) > self._perc_cache_max:
                cache.popitem(last=False)
        return preds

    # --- Point-cloud streaming API (delegated to the BlockColorCache) ------- #
    def get_map_version(self) -> int:
        """Current monotonic map version (bumped once per extracted submap)."""
        return self.colorizer.get_map_version()

    def get_poses(self) -> Tuple[np.ndarray, np.ndarray]:
        """Timestamped camera poses, for handing off to a downstream client/SLAM.

        Returns ``(timestamps (N,), poses (N, 4, 4))`` where ``timestamps[i]`` is the
        ``FrameInput.timestamp`` of the frame that produced world pose ``poses[i]``.
        """
        ts = np.asarray(self.pose_timestamps)
        poses = np.asarray(self.full_poses) if len(self.full_poses) else np.zeros((0, 4, 4))
        return ts, poses

    def get_point_cloud_snapshot(self) -> Dict[str, Any]:
        """Full streamable point cloud (one point per color block) + a map_hash the
        client can compare against to detect drift and trigger a full resync.

        WARNING: this is the BlockColorCache, which only ever *adds/updates* blocks —
        it never removes them. So as the surface shifts between submaps it ACCUMULATES
        every block ever seen → thick/fuzzy/duplicated walls. It is fine for the
        versioned delta API, but DO NOT use it as the displayed map. Use
        :meth:`get_current_cloud` (the current TSDF surface, what gradio shows)."""
        return self.colorizer.get_point_cloud_snapshot()

    @staticmethod
    def _voxel_snap(xyz: np.ndarray, rgb: np.ndarray, voxel: float):
        """Collapse a cloud to ONE point per voxel, SNAPPED to the voxel-cell centre
        (colour = mean of the cell's points). This is what makes streaming scalable:
        marching-cubes vertices jitter sub-millimetre as nvblox re-meshes, so the raw
        surface changes everywhere every submap and the content-diff degenerates to a
        full resend (bandwidth ∝ total map size). Snapping to the fixed voxel grid
        makes UNCHANGED geometry byte-identical across submaps, so its block CRCs are
        stable and only the frontier cubes actually transfer (bandwidth ∝ what
        changed). It also dedups (smaller cloud) and is the true map resolution
        anyway."""
        if xyz.shape[0] == 0:
            return xyz.astype(np.float32), rgb.astype(np.uint8)
        v = float(voxel) if voxel and voxel > 0 else 0.03
        keys = np.floor(xyz / v).astype(np.int64)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        centers = ((uniq.astype(np.float64) + 0.5) * v).astype(np.float32)
        sums = np.zeros((uniq.shape[0], 3), dtype=np.float64)
        np.add.at(sums, inv, rgb.astype(np.float64))
        counts = np.bincount(inv, minlength=uniq.shape[0]).astype(np.float64)
        cols = np.clip(np.rint(sums / counts[:, None]), 0, 255).astype(np.uint8)
        return centers, cols

    def _keyframe_accept(self, global_pose, ts=None) -> bool:
        """True if this frame should be integrated. Accept on enough motion since the
        last keyframe (skips static "breathing"), OR if too long has passed since the
        last keyframe — the TIME ESCAPE: a (near-)static 360° robot must keep
        re-observing so DYNAMIC changes (a moved/new object) are integrated and decayed
        in, instead of the map freezing on first sight. Gating disabled → always True."""
        min_t = float(getattr(self, "keyframe_min_trans_m", 0.0) or 0.0)
        min_r = float(getattr(self, "keyframe_min_rot_deg", 0.0) or 0.0)
        if min_t <= 0.0 and min_r <= 0.0:
            return True
        prev = getattr(self, "_last_kf_pose", None)
        if prev is None:
            return True
        max_dt = float(getattr(self, "keyframe_max_interval_s", 0.0) or 0.0)
        if max_dt > 0.0 and ts is not None:
            last_t = getattr(self, "_last_kf_t", None)
            if last_t is None or (float(ts) - float(last_t)) >= max_dt:
                return True
        if min_t > 0.0 and float(np.linalg.norm(global_pose[:3, 3] - prev[:3, 3])) >= min_t:
            return True
        if min_r > 0.0:
            Rrel = prev[:3, :3].T @ global_pose[:3, :3]
            cos = (np.trace(Rrel) - 1.0) * 0.5
            if np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) >= min_r:
                return True
        return False

    def get_current_cloud(self) -> Dict[str, Any]:
        """The CURRENT nvblox TSDF surface as a point cloud, voxel-snapped to one
        point per cell — what the offline gradio path shows, made streaming-stable.
        Re-derived from the live volume each submap, so a shifted surface REPLACES the
        old geometry instead of layering on it (thin walls), and snapping to the voxel
        grid keeps unchanged cubes byte-identical so the diff stays bounded. Returns
        ``{"points": (N,3) float32, "colors": (N,3) uint8, "version": int}``."""
        pcd = self.last_pcd
        version = self.get_map_version()
        if pcd is None or len(pcd.points) == 0:
            return {"points": np.zeros((0, 3), np.float32),
                    "colors": np.zeros((0, 3), np.uint8), "version": version}
        xyz = np.asarray(pcd.points, dtype=np.float32)
        cols = np.asarray(pcd.colors, dtype=np.float64)
        rgb = np.clip(np.rint(cols * 255.0), 0, 255).astype(np.uint8)
        # Voxel-snap is OPT-IN (CLOUD_VOXEL_SNAP=1): it dedups/shrinks the cloud, but
        # binning marching-cubes vertices flickers cube membership at cell boundaries,
        # which can flip a (1 m) cube's diff-version even when nothing really changed.
        # nvblox's marching cubes is deterministic for an unchanged TSDF, so the RAW
        # vertices + a geometry-only block CRC are the more diff-stable default.
        if getattr(self, "cloud_voxel_snap", False) or os.environ.get("CLOUD_VOXEL_SNAP", "0") == "1":
            xyz, rgb = self._voxel_snap(xyz, rgb, self.voxel_size)
        return {"points": xyz, "colors": rgb, "version": version}

    def get_point_cloud_delta(self, since_version: int) -> Dict[str, Any]:
        """Only the blocks changed since ``since_version`` (the delta fast-path). On a
        drop/mismatch, re-request specific ``keys`` or fall back to the snapshot."""
        return self.colorizer.get_point_cloud_delta(since_version)

    def get_esdf_slice(self, height: Optional[float] = None,
                       bounds: Optional[Tuple[float, float, float, float]] = None,
                       resolution: Optional[float] = None, margin: float = 1.0
                       ) -> Optional[Dict[str, Any]]:
        """Sample the ESDF on a horizontal (constant-Z) grid, for viz/planning.

        World is Z-up (floor at Z=0), so a horizontal slice spans X-Y at a fixed Z.

        Args:
            height: world Z of the slice plane (meters above the floor). Defaults to
                ~half the trajectory's typical standing height.
            bounds: (x_min, x_max, y_min, y_max). Defaults to trajectory extent + margin.
            resolution: meters between samples (defaults to esdf_slice_resolution).
        Returns:
            dict {xs (Wx,), ys (Hy,), distance (Hy, Wx), z, valid} of ESDF distances
            in meters (unobserved = NaN), or None if ESDF isn't being computed / empty.
        """
        if not self.compute_esdf or len(self.trajectory) == 0:
            return None
        traj = np.asarray(self.trajectory)
        if bounds is None:
            x_min, x_max = traj[:, 0].min() - margin, traj[:, 0].max() + margin
            y_min, y_max = traj[:, 1].min() - margin, traj[:, 1].max() + margin
        else:
            x_min, x_max, y_min, y_max = bounds
        if height is None:
            height = float(np.median(traj[:, 2]))   # ~camera/standing height
        res = float(resolution or self.esdf_slice_resolution)
        xs = np.arange(x_min, x_max, res, dtype=np.float32)
        ys = np.arange(y_min, y_max, res, dtype=np.float32)
        if xs.size == 0 or ys.size == 0:
            return None
        gx, gy = np.meshgrid(xs, ys)               # (Hy, Wx)
        gz = np.full_like(gx, height)
        pts = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)
        q = torch.from_numpy(pts).to(self.device)
        try:
            dist = self.tsdf.query_esdf(q).cpu().numpy().reshape(gx.shape)
        except Exception as e:  # pragma: no cover - depends on nvblox build/state
            print(f"[Engine] ESDF query failed: {e}")
            return None
        valid = int(np.isfinite(dist).sum())
        print(f"[Engine] ESDF slice @ Z={height:.2f}m: {valid}/{dist.size} observed cells.")
        return {"xs": xs, "ys": ys, "distance": dist, "z": float(height), "valid": valid}

    @staticmethod
    def _free_vram_gb():
        if not torch.cuda.is_available():
            return float("inf")
        free, _ = torch.cuda.mem_get_info()
        return free / (1024 ** 3)

    def _color_geometry(self, geometry, rgbs=None, depths=None, masks=None, poses=None):
        """Colorize an already-extracted volume mesh in place (vertices = point
        cloud). Returns the colored mesh, or the input unchanged if empty.

        ``rgbs/depths/masks/poses`` are the CURRENT window's frames; pass none to
        color purely from the persistent cache (e.g. the final full-map extraction).
        """
        if geometry is None or len(geometry.vertices) == 0:
            return geometry
        if not geometry.has_vertex_normals():
            geometry.compute_vertex_normals()
        colors = self.colorizer.color_vertices(
            np.asarray(geometry.vertices),
            np.asarray(geometry.vertex_normals),
            rgbs or [], depths or [], masks or [], poses or []
        )
        geometry.vertex_colors = o3d.utility.Vector3dVector(colors)
        return geometry

    def _extract_and_color_current(self):
        """Extract the current nvblox volume and colorize it from the cache (used at
        sequence end, when there is no 'current window' left to project)."""
        return self._color_geometry(self.tsdf.extract_geometry())

    @staticmethod
    def _mesh_to_valid_pcd(mesh):
        """Point cloud of a colored mesh's vertices, keeping only colored ones."""
        pcd = o3d.geometry.PointCloud()
        if mesh is None or len(mesh.vertices) == 0:
            return pcd
        verts = np.asarray(mesh.vertices)
        cols = np.asarray(mesh.vertex_colors) if len(mesh.vertex_colors) else np.zeros_like(verts)
        valid = cols.sum(axis=1) > 0.0
        pcd.points = o3d.utility.Vector3dVector(verts[valid])
        pcd.colors = o3d.utility.Vector3dVector(cols[valid])
        return pcd

    def _flush_to_accumulator(self, colored_mesh):
        """Append a colored volume to the persistent CPU map, then free the GPU
        volume and the color cache so VRAM/host memory stay bounded."""
        if colored_mesh is not None and len(colored_mesh.vertices) > 0:
            self.accum_mesh += colored_mesh

        # Rebuild the cached accumulator point cloud (only happens on flush).
        self.accum_pcd_cache = self._mesh_to_valid_pcd(self.accum_mesh)
        if self.accum_downsample and self.voxel_size > 0 and len(self.accum_pcd_cache.points) > 0:
            self.accum_pcd_cache = self.accum_pcd_cache.voxel_down_sample(self.voxel_size)

        # Release GPU TSDF blocks. The flushed geometry is already colored into
        # accum_mesh, and the GPU volume is cleared, so its color cache can go too.
        self.tsdf.mapper.clear()
        self.colorizer.clear()
        self.last_flush_submap = self.submap_count + 1
        gc.collect()
        torch.cuda.empty_cache()

    def _pcd_from_arrays(self, v_np, colors_np):
        """Build the displayed point cloud from numpy vertices + colorizer colors,
        keeping only colored points and folding in the (downsampled) flush cache."""
        valid = colors_np.sum(axis=1) > 0.0
        pts, cols = [v_np[valid]], [colors_np[valid]]
        if len(self.accum_pcd_cache.points) > 0:
            pts.append(np.asarray(self.accum_pcd_cache.points))
            cols.append(np.asarray(self.accum_pcd_cache.colors))
        pcd = o3d.geometry.PointCloud()
        if any(len(p) for p in pts):
            pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts, axis=0))
            pcd.colors = o3d.utility.Vector3dVector(np.concatenate(cols, axis=0))
        return pcd

    def _build_display(self, current_mesh):
        """Cheap per-submap display: cached (frozen) accumulator cloud + the current
        live window. The live mesh is just the current window; the full map is always
        present in the point cloud."""
        display_mesh = current_mesh if (current_mesh is not None and len(current_mesh.vertices) > 0) else o3d.geometry.TriangleMesh()

        pts, cols = [], []
        if len(self.accum_pcd_cache.points) > 0:
            pts.append(np.asarray(self.accum_pcd_cache.points))
            cols.append(np.asarray(self.accum_pcd_cache.colors))
        cur_pcd = self._mesh_to_valid_pcd(current_mesh)
        if len(cur_pcd.points) > 0:
            pts.append(np.asarray(cur_pcd.points))
            cols.append(np.asarray(cur_pcd.colors))

        display_pcd = o3d.geometry.PointCloud()
        if pts:
            display_pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts, axis=0))
            display_pcd.colors = o3d.utility.Vector3dVector(np.concatenate(cols, axis=0))
        return display_mesh, display_pcd

    def process_sequence(self, frames: List[FrameInput], window_size: int = 16,
                         overlap: int = 4, generate_esdf: Optional[bool] = None,
                         reset: bool = True, finalize: bool = True
                         ) -> Iterator[Tuple[Any, Any, np.ndarray, Optional[dict]]]:
        """Stream a sequence in overlapping windows, yielding one update per submap.

        Args:
            frames: list of :class:`FrameInput` (image, mask, per-frame camera height,
                timestamp). The timestamp is attached to each output pose; the
                per-frame height is used for that window's metric scaling.
            window_size, overlap: sliding-window parameters.
            generate_esdf: if not None, overrides ``self.compute_esdf`` for this run
                (whether the ESDF is recomputed for every batch).

        Yields ``(display_mesh, display_pcd, trajectory, floor_plane)`` per submap,
        then a final full-map tuple. For streaming the point cloud to a client, use
        the version/hash accessors (get_point_cloud_snapshot / get_point_cloud_delta);
        for timestamped poses use get_poses().

        Parallelism (A/B double buffer)
        -------------------------------
        Two stages run concurrently, coupled so no window is ever skipped or
        processed out of order:

          * Stage A (producer): GPU perception inference, on ``perc_executor``.
          * Stage B (consumer): mapping (pose math -> TSDF integrate -> mesh ->
            color -> point cloud), on the main thread.

        The "buffer" is exactly one in-flight window: while Stage B consumes window
        ``k``, Stage A is already producing window ``k+1`` (``perc_future``). The
        loop blocks on ``perc_future.result()`` for window ``k`` *before* launching
        ``k+1``, and consumes windows strictly in ``starts`` order, so the producer
        can be at most one window ahead and never overruns or drops a window. Set
        ``processing_mode="sequential"`` to run the two stages one after another.
        """
        self.window_size = window_size
        self.overlap = overlap
        # reset=True (default): offline/batch — wipe the map and process everything.
        # reset=False: online/streaming — KEEP the accumulated map and process only
        # windows not yet seen, so repeated calls on a growing frame list extend one
        # persistent map (the nvblox TSDF, colorizer, poses and overlap-chain all
        # carry over because we don't reset them).
        if reset:
            self.reset()
        if generate_esdf is not None:
            self.compute_esdf = bool(generate_esdf)
        parallel = (self.processing_mode == "parallel")

        num_frames = len(frames)
        t_seq_start = time.time()

        all_starts = range(0, num_frames - self.window_size + 1, self.window_size - self.overlap)
        starts = [i for i in all_starts if i not in self._done_starts]   # only NEW windows

        # Prime the pipeline: launch the first window's inference in the background so
        # it is ready (or nearly) by the time the loop body asks for it.
        if parallel and starts:
            first = starts[0]
            self.perc_future = self.perc_executor.submit(
                self._timed_perception, frames[first : first + self.window_size]
            )

        for k, i in enumerate(starts):
            self.vram_tracker.start()
            t_win_start = time.time()
            profiler = {}

            window = frames[i : i + self.window_size]
            window_frames = [f.image for f in window]
            window_masks = [f.mask for f in window]
            window_heights = [f.camera_height for f in window]
            window_timestamps = [f.timestamp for f in window]

            print(f"\n==========================================")
            print(f"[Engine] Processing Submap {self.submap_count}...")

            t0 = time.time()
            if self.tsdf_future is not None:
                self.tsdf_future.result()
                self.tsdf_future = None
            profiler["TSDF_Sync"] = time.time() - t0

            t1 = time.time()
            if parallel:
                # Collect THIS window's inference (running in the background), then
                # immediately launch the NEXT window's inference so it overlaps all of
                # this submap's mapping work below. The reported time is the residual
                # WAIT, so it shrinks to ~0 whenever mapping is the longer leg.
                preds = self.perc_future.result()
                torch.cuda.synchronize()
                if k + 1 < len(starts):
                    nxt = starts[k + 1]
                    self.perc_future = self.perc_executor.submit(
                        self._timed_perception, frames[nxt : nxt + self.window_size]
                    )
                else:
                    self.perc_future = None
                profiler["Perception_Wait"] = time.time() - t1
                profiler["Perception_Infer(bg)"] = preds.pop("_infer_time", 0.0)
            else:
                preds = self._timed_perception(window)   # cached forward
                torch.cuda.synchronize()
                # Release the perception activations and return reserved blocks to the
                # driver so the nvblox C++ allocator has room to grow.
                torch.cuda.empty_cache()
                profiler["Perception_Inference"] = time.time() - t1

            pts_list, poses = preds["points"], preds["poses"]
            
            t2 = time.time()
            mid_idx = self.window_size // 2

            # Use raw unscaled points for RANSAC Metrification. The metric target is
            # this window's per-frame camera height (instantaneous measurement).
            mid_camera_height = window_heights[mid_idx]
            floor_scale, floor_conf, floor_plane = estimate_metric_scale_from_floor(
                pts_list[mid_idx],
                target_camera_height=mid_camera_height
            )
            
            # --- Native (unscaled) canonical poses: first cam at identity --------
            # All scale + placement now comes from a single Sim3 anchor estimated
            # from the overlap (below), so per-submap poses stay in VGGT native units.
            origin_inv = np.linalg.inv(poses[0])
            canonical_native = [origin_inv @ p for p in poses]

            # --- Sim3 anchor (s, R, t): native submap frame -> world meters ------
            # World convention: Z-up, right-handed (ROS REP-103 / nvblox standard).
            # The floor is placed at Z=0, so an upright camera sits at ~+camera height.
            if self.is_first_window:
                # First submap defines absolute metric scale from the floor (fallback
                # to a median-depth guess). The single mid-frame RANSAC is noisy and
                # is the estimate most likely to be a damaging OUTLIER, so anchor from
                # the MEDIAN floor scale across several frames of this first window
                # (one-time cost) rather than the mid frame alone.
                first_scales = []
                for fi in sorted(set([0, mid_idx, self.window_size // 4,
                                      3 * self.window_size // 4, self.window_size - 1])):
                    if 0 <= fi < self.window_size:
                        fs, fc, _fp = estimate_metric_scale_from_floor(
                            pts_list[fi], target_camera_height=window_heights[fi])
                        if fs is not None and fc >= self.level_min_confidence:
                            first_scales.append(float(fs))
                if first_scales:
                    s_anchor = float(np.median(first_scales))
                    self.floor_scale_samples.append(s_anchor)
                    if floor_scale is not None:
                        floor_scale = s_anchor   # keep leveling consistent with anchor
                elif floor_scale is not None:
                    s_anchor = floor_scale
                    if floor_conf >= self.level_min_confidence:
                        self.floor_scale_samples.append(float(floor_scale))
                else:
                    first_depth = np.linalg.norm(pts_list[0], axis=-1)
                    valid_depths = first_depth[first_depth > 0.1]
                    s_anchor = (3.0 / np.median(valid_depths)) if len(valid_depths) > 0 else 1.0

                # Gravity leveling (one-time, first confident floor): rotate so the
                # floor normal points to world +Z, then translate so the floor is at
                # Z=0. Baked into the anchor; later windows inherit it via the chain.
                if not self.is_leveled and floor_plane is not None and floor_conf >= self.level_min_confidence:
                    n_canonical = canonical_native[mid_idx][:3, :3] @ floor_plane["normal"]
                    R_level = rotation_aligning_vectors(n_canonical, np.array([0.0, 0.0, 1.0]))
                    self.world_align = np.eye(4)
                    self.world_align[:3, :3] = R_level
                    if self.world_frame == "floor":
                        # Translate so the detected floor sits on Z=0.
                        c_can = (canonical_native[mid_idx] @ np.append(floor_plane["centroid"], 1.0))[:3]
                        floor_z = float((R_level @ (s_anchor * c_can))[2])
                        self.world_align[2, 3] = -floor_z
                    self.is_leveled = True
                    print(f"  > [Leveling] Z-up world ({self.world_frame} origin) "
                          f"(conf={floor_conf:.2f}).")

                R_anchor = self.world_align[:3, :3].copy()
                t_anchor = self.world_align[:3, 3].copy()
            else:
                # Jointly estimate scale + rotation + translation from the overlap
                # cameras (proper Sim3). The world is already metric (inherited through
                # the chain), so the Umeyama scale lands directly in meters - no
                # damping / clamp heuristics needed.
                src_cam_np = np.stack(canonical_native[:self.overlap])
                tgt_cam_np = np.stack(self.prev_overlap_global_poses)
                s_est, R_anchor, t_anchor, src_ctr, tgt_ctr = register_camera_poses_sim3(src_cam_np, tgt_cam_np)
                if s_est is None:
                    # Degenerate baseline (camera barely moved across the overlap):
                    # keep the previous metric scale rather than inventing one.
                    s_est = self.current_metric_scale

                # Collect confident floor estimates until the scale is committed.
                if (not self._scale_committed and floor_scale is not None
                        and floor_conf >= self.level_min_confidence):
                    self.floor_scale_samples.append(float(floor_scale))

                if not self._scale_committed and len(self.floor_scale_samples) > 0:
                    # SCALE WARM-UP: re-anchor to the running MEDIAN of the first few
                    # confident floor estimates instead of trusting window 0 alone, so
                    # one bad first-window estimate can't inflate the whole map (which
                    # leaves the cameras and the geometry at different scales → the
                    # live-viewer ghosting/drift). The Umeyama rotation is
                    # scale-independent, so only the placement scale changes here.
                    s_anchor = float(np.median(self.floor_scale_samples))
                    if len(self.floor_scale_samples) >= self.scale_warmup_windows:
                        self._scale_committed = True
                        print(f"  > [Scale Lock] committed median s={s_anchor:.4f} "
                              f"over {len(self.floor_scale_samples)} windows")
                    else:
                        print(f"  > [Scale Warmup] {len(self.floor_scale_samples)}/"
                              f"{self.scale_warmup_windows}: median s={s_anchor:.4f} "
                              f"(this floor s={floor_scale:.4f})")
                elif os.environ.get("LOCK_SCALE_AFTER_FIRST", "1") == "1":
                    # Committed: every later window REUSES the locked scale and estimates
                    # only rotation+translation from the overlap. Leaving the overlap
                    # Sim3 free re-estimates scale each submap; even a ~1% drift
                    # compounds → the map inflates and submaps no longer fuse in the TSDF
                    # (ghosting/cloning + unbounded point growth). Set
                    # LOCK_SCALE_AFTER_FIRST=0 to restore the free-Sim3 behaviour.
                    s_anchor = self.current_metric_scale
                else:
                    if (os.environ.get("SCALE_TRACK_FLOOR", "0") == "1"
                            and floor_scale is not None and floor_conf > 0.4):
                        s_est = 0.9 * s_est + 0.1 * floor_scale
                    s_anchor = float(np.clip(s_est, 0.1, 5.0))
                # Keep translation consistent with the (locked or estimated) scale.
                t_anchor = tgt_ctr - s_anchor * (R_anchor @ src_ctr)

            self.current_metric_scale = s_anchor

            # ── per-submap anchor diagnostics ───────────────────────────────
            # Watch these across submaps to localise the online "cloning" drift:
            #   s ~1.000 stable  → scale is fine;  s creeping → scale drift.
            #   t_z / cam0_z creeping monotonically → vertical placement drift
            #   (matches a floor that climbs 0/10/20 cm per submap).
            #   floor_conf low/variable → unreliable floor → bad first anchor.
            if not self.is_first_window:
                _cam0z = float((s_anchor * (R_anchor @ canonical_native[0][:3, 3])
                                + t_anchor)[2])
                print(f"  > [Anchor] submap {self.submap_count}: s={s_anchor:.4f}  "
                      f"t_z={float(t_anchor[2]):+.3f}  cam0_z={_cam0z:+.3f}  "
                      f"floor_conf={floor_conf:.2f}")

            def _to_world(cn):
                gp = np.eye(4)
                gp[:3, :3] = R_anchor @ cn[:3, :3]
                gp[:3, 3] = s_anchor * (R_anchor @ cn[:3, 3]) + t_anchor
                return gp

            global_poses = [_to_world(cn) for cn in canonical_native]

            # ── Overlap pose pinning (online drift fix) ────────────────────────
            # In online mode (reset=False), each window's VGGT inference produces
            # slightly different predictions for the shared overlap frames. The
            # Sim3 registration absorbs this as a small translation residual that
            # accumulates monotonically across submaps (the "floor cloning" bug).
            # Fix: pin the overlap frames' global poses to EXACTLY the values
            # computed by the previous window. The Sim3 transform still guides
            # the new (non-overlap) frames correctly; only the anchors are locked.
            if not self.is_first_window and len(self.prev_overlap_global_poses) == self.overlap:
                # Diagnostic: measure the residual the Sim3 left on the overlap
                # frames before we pin them (should shrink to ~0 with a good fit).
                overlap_residuals = []
                for oi in range(self.overlap):
                    sim3_pos = global_poses[oi][:3, 3]
                    prev_pos = self.prev_overlap_global_poses[oi][:3, 3]
                    overlap_residuals.append(sim3_pos - prev_pos)
                resid_arr = np.array(overlap_residuals)
                mean_resid = resid_arr.mean(axis=0)
                max_resid = np.abs(resid_arr).max(axis=0)
                print(f"  > [Overlap Pin] mean residual xyz={mean_resid[0]:+.4f}, "
                      f"{mean_resid[1]:+.4f}, {mean_resid[2]:+.4f}  "
                      f"max |residual| xyz={max_resid[0]:.4f}, "
                      f"{max_resid[1]:.4f}, {max_resid[2]:.4f}")

                # Pin: overwrite overlap poses with the exact previous-window values
                for oi in range(self.overlap):
                    global_poses[oi] = self.prev_overlap_global_poses[oi].copy()

            # Record the detected floor plane in (leveled) world coordinates so the UI
            # can render the exact plane that was found this submap.
            if floor_plane is not None and floor_conf >= self.level_min_confidence:
                global_pose_mid = global_poses[mid_idx]
                centroid_metric = floor_plane["centroid"] * self.current_metric_scale
                extent_metric = floor_plane["extent"] * self.current_metric_scale
                normal_world = global_pose_mid[:3, :3] @ floor_plane["normal"]
                centroid_world = (global_pose_mid @ np.append(centroid_metric, 1.0))[:3]
                self.last_floor_plane = {
                    "normal": normal_world / (np.linalg.norm(normal_world) + 1e-12),
                    "centroid": centroid_world,
                    "extent": float(max(extent_metric, 0.5)),
                }

            # ── Depth-integration scale ────────────────────────────────────────
            # Integrate depth at the SINGLE committed metric scale — identical to the
            # offline gradio pipeline (which reconstructs cleanly). The camera height
            # only anchors metric scale on the first scene (the warm-up windows); from
            # then on PanoVGGT carries scale through the locked overlap-Sim3 chain, so
            # depth must be integrated at that one locked scale for ALL windows.
            #
            # DEPTH_SCALE_FROM_FLOOR=1 (opt-in) instead re-scales each window's depth
            # to that window's own floor estimate. This pins the floor exactly flat,
            # but because the per-window floor RANSAC has ~±4% noise it makes the
            # WALLS jitter in size window-to-window → wall ghosting. Only enable it if
            # you see a genuine MONOTONIC floor climb (native-scale drift across very
            # different scenes), not the static scatter the noise produces. Default
            # OFF so the live pipeline matches gradio exactly.
            depth_scale = self.current_metric_scale
            if (os.environ.get("DEPTH_SCALE_FROM_FLOOR", "0") == "1"
                    and not self.is_first_window
                    and floor_scale is not None
                    and floor_conf >= self.level_min_confidence):
                depth_scale = float(floor_scale)
                if abs(depth_scale - self.current_metric_scale) > 1e-6:
                    print(f"  > [Depth Scale] floor-anchored s={depth_scale:.4f} "
                          f"(pose-chain locked s={self.current_metric_scale:.4f}, "
                          f"Δ={100*(depth_scale/self.current_metric_scale - 1):+.2f}%)")

            batch_depths, batch_rgbs, batch_masks, batch_poses = [], [], [], []
            start_idx = 0 if self.is_first_window else self.overlap

            for j in range(self.window_size):
                global_pose = global_poses[j]

                if j >= start_idx:
                    # Low-frequency capture (2-3 Hz): keep EVERY frame. No spatial
                    # decimation - every observation is integrated so we retain as
                    # much geometry as possible.
                    current_pos = global_pose[:3, 3]

                    self.trajectory.append(current_pos)
                    self.processed_indices.append(i + j)
                    self.full_poses.append(global_pose)
                    self.pose_timestamps.append(window_timestamps[j])

                    # ── Keyframe gating (online ghost/breathing/bandwidth fix) ──
                    # Integrate only if the camera moved enough vs the last keyframe.
                    # Pose bookkeeping above stays unconditional so the trajectory and
                    # pose-correction chain are intact even on skipped frames.
                    if self._keyframe_accept(global_pose, window_timestamps[j]):
                        tsdf_pose = global_pose.copy()

                        # Cubemap upside-down guard. The camera's down axis (col 1,
                        # image +Y) should point toward world -Z; if it points +Z the
                        # camera is flipped, so rotate 180 deg about its X.
                        if tsdf_pose[2, 1] > 0:
                            tsdf_pose[:3, :3] = tsdf_pose[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

                        scaled_pts = pts_list[j] * depth_scale
                        depth_map = np.nan_to_num(np.linalg.norm(scaled_pts, axis=-1), nan=0.0, posinf=0.0, neginf=0.0)

                        batch_depths.append(depth_map)
                        batch_rgbs.append(window_frames[j])
                        batch_masks.append(window_masks[j])
                        batch_poses.append(tsdf_pose)
                        self._last_kf_pose = global_pose.copy()
                        self._last_kf_t = window_timestamps[j]

                if j >= self.window_size - self.overlap:
                    if j == self.window_size - self.overlap:
                        self.prev_overlap_global_poses = []
                    # Save the Sim3-computed (NOT pinned) global pose for the tail
                    # overlap frames. These are computed from _to_world() on the
                    # new (non-overlap) part of the window, so they carry the
                    # current window's honest transform. The NEXT window will pin
                    # its head to these exact values, preventing drift accumulation.
                    self.prev_overlap_global_poses.append(global_pose)
            
            self.prev_overlap_raw_pts = pts_list[-self.overlap:]
            self.is_first_window = False
            profiler["Scale_&_Pose_Math"] = time.time() - t2

            t3 = time.time()
            if len(batch_poses) > 0:
                print(f"  > [TSDF] Sending {len(batch_poses)} Keyframes to C++ Background Mapper...")
                self.tsdf_future = self.tsdf_executor.submit(
                    self._async_tsdf_task, batch_depths, batch_rgbs, batch_masks, batch_poses
                )
            profiler["Nvblox_Enqueue"] = time.time() - t3
            
            # Wait for the background mapper, then pull the geometry. The point cloud
            # is just the vertices of this structure (connectivity is kept only for
            # normals / optional .glb export). Profiled on its own.
            t_integ = time.time()
            if self.tsdf_future is not None:
                self.tsdf_future.result()
                self.tsdf_future = None
            profiler["TSDF_Integrate"] = time.time() - t_integ

            # TSDF prune (navigation local map): clear the nvblox volume outside a sphere
            # around the robot, so the ESDF only reflects a bounded, recent area AND the
            # mesh-pull cost (Mesh_GetHandles, the live-latency driver) stays constant as
            # the robot travels. Supersedes decay when on. tsdf_prune_radius<=0 → off, i.e.
            # full accumulation for SoTA benchmarks (set by the caller).
            pruned = False
            prune_r = float(getattr(self, "tsdf_prune_radius", 0.0) or 0.0)
            if prune_r > 0 and len(batch_poses) > 0:
                try:
                    center = np.asarray(batch_poses[-1], dtype=np.float32)[:3, 3]
                    self.tsdf.prune_outside_radius(center, prune_r)
                    pruned = True
                except Exception as _e:
                    if not getattr(self, "_prune_warned", False):
                        print(f"  > [Prune] nvblox TSDF prune unavailable, skipping: {_e}")
                        self._prune_warned = True

            # Optional active carving of stale voxels via nvblox decay (guarded;
            # off unless enabled, and no-ops loudly-once if the API isn't present).
            if not pruned and getattr(self, "tsdf_decay", False) and len(batch_poses) > 0:
                self._decay_count = getattr(self, "_decay_count", 0) + 1
                if self._decay_count % max(1, int(getattr(self, "decay_every_n", 1))) == 0:
                    try:
                        self.tsdf.decay()
                    except Exception as _e:
                        if not getattr(self, "_decay_warned", False):
                            print(f"  > [Decay] nvblox decay unavailable, skipping: {_e}")
                            self._decay_warned = True

            # --- ESDF (optional, off by default) --------------------------------
            if self.compute_esdf:
                t_esdf = time.time()
                self.tsdf.update_esdf()
                profiler["ESDF_Update"] = time.time() - t_esdf

            # --- Geometry extraction (gated by cadence) --------------------------
            # The costly part (pulling + rebuilding the full Open3D mesh) scales with
            # TOTAL map size, so only do it every ``mesh_extract_every`` submaps.
            # Depth was already integrated this submap (above); nvblox re-meshes every
            # block dirtied since the last update on the next extraction.
            # Skip extraction when nothing was integrated this window (fully gated /
            # static): re-coloring with an empty batch would blank the cloud, and
            # re-meshing an unchanged volume just wastes time.
            do_extract = (len(batch_poses) > 0
                          and self.submap_count % max(1, int(self.mesh_extract_every)) == 0)
            if do_extract:
                # Marching cubes is incremental in nvblox and shared by both paths.
                t0 = time.time()
                self.tsdf.update_mesh()
                torch.cuda.synchronize()
                profiler["Mesh_MarchingCubes"] = time.time() - t0

                # New map version for this extraction; blocks the colorizer touches
                # below are stamped with it for delta streaming.
                self.colorizer.begin_submap()

                if self.point_cloud_only:
                    # Pull only vertices + triangles, derive normals on the GPU, color
                    # the vertices, and emit just the point cloud (no TriangleMesh).
                    t0 = time.time()
                    cmesh = self.tsdf.get_color_mesh_raw()
                    v_t, tri_t = cmesh.vertices(), cmesh.triangles()
                    torch.cuda.synchronize()
                    profiler["Mesh_GetHandles"] = time.time() - t0

                    t0 = time.time()
                    normals_t = self.colorizer.vertex_normals_gpu(v_t, tri_t)
                    torch.cuda.synchronize()
                    profiler["Mesh_Normals"] = time.time() - t0

                    t0 = time.time()
                    v_np = v_t.cpu().numpy()
                    n_np = normals_t.cpu().numpy()
                    profiler["Mesh_GPU2CPU"] = time.time() - t0

                    t_color = time.time()
                    colors_np = self.colorizer.color_vertices(
                        v_np, n_np, batch_rgbs, batch_depths, batch_masks, batch_poses
                    )
                    profiler["Coloring"] = time.time() - t_color

                    t0 = time.time()
                    display_pcd = self._pcd_from_arrays(v_np, colors_np)
                    profiler["PCD_Build"] = time.time() - t0

                    self.last_pcd = display_pcd
                    self.last_mesh = None
                    display_mesh = o3d.geometry.TriangleMesh()
                else:
                    # Full mesh path (kept for .glb-per-submap / A-B testing).
                    t0 = time.time()
                    cmesh = self.tsdf.get_color_mesh_raw()
                    v_t, c_t, tri_t = cmesh.vertices(), cmesh.vertex_colors(), cmesh.triangles()
                    torch.cuda.synchronize()
                    profiler["Mesh_GetHandles"] = time.time() - t0

                    t0 = time.time()
                    v_np = v_t.cpu().numpy()
                    c_np = (c_t.to(torch.float64) / 255.0).cpu().numpy()
                    tri_np = tri_t.cpu().numpy()
                    profiler["Mesh_GPU2CPU"] = time.time() - t0

                    t0 = time.time()
                    current_geometry = o3d.geometry.TriangleMesh()
                    current_geometry.vertices = o3d.utility.Vector3dVector(v_np)
                    current_geometry.vertex_colors = o3d.utility.Vector3dVector(c_np)
                    current_geometry.triangles = o3d.utility.Vector3iVector(tri_np)
                    profiler["Mesh_O3D_Build"] = time.time() - t0

                    t0 = time.time()
                    current_geometry.compute_vertex_normals()
                    profiler["Mesh_Normals"] = time.time() - t0

                    t_color = time.time()
                    self.last_mesh = self._color_geometry(
                        current_geometry, batch_rgbs, batch_depths, batch_masks, batch_poses
                    )
                    profiler["Coloring"] = time.time() - t_color

                    # --- Flush the GPU volume to the CPU accumulator if needed --
                    free_gb = self._free_vram_gb()
                    submaps_since_flush = (self.submap_count + 1) - self.last_flush_submap
                    need_flush = self.map_accumulate and (
                        submaps_since_flush >= self.map_flush_every_n or free_gb < self.map_flush_min_free_gb
                    )
                    if need_flush:
                        self._flush_to_accumulator(self.last_mesh)
                        current_for_display = None
                        print(f"  > [Map] Flushed volume to CPU accumulator (free VRAM was {free_gb:.2f} GB).")
                    else:
                        current_for_display = self.last_mesh

                    display_mesh, display_pcd = self._build_display(current_for_display)
                    self.last_pcd = display_pcd
            else:
                # Cadence skip: reuse the last extracted/colored geometry. Depth is
                # still integrated; the displayed map refreshes on the next extract.
                profiler["Mesh_Skipped(cadence)"] = 0.0
                display_mesh = self.last_mesh if (self.last_mesh is not None and len(self.last_mesh.vertices) > 0) else o3d.geometry.TriangleMesh()
                display_pcd = self.last_pcd if self.last_pcd is not None else o3d.geometry.PointCloud()

            gc.collect()
            torch.cuda.empty_cache()

            pt_alloc, pt_res, sys_used, sys_total, sys_peak = self.vram_tracker.stop()
            # nvblox lives outside PyTorch's allocator. Estimate it from the
            # *instantaneous* system usage minus PyTorch's *current* reserved (NOT the
            # peak pt_res, which can exceed live usage once perception frees back).
            cur_res = torch.cuda.memory_reserved(0) / (1024 ** 3) if torch.cuda.is_available() else 0.0
            nvblox_other = max(sys_used - cur_res, 0.0)
            print("  --- ⏱️ Process Timing (Seconds) ---")
            for k, v in profiler.items():
                print(f"    - {k:<20}: {v:.3f}s")
            mesh_total = sum(v for k, v in profiler.items() if k.startswith("Mesh_"))
            print(f"    - [Mesh subtotal]     : {mesh_total:.3f}s")
            print(f"    = Total Submap Time   : {(time.time() - t_win_start):.3f}s")

            if self.nvblox_timing_every and (self.submap_count % self.nvblox_timing_every == 0):
                print("  --- 🧱 nvblox internal timers ---")
                self.tsdf.print_nvblox_timing()

            n_blocks = self.tsdf.num_tsdf_blocks()
            free_now = self._free_vram_gb()
            print("  --- 💾 GPU Memory (VRAM) ---")
            print(f"    - PyTorch Peak Alloc  : {pt_alloc:.2f} GB")
            print(f"    - PyTorch Reserved    : {pt_res:.2f} GB")
            print(f"    - nvblox / Other      : {nvblox_other:.2f} GB")
            print(f"    - System VRAM Usage   : {sys_used:.2f} GB / {sys_total:.2f} GB  (free {free_now:.2f} GB)")
            print(f"    - System VRAM Peak    : {sys_peak:.2f} GB  (sampled during submap)")
            print(f"    - nvblox TSDF Blocks  : {n_blocks}")
            # nvblox grows its block hash in power-of-two steps; a resize transiently
            # allocates the NEW buffer before freeing the old (~2x spike) ON TOP of
            # resident perception. Warn when the next resize could exceed free VRAM.
            if n_blocks > 0:
                next_cap = 1 << int(np.ceil(np.log2(max(n_blocks, 1))))
                if free_now < (pt_res + 2.0) and n_blocks > 0.6 * next_cap:
                    print(f"    ⚠️  approaching a hash resize (~{next_cap} cap) with only "
                          f"{free_now:.1f} GB free - risk of a transient-spike OOM. "
                          f"Lower max_depth/voxel or set processing_mode='sequential'.")
            self.submap_count += 1
            self._done_starts.add(i)

            yield display_mesh, display_pcd, np.array(self.trajectory), self.last_floor_plane

        if self.tsdf_future is not None:
            self.tsdf_future.result()

        if not finalize:
            return    # online/streaming caller emits per-submap output itself

        print("\n==========================================")
        print(f"[Engine] Sequence Complete ({(time.time() - t_seq_start):.2f}s). Extracting Unified Global Point Cloud...")

        # Fold any geometry still on the GPU into the accumulator, then the full map
        # is simply the CPU accumulator.
        tail_mesh = self._extract_and_color_current()
        if tail_mesh is not None and len(tail_mesh.vertices) > 0:
            self.accum_mesh += tail_mesh

        final_mesh = self.accum_mesh
        final_pcd = o3d.geometry.PointCloud()

        if final_mesh is not None and len(final_mesh.vertices) > 0:
            final_colors = np.asarray(final_mesh.vertex_colors) if len(final_mesh.vertex_colors) else np.zeros((len(final_mesh.vertices), 3))
            
            valid_color_mask = final_colors.sum(axis=1) > 0.0
            final_pcd.points = o3d.utility.Vector3dVector(np.asarray(final_mesh.vertices)[valid_color_mask])
            final_pcd.colors = o3d.utility.Vector3dVector(final_colors[valid_color_mask])

        yield final_mesh, final_pcd, np.array(self.trajectory), self.last_floor_plane