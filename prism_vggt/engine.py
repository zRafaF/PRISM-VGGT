import gc
import time
import hashlib
import torch
import numpy as np
import open3d as o3d
from concurrent.futures import ThreadPoolExecutor

from .tsdf import NvbloxPanoTSDF
from .perception_base import BasePerceptionExtractor

from .utils.geometry import register_camera_poses_sim3, rotation_aligning_vectors
from .utils.metricfication import estimate_metric_scale_from_floor
from .utils.profiler import VRAMProfiler

class StreamingWindowEngine:
    def __init__(self, perception: BasePerceptionExtractor, voxel_size=0.02, max_depth=4.5, target_camera_height=1.5,
                 face_size=512, crop_margin=24, device="cuda"):
        self.perception = perception
        self.device = device

        self.max_depth = max_depth
        self.voxel_size = voxel_size
        self.target_camera_height = target_camera_height

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
        self.color_block_mult = 2.5

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
        self.world_frame = "floor"

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
        # When enabled, the NEXT window's GPU inference is launched on a background
        # thread while THIS window's mapping (integrate -> mesh -> color) runs on the
        # main thread, overlapping the two ~equal costs. No data is skipped: every
        # window is still fully integrated and meshed (unlike mesh_extract_every).
        self.pipeline_inference = True

        self.vram_tracker = VRAMProfiler()
        self.tsdf_executor = ThreadPoolExecutor(max_workers=1)
        self.tsdf_future = None
        self.perc_executor = ThreadPoolExecutor(max_workers=1)
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
        self.processed_indices = []

        # --- Persistent block color cache (incremental coloring) ---------------
        # Coloring no longer re-projects the whole map against every keyframe each
        # submap. Instead we keep a sparse cache of best-view colors keyed by a
        # coarse "color block" grid (voxel_size * COLOR_BLOCK_MULT). Each submap we
        # only (re)project the blocks near the CURRENT window's cameras against THAT
        # window's frames; all other blocks reuse their cached color. This bounds
        # per-submap coloring cost to the live region (flat over time) and keeps the
        # currently-viewed area refreshed for dynamic obstacles.
        self._cache_packed = np.empty((0,), dtype=np.int64)   # sorted block keys
        self._cache_color = np.empty((0, 3), dtype=np.float64) # rgb in [0, 1]
        self._cache_score = np.empty((0,), dtype=np.float64)   # best-view score
        self._cache_version = np.empty((0,), dtype=np.int64)   # map_version a block last changed at

        # Monotonic map version for delta streaming: bumped once per submap that
        # touches the map; each block records the version it was last modified at,
        # so a client can request "everything changed since version N".
        self.map_version = 0

        self.submap_count = 0
        self.is_first_window = True
        self.current_metric_scale = 1.0

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

    def _timed_perception(self, window_frames):
        """Run perception and stash its true wall-clock cost in the result, so the
        (otherwise hidden, background) inference time is visible in the profiler."""
        t = time.time()
        preds = self.perception.process_sequence(window_frames)
        preds["_infer_time"] = time.time() - t
        return preds

    # --- Color block key packing -------------------------------------------
    # Pack an integer (x, y, z) block index into a single int64 so blocks can be
    # cached/looked-up with vectorised numpy set operations. Offset keeps negative
    # indices non-negative; range is +/- 2^19 blocks (~26 km at 5 cm blocks).
    _KEY_OFFSET = 1 << 19
    _KEY_STRIDE = 1 << 20

    def _block_size(self):
        return self.voxel_size * self.color_block_mult

    def _pack_keys(self, keys_xyz):
        k = keys_xyz.astype(np.int64) + self._KEY_OFFSET
        s = self._KEY_STRIDE
        return (k[:, 0] * s + k[:, 1]) * s + k[:, 2]

    @torch.no_grad()
    def _project_blocks(self, pts_np, nrm_np, rgbs, depths, masks, poses):
        """Best-view color + score for a (small, bounded) set of block reps.

        This is the original PyTorch projection math, but now run only over the
        blocks near the current window and only against the current window's frames.
        Returns (colors [M,3] in [0,1], scores [M,]) as numpy arrays.
        """
        device = self.device
        M = pts_np.shape[0]
        num_cams = len(rgbs)
        if M == 0 or num_cams == 0:
            return np.zeros((M, 3)), np.full((M,), -1.0)

        vertices = torch.from_numpy(pts_np).float().to(device)
        normals = torch.from_numpy(nrm_np).float().to(device)
        final_colors = torch.zeros((M, 3), device=device)
        best_score = torch.full((M,), -1.0, device=device)

        rgbs_all = np.stack(rgbs)
        depths_all = np.stack(depths)
        masks_all = np.stack(masks)
        poses_all = np.stack(poses)

        CAM_BATCH = max(1, int(self.color_cam_batch))
        CHUNK_SIZE = max(1, int(self.color_point_chunk))

        for cam_start in range(0, num_cams, CAM_BATCH):
            cam_end = min(cam_start + CAM_BATCH, num_cams)

            imgs_t = torch.from_numpy(rgbs_all[cam_start:cam_end]).float().to(device) / 255.0
            imgs_t = imgs_t.permute(0, 3, 1, 2)
            depths_t = torch.from_numpy(depths_all[cam_start:cam_end]).float().to(device).unsqueeze(1)
            masks_t = torch.from_numpy(masks_all[cam_start:cam_end]).float().to(device).unsqueeze(1)
            poses_t = torch.from_numpy(poses_all[cam_start:cam_end]).float().to(device)

            R_w_c = poses_t[:, :3, :3]
            t_w_c = poses_t[:, :3, 3]

            for start_idx in range(0, M, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, M)
                v_chunk = vertices[start_idx:end_idx]
                n_chunk = normals[start_idx:end_idx]
                C_size = v_chunk.shape[0]

                diff = v_chunk.unsqueeze(0) - t_w_c.unsqueeze(1)
                V_c = torch.bmm(diff, R_w_c)

                X, Y, Z = V_c[..., 0], V_c[..., 1], V_c[..., 2]
                dist = torch.sqrt(X**2 + Y**2 + Z**2)

                u_norm = torch.atan2(X, Z) / torch.pi
                v_norm = torch.asin(torch.clamp(Y / (dist + 1e-6), -1.0, 1.0)) / (torch.pi / 2.0)
                grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(2)

                sampled_colors = torch.nn.functional.grid_sample(imgs_t, grid, mode='bilinear', align_corners=False).squeeze(3)
                sampled_depths = torch.nn.functional.grid_sample(depths_t, grid, mode='nearest', align_corners=False).squeeze(3).squeeze(1)
                sampled_masks = torch.nn.functional.grid_sample(masks_t, grid, mode='nearest', align_corners=False).squeeze(3).squeeze(1)

                is_visible = dist <= (sampled_depths + 0.15)
                view_dirs_w = -diff / (dist.unsqueeze(-1) + 1e-6)
                dot_prod = (view_dirs_w * n_chunk.unsqueeze(0)).sum(dim=-1)

                valid_condition = (dist > 0.1) & (sampled_masks > 0.5) & is_visible & (dot_prod > 0) & (sampled_colors.sum(dim=1) > 0.15)

                score = torch.where(
                    valid_condition,
                    (dot_prod * torch.cos(v_norm * (torch.pi / 2.0))) / (dist**2 + 1e-6),
                    torch.tensor(-1.0, device=device)
                )

                batch_max, batch_best = torch.max(score, dim=0)

                update_mask = batch_max > best_score[start_idx:end_idx]
                if update_mask.any():
                    arange = torch.arange(C_size, device=device)
                    chosen_colors = sampled_colors[batch_best, :, arange]
                    final_colors[start_idx:end_idx][update_mask] = chosen_colors[update_mask]
                    best_score[start_idx:end_idx][update_mask] = batch_max[update_mask]

            del imgs_t, depths_t, masks_t, poses_t, R_w_c, t_w_c

        return final_colors.cpu().numpy(), best_score.cpu().numpy()

    def _cache_lookup(self, query_packed):
        """Return (colors [Q,3], scores [Q], found [Q]) for the queried block keys."""
        Q = query_packed.shape[0]
        colors = np.zeros((Q, 3), dtype=np.float64)
        scores = np.full((Q,), -1.0, dtype=np.float64)
        found = np.zeros((Q,), dtype=bool)
        if self._cache_packed.shape[0] == 0:
            return colors, scores, found
        pos = np.searchsorted(self._cache_packed, query_packed)
        pos = np.clip(pos, 0, self._cache_packed.shape[0] - 1)
        hit = self._cache_packed[pos] == query_packed
        colors[hit] = self._cache_color[pos[hit]]
        scores[hit] = self._cache_score[pos[hit]]
        found[hit] = True
        return colors, scores, found

    def _cache_update(self, keys_packed, colors, scores):
        """Merge improved block colors into the persistent cache (best-view wins).

        Every block that actually changes (a better-scoring view, or a brand-new
        block) is stamped with the current ``map_version`` so delta streaming can
        report exactly what moved since any prior version.
        """
        if keys_packed.shape[0] == 0:
            return
        if self._cache_packed.shape[0] > 0:
            pos = np.searchsorted(self._cache_packed, keys_packed)
            pos = np.clip(pos, 0, self._cache_packed.shape[0] - 1)
            exists = self._cache_packed[pos] == keys_packed
            # Update existing blocks where the new view scores better.
            ex_pos, ex_new = pos[exists], np.nonzero(exists)[0]
            better = scores[ex_new] > self._cache_score[ex_pos]
            upd = ex_pos[better]
            self._cache_color[upd] = colors[ex_new][better]
            self._cache_score[upd] = scores[ex_new][better]
            self._cache_version[upd] = self.map_version
            new = ~exists
        else:
            new = np.ones((keys_packed.shape[0],), dtype=bool)
        # Insert brand-new blocks, keeping the cache arrays sorted by key.
        if new.any():
            n_new = int(new.sum())
            merged_packed = np.concatenate([self._cache_packed, keys_packed[new]])
            merged_color = np.concatenate([self._cache_color, colors[new]])
            merged_score = np.concatenate([self._cache_score, scores[new]])
            merged_version = np.concatenate([
                self._cache_version, np.full((n_new,), self.map_version, dtype=np.int64)
            ])
            order = np.argsort(merged_packed, kind="stable")
            self._cache_packed = merged_packed[order]
            self._cache_color = merged_color[order]
            self._cache_score = merged_score[order]
            self._cache_version = merged_version[order]

    def _clear_color_cache(self):
        self._cache_packed = np.empty((0,), dtype=np.int64)
        self._cache_color = np.empty((0, 3), dtype=np.float64)
        self._cache_score = np.empty((0,), dtype=np.float64)
        self._cache_version = np.empty((0,), dtype=np.int64)

    # ------------------------------------------------------------------ #
    #  Point-cloud streaming API (block-granular, version + hash based).  #
    #  The streamable cloud is one point per color block (voxel_size *    #
    #  color_block_mult); this is the bounded, delta-able representation  #
    #  intended for a remote client. (The dense per-submap display cloud  #
    #  from process_sequence is a separate, local-viz product.)           #
    # ------------------------------------------------------------------ #
    def _unpack_keys(self, packed):
        """Inverse of _pack_keys: packed int64 -> (K, 3) int block indices."""
        s, off = self._KEY_STRIDE, self._KEY_OFFSET
        kz = (packed % s) - off
        rem = packed // s
        ky = (rem % s) - off
        kx = (rem // s) - off
        return np.stack([kx, ky, kz], axis=1)

    def _block_centers(self, packed):
        if packed.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return (self._unpack_keys(packed).astype(np.float64) + 0.5) * self._block_size()

    @staticmethod
    def _block_hashes(packed, colors):
        """Per-block content hash (uint64) over the block id + quantized color."""
        if packed.size == 0:
            return np.zeros((0,), dtype=np.uint64)
        col8 = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint64)
        col_packed = (col8[:, 0] << np.uint64(16)) | (col8[:, 1] << np.uint64(8)) | col8[:, 2]
        k = packed.astype(np.uint64)
        return (k * np.uint64(1099511628211)) ^ (col_packed * np.uint64(2654435761))

    def get_map_version(self):
        """Current monotonic map version (bumped once per extracted submap)."""
        return int(self.map_version)

    def _pack_cloud(self, packed, colors, since_version=None):
        pts = self._block_centers(packed).astype(np.float32)
        bh = self._block_hashes(packed, colors)
        map_hash = 0
        if packed.size:
            map_hash = int(hashlib.blake2b(
                np.ascontiguousarray(packed).tobytes() + np.ascontiguousarray(bh).tobytes(),
                digest_size=8).hexdigest(), 16)
        out = {
            "points": pts,                              # (K,3) float32 world coords
            "colors": np.ascontiguousarray(colors, dtype=np.float32),  # (K,3) rgb [0,1]
            "keys": packed.copy(),                      # (K,) int64 stable block ids
            "block_hashes": bh,                         # (K,) uint64 per-block hash
            "version": int(self.map_version),
            "map_hash": map_hash,                       # whole-cloud hash (drift check)
        }
        if since_version is not None:
            out["from_version"] = int(since_version)
        return out

    def get_point_cloud_snapshot(self):
        """Full streamable point cloud (one point per color block) + a map_hash the
        client can compare against to detect drift and trigger a full resync."""
        return self._pack_cloud(self._cache_packed, self._cache_color)

    def get_point_cloud_delta(self, since_version):
        """Only the blocks changed since ``since_version`` (the delta fast-path).

        Same fields as the snapshot. On a detected drop/mismatch the client should
        either re-request specific ``keys`` or fall back to get_point_cloud_snapshot().
        Note: block removal (e.g. via TSDF decay) is not yet tracked here.
        """
        if self._cache_packed.size == 0:
            return self._pack_cloud(self._cache_packed, self._cache_color, since_version)
        changed = self._cache_version > int(since_version)
        return self._pack_cloud(self._cache_packed[changed], self._cache_color[changed], since_version)

    def get_esdf_slice(self, height=None, bounds=None, resolution=None, margin=1.0):
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

    @torch.no_grad()
    def _apply_pytorch_colors(self, vertices_np, normals_np, batch_rgbs, batch_depths, batch_masks, batch_poses):
        """Incremental best-view colorization.

        Quantize every mesh vertex to a coarse color-block grid (our own grid, so a
        dense vertex and its block representative share the exact same key -> exact
        O(N) propagation, no KDTree). Only blocks near the CURRENT window's cameras
        are (re)projected, and only against THAT window's frames; all other blocks
        read their color straight from the persistent cache. Per-submap cost is thus
        bounded by the live region instead of growing with the whole map.
        """
        num_pts = vertices_np.shape[0]
        if num_pts == 0:
            return np.zeros((num_pts, 3))

        bsize = self._block_size()
        keys = np.floor(vertices_np / bsize).astype(np.int64)
        packed = self._pack_keys(keys)
        # np.unique returns (unique, index, inverse) in THAT order.
        uniq_packed, first_idx, inv = np.unique(packed, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        rep_xyz = vertices_np[first_idx]
        rep_nrm = normals_np[first_idx]

        # Start every block from its cached color (covers static, out-of-view areas).
        colors, scores, _ = self._cache_lookup(uniq_packed)

        num_cams = len(batch_rgbs)
        if num_cams > 0:
            cam_centers = np.stack([p[:3, 3] for p in batch_poses])
            nearest = np.linalg.norm(
                rep_xyz[:, None, :] - cam_centers[None, :, :], axis=2
            ).min(axis=1)
            active = nearest <= (self.max_depth + bsize)
            if active.any():
                a = np.nonzero(active)[0]
                new_cols, new_sco = self._project_blocks(
                    rep_xyz[a], rep_nrm[a], batch_rgbs, batch_depths, batch_masks, batch_poses
                )
                better = new_sco > scores[a]
                ab = a[better]
                colors[ab] = new_cols[better]
                scores[ab] = new_sco[better]
                # Persist improved live-region colors for future submaps.
                self._cache_update(uniq_packed[ab], new_cols[better], new_sco[better])

        return colors[inv]

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
        colors = self._apply_pytorch_colors(
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
        self._clear_color_cache()
        self.last_flush_submap = self.submap_count + 1
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def _vertex_normals_gpu(self, v_t, tri_t):
        """Area-weighted per-vertex normals computed on the GPU from the raw nvblox
        vertex/triangle tensors (replaces o3d ``compute_vertex_normals`` so we never
        need to build a TriangleMesh just to get normals for the colorizer)."""
        v = v_t.float()
        tri = tri_t.long()
        v0, v1, v2 = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
        # Cross product magnitude encodes triangle area -> area-weighted accumulation.
        face_n = torch.cross(v1 - v0, v2 - v0, dim=1)
        n = torch.zeros_like(v)
        n.index_add_(0, tri[:, 0], face_n)
        n.index_add_(0, tri[:, 1], face_n)
        n.index_add_(0, tri[:, 2], face_n)
        return torch.nn.functional.normalize(n, dim=1, eps=1e-8)

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

    def process_sequence(self, frames, masks, window_size=16, overlap=4):
        """Stream a sequence in overlapping windows, yielding one update per submap.

        Yields ``(display_mesh, display_pcd, trajectory, floor_plane)`` per submap,
        then a final full-map tuple. For streaming the point cloud to a client, use
        the version/hash accessors (get_point_cloud_snapshot / get_point_cloud_delta)
        rather than the dense display cloud.

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
        ``pipeline_inference=False`` to run the two stages sequentially instead.
        """
        self.window_size = window_size
        self.overlap = overlap
        self.reset()

        num_frames = len(frames)
        t_seq_start = time.time()
        
        starts = list(range(0, num_frames - self.window_size + 1, self.window_size - self.overlap))

        # Prime the pipeline: launch the first window's inference in the background so
        # it is ready (or nearly) by the time the loop body asks for it.
        if self.pipeline_inference and starts:
            first = starts[0]
            self.perc_future = self.perc_executor.submit(
                self._timed_perception, frames[first : first + self.window_size]
            )

        for k, i in enumerate(starts):
            self.vram_tracker.start()
            t_win_start = time.time()
            profiler = {}

            window_frames = frames[i : i + self.window_size]
            window_masks = masks[i : i + self.window_size]

            print(f"\n==========================================")
            print(f"[Engine] Processing Submap {self.submap_count}...")

            t0 = time.time()
            if self.tsdf_future is not None:
                self.tsdf_future.result()
                self.tsdf_future = None
            profiler["TSDF_Sync"] = time.time() - t0

            t1 = time.time()
            if self.pipeline_inference:
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
                preds = self.perception.process_sequence(window_frames)
                torch.cuda.synchronize()
                # Release the perception activations and return reserved blocks to the
                # driver so the nvblox C++ allocator has room to grow.
                torch.cuda.empty_cache()
                profiler["Perception_Inference"] = time.time() - t1

            pts_list, poses = preds["points"], preds["poses"]
            
            t2 = time.time()
            mid_idx = self.window_size // 2

            # Use raw unscaled points for RANSAC Metrification
            floor_scale, floor_conf, floor_plane = estimate_metric_scale_from_floor(
                pts_list[mid_idx],
                target_camera_height=self.target_camera_height
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
                # to a median-depth guess).
                if floor_scale is not None:
                    s_anchor = floor_scale
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
                # Light absolute pull toward the floor scale to curb slow drift.
                if floor_scale is not None and floor_conf > 0.4:
                    s_est = 0.9 * s_est + 0.1 * floor_scale
                s_anchor = float(np.clip(s_est, 0.1, 5.0))
                # Keep translation consistent with the final (blended/clipped) scale.
                t_anchor = tgt_ctr - s_anchor * (R_anchor @ src_ctr)

            self.current_metric_scale = s_anchor

            def _to_world(cn):
                gp = np.eye(4)
                gp[:3, :3] = R_anchor @ cn[:3, :3]
                gp[:3, 3] = s_anchor * (R_anchor @ cn[:3, 3]) + t_anchor
                return gp

            global_poses = [_to_world(cn) for cn in canonical_native]

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

                    tsdf_pose = global_pose.copy()

                    # Cubemap upside-down guard. The camera's down axis (col 1, image
                    # +Y) should point toward world -Z; if it points +Z the camera is
                    # flipped, so rotate 180 deg about its X to re-orient the faces.
                    if tsdf_pose[2, 1] > 0:
                        tsdf_pose[:3, :3] = tsdf_pose[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

                    scaled_pts = pts_list[j] * self.current_metric_scale
                    depth_map = np.nan_to_num(np.linalg.norm(scaled_pts, axis=-1), nan=0.0, posinf=0.0, neginf=0.0)

                    batch_depths.append(depth_map)
                    batch_rgbs.append(window_frames[j])
                    batch_masks.append(window_masks[j])
                    batch_poses.append(tsdf_pose)

                if j >= self.window_size - self.overlap:
                    if j == self.window_size - self.overlap:
                        self.prev_overlap_global_poses = []
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
            do_extract = (self.submap_count % max(1, int(self.mesh_extract_every)) == 0)
            if do_extract:
                # Marching cubes is incremental in nvblox and shared by both paths.
                t0 = time.time()
                self.tsdf.update_mesh()
                torch.cuda.synchronize()
                profiler["Mesh_MarchingCubes"] = time.time() - t0

                # New map version for this extraction; blocks the colorizer touches
                # below are stamped with it (see _cache_update) for delta streaming.
                self.map_version += 1

                if self.point_cloud_only:
                    # Pull only vertices + triangles, derive normals on the GPU, color
                    # the vertices, and emit just the point cloud (no TriangleMesh).
                    t0 = time.time()
                    cmesh = self.tsdf.get_color_mesh_raw()
                    v_t, tri_t = cmesh.vertices(), cmesh.triangles()
                    torch.cuda.synchronize()
                    profiler["Mesh_GetHandles"] = time.time() - t0

                    t0 = time.time()
                    normals_t = self._vertex_normals_gpu(v_t, tri_t)
                    torch.cuda.synchronize()
                    profiler["Mesh_Normals"] = time.time() - t0

                    t0 = time.time()
                    v_np = v_t.cpu().numpy()
                    n_np = normals_t.cpu().numpy()
                    profiler["Mesh_GPU2CPU"] = time.time() - t0

                    t_color = time.time()
                    colors_np = self._apply_pytorch_colors(
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

            pt_alloc, pt_res, sys_used, sys_total = self.vram_tracker.stop()
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
            print(f"    - nvblox TSDF Blocks  : {n_blocks}")
            # nvblox grows its block hash in power-of-two steps; a resize transiently
            # allocates the NEW buffer before freeing the old (~2x spike) ON TOP of
            # resident perception. Warn when the next resize could exceed free VRAM.
            if n_blocks > 0:
                next_cap = 1 << int(np.ceil(np.log2(max(n_blocks, 1))))
                if free_now < (pt_res + 2.0) and n_blocks > 0.6 * next_cap:
                    print(f"    ⚠️  approaching a hash resize (~{next_cap} cap) with only "
                          f"{free_now:.1f} GB free - risk of a transient-spike OOM. "
                          f"Lower max_depth/voxel or set pipeline_inference=False.")
            self.submap_count += 1

            yield display_mesh, display_pcd, np.array(self.trajectory), self.last_floor_plane

        if self.tsdf_future is not None:
            self.tsdf_future.result()

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