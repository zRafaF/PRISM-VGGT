import gc
import torch
import numpy as np
import open3d as o3d
import time
from concurrent.futures import ThreadPoolExecutor

from .tsdf import NvbloxPanoTSDF
from .perception_base import BasePerceptionExtractor

from .utils.alignment import align_cam_pts_irls
from .utils.geometry import register_camera_poses_kabsch, rotation_aligning_vectors
from .utils.metricfication import estimate_metric_scale_from_floor
from .utils.profiler import VRAMProfiler

class StreamingWindowEngine:
    def __init__(self, perception: BasePerceptionExtractor, voxel_size=0.02, max_depth=4.5, target_camera_height=1.5, device="cuda"):
        self.perception = perception
        self.device = device

        self.max_depth = max_depth
        self.voxel_size = voxel_size
        self.target_camera_height = target_camera_height

        # Colorizer memory budget. Coloring streams over ALL keyframes (no data is
        # dropped), but processes them in bounded camera batches x point chunks so
        # VRAM no longer scales with sequence length. Tune down if you still OOM.
        self.color_cam_batch = 8
        self.color_point_chunk = 500_000

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

        self.vram_tracker = VRAMProfiler()
        self.tsdf_executor = ThreadPoolExecutor(max_workers=1)
        self.tsdf_future = None
        self.tsdf = None
        
        print("[Engine] Initializing PRISM-VGGT Engine...")
        self.reset()
        
    def reset(self):
        if self.tsdf_future is not None:
            self.tsdf_future.result()
            self.tsdf_future = None

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
            crop_margin=24, 
            device=self.device
        )
        
        self.prev_overlap_raw_pts = []
        self.prev_overlap_global_poses = []
        self.trajectory = []
        self.full_poses = []
        self.processed_indices = []

        self.kf_rgbs, self.kf_depths, self.kf_masks, self.kf_poses = [], [], [], []
        
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

    @torch.no_grad()
    def _apply_pytorch_colors(self, vertices_np, normals_np, batch_rgbs, batch_depths, batch_masks, batch_poses):
        """Best-view colorization of the geometry vertices over ALL keyframes.

        For every vertex we keep the color from the keyframe that observes it with
        the highest quality score (frontal, close, well-sampled, unoccluded). To keep
        VRAM bounded regardless of how many keyframes accumulate, we stream over the
        cameras in batches and over the points in chunks, maintaining a running
        best-score per vertex. The result is bit-for-bit equivalent to scoring every
        camera at once - we just never materialize the full (num_cams x num_pts)
        tensor that previously caused the out-of-memory crash.
        """
        num_pts = vertices_np.shape[0]
        num_cams = len(batch_rgbs)
        if num_pts == 0 or num_cams == 0:
            return np.zeros((num_pts, 3))

        device = self.device

        vertices = torch.from_numpy(vertices_np).float().to(device)
        normals = torch.from_numpy(normals_np).float().to(device)
        final_colors = torch.zeros((num_pts, 3), device=device)
        best_score = torch.full((num_pts,), -1.0, device=device)

        rgbs_all = np.stack(batch_rgbs)
        depths_all = np.stack(batch_depths)
        masks_all = np.stack(batch_masks)
        poses_all = np.stack(batch_poses)

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

            for start_idx in range(0, num_pts, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, num_pts)
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

                # Best camera within THIS batch for each point.
                batch_max, batch_best = torch.max(score, dim=0)

                # Merge with the running best across previously processed batches.
                update_mask = batch_max > best_score[start_idx:end_idx]
                if update_mask.any():
                    arange = torch.arange(C_size, device=device)
                    chosen_colors = sampled_colors[batch_best, :, arange]  # (C_size, 3)
                    final_colors[start_idx:end_idx][update_mask] = chosen_colors[update_mask]
                    best_score[start_idx:end_idx][update_mask] = batch_max[update_mask]

            del imgs_t, depths_t, masks_t, poses_t, R_w_c, t_w_c

        return final_colors.cpu().numpy()

    @staticmethod
    def _free_vram_gb():
        if not torch.cuda.is_available():
            return float("inf")
        free, _ = torch.cuda.mem_get_info()
        return free / (1024 ** 3)

    def _color_geometry(self, geometry):
        """Colorize an already-extracted volume mesh in place (vertices = point
        cloud). Returns the colored mesh, or the input unchanged if empty."""
        if geometry is None or len(geometry.vertices) == 0 or len(self.kf_rgbs) == 0:
            return geometry
        geometry.compute_vertex_normals()
        colors = self._apply_pytorch_colors(
            np.asarray(geometry.vertices),
            np.asarray(geometry.vertex_normals),
            self.kf_rgbs, self.kf_depths, self.kf_masks, self.kf_poses
        )
        geometry.vertex_colors = o3d.utility.Vector3dVector(colors)
        return geometry

    def _extract_and_color_current(self):
        """Extract the current nvblox volume and colorize it (used at sequence end)."""
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
        volume and the per-window keyframe buffers so VRAM stays bounded."""
        if colored_mesh is not None and len(colored_mesh.vertices) > 0:
            self.accum_mesh += colored_mesh

        # Rebuild the cached accumulator point cloud (only happens on flush).
        self.accum_pcd_cache = self._mesh_to_valid_pcd(self.accum_mesh)
        if self.accum_downsample and self.voxel_size > 0 and len(self.accum_pcd_cache.points) > 0:
            self.accum_pcd_cache = self.accum_pcd_cache.voxel_down_sample(self.voxel_size)

        # Release GPU TSDF blocks and per-window keyframes.
        self.tsdf.mapper.clear()
        self.kf_rgbs, self.kf_depths, self.kf_masks, self.kf_poses = [], [], [], []
        self.last_flush_submap = self.submap_count + 1
        gc.collect()
        torch.cuda.empty_cache()

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
        self.window_size = window_size
        self.overlap = overlap
        self.reset()
        
        num_frames = len(frames)
        t_seq_start = time.time()
        
        for i in range(0, num_frames - self.window_size + 1, self.window_size - self.overlap):
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
            
            # --- FIX 2: Restored Scale Dampening ---
            if self.is_first_window:
                if floor_scale is not None:
                    self.current_metric_scale = floor_scale
                else:
                    first_depth = np.linalg.norm(pts_list[0], axis=-1)
                    valid_depths = first_depth[first_depth > 0.1]
                    if len(valid_depths) > 0:
                        self.current_metric_scale = 3.0 / np.median(valid_depths)
            else:
                raw_scale_diff = align_cam_pts_irls(
                    torch.from_numpy(pts_list[0].copy()), 
                    torch.from_numpy(self.prev_overlap_raw_pts[0].copy()), 
                    torch.from_numpy(window_masks[0].copy())
                )
                clipped_scale = np.clip(raw_scale_diff, 0.95, 1.05)
                
                # 80% memory / 20% new IRLS reading
                relative_scale = self.current_metric_scale * (0.8 * 1.0 + 0.2 * clipped_scale)
                
                if floor_scale is not None and floor_conf > 0.4:
                    absolute_floor_scale = floor_scale
                    # 90% running scale / 10% floor correction
                    self.current_metric_scale = 0.9 * relative_scale + 0.1 * absolute_floor_scale
                else:
                    self.current_metric_scale = relative_scale

            self.current_metric_scale = np.clip(self.current_metric_scale, 0.1, 5.0)

            metric_local_poses = []
            for p in poses:
                mp = p.copy()
                mp[:3, 3] *= self.current_metric_scale 
                metric_local_poses.append(mp)
                
            submap_origin_inv = np.linalg.inv(metric_local_poses[0])
            canonical_poses = [submap_origin_inv @ mp for mp in metric_local_poses]

            # --- Gravity leveling (one-time) ---------------------------------
            # On the first window with a confident floor, rotate the world frame so
            # the detected floor normal points along world "up" (-Y, OpenCV down
            # convention). This is baked into every pose below, so nvblox builds the
            # whole map level. Subsequent windows inherit it via Kabsch anchoring.
            if self.is_first_window and not self.is_leveled and floor_plane is not None and floor_conf >= self.level_min_confidence:
                n_local = floor_plane["normal"]
                n_canonical = canonical_poses[mid_idx][:3, :3] @ n_local
                R_level = rotation_aligning_vectors(n_canonical, np.array([0.0, -1.0, 0.0]))
                self.world_align = np.eye(4)
                self.world_align[:3, :3] = R_level
                self.is_leveled = True
                print(f"  > [Leveling] World frame aligned to floor (conf={floor_conf:.2f}).")

            # The leveling rotation only needs to be applied explicitly to the very
            # first window; later windows are anchored to already-leveled global poses.
            base = self.world_align if self.is_first_window else np.eye(4)

            anchor_pose = np.eye(4)
            if not self.is_first_window:
                src_cam_np = np.stack(canonical_poses[:self.overlap])
                tgt_cam_np = np.stack(self.prev_overlap_global_poses)
                R_align, t_align = register_camera_poses_kabsch(src_cam_np, tgt_cam_np, scale=1.0)
                anchor_pose[:3, :3] = R_align
                anchor_pose[:3, 3] = t_align

            # Record the detected floor plane in (leveled) world coordinates so the UI
            # can render the exact plane that was found this submap.
            if floor_plane is not None and floor_conf >= self.level_min_confidence:
                global_pose_mid = base @ anchor_pose @ canonical_poses[mid_idx]
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
                global_pose = base @ anchor_pose @ canonical_poses[j]

                if j >= start_idx:
                    # Low-frequency capture (2-3 Hz): keep EVERY frame. No spatial
                    # decimation - every observation is integrated so we retain as
                    # much geometry as possible.
                    current_pos = global_pose[:3, 3]

                    self.trajectory.append(current_pos)
                    self.processed_indices.append(i + j)
                    self.full_poses.append(global_pose)

                    tsdf_pose = global_pose.copy()

                    if tsdf_pose[1, 1] < 0:
                        tsdf_pose[:3, :3] = tsdf_pose[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

                    scaled_pts = pts_list[j] * self.current_metric_scale
                    depth_map = np.nan_to_num(np.linalg.norm(scaled_pts, axis=-1), nan=0.0, posinf=0.0, neginf=0.0)

                    batch_depths.append(depth_map)
                    batch_rgbs.append(window_frames[j])
                    batch_masks.append(window_masks[j])
                    batch_poses.append(tsdf_pose)

                    self.kf_depths.append(depth_map)
                    self.kf_rgbs.append(window_frames[j])
                    self.kf_masks.append(window_masks[j])
                    self.kf_poses.append(tsdf_pose)

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
            t_extract = time.time()
            if self.tsdf_future is not None:
                self.tsdf_future.result()
                self.tsdf_future = None

            current_geometry = self.tsdf.extract_geometry()
            profiler["Geometry_Extraction"] = time.time() - t_extract

            # Coloring runs EVERY submap (separate, heavy pass - profiled on its own).
            t_color = time.time()
            self.last_mesh = self._color_geometry(current_geometry)
            profiler["Coloring"] = time.time() - t_color

            # --- Flush the GPU volume to the CPU accumulator if needed ----------
            # This is what actually bounds nvblox VRAM: once we've extracted and
            # colored the current volume, we can push it to the CPU map and clear
            # the GPU so the block hash never has to grow without limit.
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

            gc.collect()
            torch.cuda.empty_cache()

            pt_alloc, pt_res, sys_used, sys_total = self.vram_tracker.stop()
            nvblox_other = max(sys_used - pt_res, 0.0)
            print("  --- ⏱️ Process Timing (Seconds) ---")
            for k, v in profiler.items():
                print(f"    - {k:<20}: {v:.3f}s")
            print(f"    = Total Submap Time   : {(time.time() - t_win_start):.3f}s")

            print("  --- 💾 GPU Memory (VRAM) ---")
            print(f"    - PyTorch Peak Alloc  : {pt_alloc:.2f} GB")
            print(f"    - PyTorch Reserved    : {pt_res:.2f} GB")
            print(f"    - nvblox / Other      : {nvblox_other:.2f} GB")
            print(f"    - System VRAM Usage   : {sys_used:.2f} GB / {sys_total:.2f} GB")
            print(f"    - Accumulated Verts   : {len(self.accum_mesh.vertices)}")
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