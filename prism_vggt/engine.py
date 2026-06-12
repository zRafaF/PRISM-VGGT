import torch
import numpy as np
import open3d as o3d
import time
from concurrent.futures import ThreadPoolExecutor

from .tsdf import NvbloxPanoTSDF
from .perception_base import BasePerceptionExtractor

from .utils.alignment import align_cam_pts_irls 
from .utils.geometry import register_camera_poses_kabsch
from .utils.metricfication import estimate_metric_scale_from_floor
from .utils.profiler import VRAMProfiler

class StreamingWindowEngine:
    def __init__(self, perception: BasePerceptionExtractor, voxel_size=0.02, max_depth=4.5, target_camera_height=1.5, device="cuda"):
        self.perception = perception
        self.device = device
        
        self.max_depth = max_depth
        self.voxel_size = voxel_size
        self.target_camera_height = target_camera_height 
        self.min_translation_m = 0.10  # --- FIX 1: Restored Keyframe Filter Limit ---
        
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
        
        self.last_integrated_position = None

    def _async_tsdf_task(self, depth_maps, rgb_frames, masks, poses):
        for j in range(len(poses)):
            self.tsdf.integrate(depth_maps[j], rgb_frames[j], masks[j], poses[j])
        torch.cuda.synchronize()

    @torch.no_grad()
    def _apply_pytorch_colors(self, vertices_np, normals_np, batch_rgbs, batch_depths, batch_masks, batch_poses):
        """Projects vertices into all keyframes simultaneously."""
        num_pts = vertices_np.shape[0]
        if num_pts == 0 or len(batch_rgbs) == 0:
            return np.zeros((num_pts, 3))

        B = len(batch_rgbs)
        device = self.device

        imgs_t = torch.from_numpy(np.stack(batch_rgbs)).float().to(device) / 255.0
        imgs_t = imgs_t.permute(0, 3, 1, 2)
        depths_t = torch.from_numpy(np.stack(batch_depths)).float().to(device).unsqueeze(1)
        masks_t = torch.from_numpy(np.stack(batch_masks)).float().to(device).unsqueeze(1)
        poses_t = torch.from_numpy(np.stack(batch_poses)).float().to(device)
        
        R_w_c = poses_t[:, :3, :3]
        t_w_c = poses_t[:, :3, 3]

        vertices = torch.from_numpy(vertices_np).float().to(device)
        normals = torch.from_numpy(normals_np).float().to(device)
        final_colors = torch.zeros((num_pts, 3), device=device)
        
        CHUNK_SIZE = 500_000 
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

            max_scores, best_cam_idx = torch.max(score, dim=0)
            update_mask = max_scores > -1.0
            
            if update_mask.any():
                final_colors[start_idx:end_idx][update_mask] = sampled_colors[best_cam_idx[update_mask], :, torch.arange(C_size, device=device)[update_mask]]

        return final_colors.cpu().numpy()

    @torch.no_grad()
    def get_full_map(self):
        """Extract the current fused TSDF surface as a colored mesh + point cloud.

        This is the "fetch the whole map" half of the streaming contract: it is
        independent of the per-submap point stream and can be called on demand
        (e.g. by the parent project) at whatever cadence it needs, since it pays
        the cost of GPU mesh extraction (marching cubes) + multi-keyframe vertex
        coloring.

        Returns:
            (mesh, pcd): an open3d.geometry.TriangleMesh and a colored
            open3d.geometry.PointCloud built from its vertices. Both are empty
            (but valid) geometries if the map has no integrated data yet.
        """
        mesh = self.tsdf.extract_mesh()
        pcd = o3d.geometry.PointCloud()

        if mesh is not None and len(mesh.vertices) > 0:
            mesh.compute_vertex_normals()
            colors = self._apply_pytorch_colors(
                np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals),
                self.kf_rgbs, self.kf_depths, self.kf_masks, self.kf_poses
            )
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

            valid_color_mask = colors.sum(axis=1) > 0.0
            pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[valid_color_mask])
            pcd.colors = o3d.utility.Vector3dVector(colors[valid_color_mask])
        else:
            mesh = o3d.geometry.TriangleMesh()

        return mesh, pcd

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
            profiler["Perception_Inference"] = time.time() - t1

            pts_list, poses = preds["points"], preds["poses"]
            
            t2 = time.time()
            mid_idx = self.window_size // 2

            # Estimate absolute scale (raw VGGT units -> meters) from the floor plane
            # in this window's RAW points. Must stay in raw units: the RANSAC
            # distance_threshold inside this function is a fixed value, so feeding it
            # already-rescaled points would make its effective real-world tolerance
            # drift as current_metric_scale changes across submaps.
            floor_scale, floor_conf = estimate_metric_scale_from_floor(
                pts_list[mid_idx],
                target_camera_height=self.target_camera_height
            )

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
                    # 90% running scale / 10% absolute floor-based scale
                    self.current_metric_scale = 0.9 * relative_scale + 0.1 * floor_scale
                else:
                    self.current_metric_scale = relative_scale

            self.current_metric_scale = np.clip(self.current_metric_scale, 0.1, 5.0)

            print(f"  [Scale] floor_scale={floor_scale}, floor_conf={floor_conf:.3f} "
                  f"-> current_metric_scale={self.current_metric_scale:.4f}")

            metric_local_poses = []
            for p in poses:
                mp = p.copy()
                mp[:3, 3] *= self.current_metric_scale 
                metric_local_poses.append(mp)
                
            submap_origin_inv = np.linalg.inv(metric_local_poses[0])
            canonical_poses = [submap_origin_inv @ mp for mp in metric_local_poses]

            anchor_pose = np.eye(4)
            if not self.is_first_window:
                src_cam_np = np.stack(canonical_poses[:self.overlap])
                tgt_cam_np = np.stack(self.prev_overlap_global_poses)
                R_align, t_align = register_camera_poses_kabsch(src_cam_np, tgt_cam_np, scale=1.0)
                anchor_pose[:3, :3] = R_align
                anchor_pose[:3, 3] = t_align

                aligned_pos = np.array([(anchor_pose @ canonical_poses[k])[:3, 3] for k in range(self.overlap)])
                tgt_pos = tgt_cam_np[:, :3, 3]
                residual = np.linalg.norm(aligned_pos - tgt_pos, axis=-1)
                print(f"  [Align] Kabsch overlap residual (m): mean={residual.mean():.4f}, max={residual.max():.4f}")

            batch_depths, batch_rgbs, batch_masks, batch_poses = [], [], [], []
            start_idx = 0 if self.is_first_window else self.overlap
            
            for j in range(self.window_size):
                global_pose = anchor_pose @ canonical_poses[j]
                
                if j >= start_idx:
                    current_pos = global_pose[:3, 3]

                    tsdf_pose = global_pose.copy()
                    if tsdf_pose[1, 1] < 0:
                        print(f"  [FlipCheck] frame {i+j}: R[1,1]={global_pose[1,1]:.3f} -> "
                              f"applying 180deg flip to tsdf_pose only (trajectory pose left untouched)")
                        tsdf_pose[:3, :3] = tsdf_pose[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

                    if i == 0 and start_idx <= j < start_idx + 3:
                        height_above_floor = current_pos[1]  # Y-down: floor is at +height
                        print(f"  [Height] frame {i+j}: world_pos=({current_pos[0]:.3f}, "
                              f"{current_pos[1]:.3f}, {current_pos[2]:.3f}) "
                              f"-> trajectory 'height above floor' (z=-Y) = {-height_above_floor:.3f} m")

                    scaled_pts = pts_list[j] * self.current_metric_scale
                    depth_map = np.nan_to_num(np.linalg.norm(scaled_pts, axis=-1), nan=0.0, posinf=0.0, neginf=0.0)

                    # Always integrate every processed frame into the TSDF, even if the
                    # camera hasn't moved. A stationary camera can still see moving
                    # objects (e.g. someone walking through the scene), and nvblox
                    # needs continuous integration to update/clear those voxels.
                    batch_depths.append(depth_map)
                    batch_rgbs.append(window_frames[j])
                    batch_masks.append(window_masks[j])
                    batch_poses.append(tsdf_pose)

                    # Trajectory dots and coloring keyframes are spatially decimated
                    # so they don't pile up near-duplicate entries while the camera
                    # is stationary.
                    should_keyframe = False
                    if self.last_integrated_position is None:
                        should_keyframe = True
                    else:
                        dist = np.linalg.norm(current_pos - self.last_integrated_position)
                        if dist >= self.min_translation_m:
                            should_keyframe = True

                    if should_keyframe:
                        self.last_integrated_position = current_pos.copy()

                        self.trajectory.append(current_pos)
                        self.processed_indices.append(i + j)
                        self.full_poses.append(global_pose)

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
            
            t4 = time.time()

            if self.tsdf_future is not None:
                self.tsdf_future.result()
                self.tsdf_future = None

            # Always extract + color the fused map for this submap's stream output.
            # (Previously throttled to every 3rd submap, which left stale geometry
            # in the stream for the skipped submaps.)
            self.last_mesh, self.last_pcd = self.get_full_map()

            profiler["Meshing_&_Coloring"] = time.time() - t4
            
            torch.cuda.empty_cache()

            pt_alloc, pt_res, sys_used, sys_total = self.vram_tracker.stop()
            print("  --- ⏱️ Process Timing (Seconds) ---")
            for k, v in profiler.items():
                print(f"    - {k:<20}: {v:.3f}s")
            print(f"    = Total Submap Time   : {(time.time() - t_win_start):.3f}s")
            
            print("  --- 💾 GPU Memory (VRAM) ---")
            print(f"    - PyTorch Peak Alloc  : {pt_alloc:.2f} GB")
            print(f"    - System VRAM Usage   : {sys_used:.2f} GB / {sys_total:.2f} GB")
            self.submap_count += 1

            yield_mesh = self.last_mesh if self.last_mesh else o3d.geometry.TriangleMesh()
            yield_pcd = self.last_pcd if self.last_pcd else o3d.geometry.PointCloud()
            yield yield_mesh, yield_pcd, np.array(self.trajectory), []

        if self.tsdf_future is not None:
            self.tsdf_future.result()

        print("\n==========================================")
        print(f"[Engine] Sequence Complete ({(time.time() - t_seq_start):.2f}s). Extracting Unified Global Mesh...")
        final_mesh, final_pcd = self.get_full_map()

        yield final_mesh, final_pcd, np.array(self.trajectory), []