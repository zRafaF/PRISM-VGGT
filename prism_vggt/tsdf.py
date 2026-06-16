import math
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d

from nvblox_torch.mapper import Mapper
from nvblox_torch.sensor import Sensor
from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams

class NvbloxPanoTSDF:
    def __init__(self, voxel_size_m=0.02, max_depth=4.5, face_size=1024, crop_margin=24,
                 sensor_mode="cubemap", device="cuda"):
        self.device = torch.device(device)
        self.voxel_size_m = voxel_size_m
        self.max_depth = max_depth
        self.face_size = face_size
        self.crop_margin = crop_margin
        self.sensor_mode = sensor_mode

        print(f"[TSDF] Initializing C++ Nvblox Mapper ({voxel_size_m}m Voxels, {max_depth}m Cap, sensor={sensor_mode})...")

        proj_params = ProjectiveIntegratorParams()
        proj_params.projective_integrator_max_integration_distance_m = self.max_depth

        mapper_params = MapperParams()
        mapper_params.set_projective_integrator_params(proj_params)

        self.mapper = Mapper(
            voxel_sizes_m=self.voxel_size_m,
            mapper_parameters=mapper_params
        )

        if self.sensor_mode == "lidar":
            # Single spherical (lidar) frame per keyframe: the equirect range map is
            # fed directly, so no cubemap / grid_sample. Sensor + lidar->camera
            # rotation are built lazily on the first frame (we need H, W).
            self.lidar = None
            self.T_c_l = None
            print("[TSDF] Sensor mode: LIDAR (1 spherical frame/keyframe, no cubemap).")
        else:
            f = self.face_size / 2.0
            w = self.face_size - (2 * self.crop_margin)
            h = self.face_size - (2 * self.crop_margin)
            c = w / 2.0

            self.camera = Sensor.from_camera(fu=f, fv=f, cu=c, cv=c, width=w, height=h)
            self._precompute_cubemap_grids()

    def _precompute_cubemap_grids(self):
        u = torch.linspace(-1, 1, self.face_size, device=self.device)
        v = torch.linspace(-1, 1, self.face_size, device=self.device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
        base_rays = torch.stack([u_grid, v_grid, torch.ones_like(u_grid)], dim=-1)
        
        self.face_rotations = [
            torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], device=self.device, dtype=torch.float32), 
            torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], device=self.device, dtype=torch.float32),
            torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], device=self.device, dtype=torch.float32),
            torch.tensor([[0, 0, -1], [0, 1, 0], [1, 0, 0]], device=self.device, dtype=torch.float32),
            torch.tensor([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], device=self.device, dtype=torch.float32),
            torch.tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], device=self.device, dtype=torch.float32),
        ]
        
        grids = []
        for R in self.face_rotations:
            ray_global = base_rays @ R.T
            X, Y, Z = ray_global[..., 0], ray_global[..., 1], ray_global[..., 2]
            norm = torch.sqrt(X**2 + Y**2 + Z**2)
            u_norm = torch.atan2(X, Z) / torch.pi
            v_norm = torch.asin(torch.clamp(Y / norm, -1.0, 1.0)) / (torch.pi / 2.0)
            grids.append(torch.stack([u_norm, v_norm], dim=-1))
            
        self.batched_grids = torch.stack(grids, dim=0)
        self.batched_z_mults = (1.0 / torch.sqrt(u_grid**2 + v_grid**2 + 1.0)).unsqueeze(0).expand(6, -1, -1)

    @torch.no_grad()
    def integrate(self, pano_depth_map, pano_rgb, mask, pose):
        if self.sensor_mode == "lidar":
            return self._integrate_lidar(pano_depth_map, mask, pose)
        return self._integrate_cubemap(pano_depth_map, pano_rgb, mask, pose)

    def _ensure_lidar(self, H, W):
        """Lazily build the nvblox Lidar sensor and the fixed lidar->camera rotation.

        Mapping (derived from nvblox/sensors/lidar.h against PanoVGGT's equirect
        convention theta=(u/W-0.5)*2pi, phi=(v/H-0.5)*pi):
          * nvblox ray(u,v) = (cos az cos e, sin az cos e, sin e), az from +X, Z up.
          * equirect ray    = (cos phi sin th, sin phi, cos phi cos th), th from +Z, Y up.
          * With e=phi, az=theta these are a cyclic permutation -> fixed rotation
            R_c_l = [[0,1,0],[0,0,1],[1,0,0]] (lidar axes expressed in camera frame).
          * nvblox rows run elevation +pi/2 (row 0) -> -pi/2; equirect runs -pi/2 ->
            +pi/2, so the depth rows are remapped (exact gather, see row_map below).
        """
        if self.lidar is not None:
            return
        if W % 2 != 0:
            raise ValueError(f"nvblox lidar requires an even azimuth count; got width={W}.")
        self.lidar = Sensor.from_lidar(
            num_azimuth_divisions=W,
            num_elevation_divisions=H,
            vertical_fov_rad=math.pi,        # full sphere (poles are masked upstream)
            min_valid_range_m=0.1,
        )
        R = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        self.T_c_l = torch.from_numpy(T).to(self.device)

        # Exact inverse of nvblox's elevation sampling: lidar row r samples the
        # equirect row whose elevation matches lidar row r. (Verified to 0 px vs a
        # plain vertical flip's ~1.5 px systematic offset.)
        row_map = np.clip(np.round(H * (1.0 - np.arange(H) / (H - 1))).astype(np.int64), 0, H - 1)
        self.lidar_row_map = torch.from_numpy(row_map).to(self.device)
        print(f"[TSDF] Lidar sensor ready ({W}x{H}, 180deg vertical FOV).")

    @torch.no_grad()
    def _integrate_lidar(self, pano_depth_map, mask, pose):
        if isinstance(pano_depth_map, np.ndarray):
            pano_depth_map = torch.from_numpy(pano_depth_map).float().to(self.device)
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose).float().to(self.device)
        if torch.isnan(pose).any() or torch.isinf(pose).any():
            return

        H, W = pano_depth_map.shape
        self._ensure_lidar(H, W)

        if mask is not None:
            mask = torch.from_numpy(mask.copy()).float().to(self.device) if isinstance(mask, np.ndarray) else mask.float()
        else:
            mask = torch.ones_like(pano_depth_map)

        # Anti-erosion edge detection (identical to the cubemap path).
        diff_x = torch.abs(pano_depth_map - torch.roll(pano_depth_map, shifts=-1, dims=1))
        diff_y = torch.abs(pano_depth_map - torch.roll(pano_depth_map, shifts=-1, dims=0))
        mask = mask * ((diff_x < 0.08) & (diff_y < 0.08)).float()
        edges = ((diff_x > 0.15) | (diff_y > 0.15)).float().unsqueeze(0).unsqueeze(0)
        dilated = F.max_pool2d(edges, kernel_size=7, stride=1, padding=3).squeeze()
        mask = mask * (dilated == 0.0).float()

        depth = torch.nan_to_num(pano_depth_map, nan=0.0, posinf=0.0, neginf=0.0).clone()
        invalid = (mask < 0.5) | (depth > self.max_depth) | (depth < 0.1)
        depth[invalid] = 0.0  # 0 -> invalid pixel, ignored by the integrator

        # Reorder rows into nvblox lidar elevation order (exact gather, see
        # _ensure_lidar) and feed the equirect range map as ONE spherical frame.
        lidar_depth = depth[self.lidar_row_map].contiguous()
        lidar_pose = pose @ self.T_c_l   # T_w_l = T_w_c @ T_c_l
        self.mapper.add_depth_frame(lidar_depth, lidar_pose.cpu(), self.lidar)

    @torch.no_grad()
    def _integrate_cubemap(self, pano_depth_map, pano_rgb, mask, pose):
        if isinstance(pano_depth_map, np.ndarray):
            pano_depth_map = torch.from_numpy(pano_depth_map).float().to(self.device)
        if isinstance(pose, np.ndarray):
            pose = torch.from_numpy(pose).float().to(self.device)

        if torch.isnan(pose).any() or torch.isinf(pose).any(): return

        if mask is not None:
            mask = torch.from_numpy(mask.copy()).float().to(self.device) if isinstance(mask, np.ndarray) else mask.float()
        else:
            mask = torch.ones_like(pano_depth_map)
            
        # Anti-erosion edge detection
        diff_x = torch.abs(pano_depth_map - torch.roll(pano_depth_map, shifts=-1, dims=1))
        diff_y = torch.abs(pano_depth_map - torch.roll(pano_depth_map, shifts=-1, dims=0))
        mask = mask * ((diff_x < 0.08) & (diff_y < 0.08)).float()
        
        edges = ((diff_x > 0.15) | (diff_y > 0.15)).float().unsqueeze(0).unsqueeze(0)
        dilated_edges = F.max_pool2d(edges, kernel_size=7, stride=1, padding=3).squeeze()
        mask = mask * (dilated_edges == 0.0).float()

        pano_depth_tensor = pano_depth_map.unsqueeze(0).unsqueeze(0)
        pano_mask_tensor = mask.unsqueeze(0).unsqueeze(0)

        for i in range(6):
            radial_depth = F.grid_sample(pano_depth_tensor, self.batched_grids[i:i+1], mode='nearest', align_corners=False).squeeze()
            radial_depth = torch.nan_to_num(radial_depth, nan=-1.0, posinf=-1.0, neginf=-1.0)
            face_mask = F.grid_sample(pano_mask_tensor, self.batched_grids[i:i+1], mode='nearest', align_corners=False).squeeze()

            optical_depth = radial_depth * self.batched_z_mults[i]
            optical_depth[face_mask < 0.5] = -1.0
            optical_depth[(optical_depth > self.max_depth) | (optical_depth < 0.1)] = -1.0 
            
            if self.crop_margin > 0:
                c = self.crop_margin
                optical_depth = optical_depth[c:-c, c:-c].contiguous()

            face_pose = pose.clone()
            face_pose[:3, :3] = pose[:3, :3] @ self.face_rotations[i]
            self.mapper.add_depth_frame(optical_depth, face_pose.cpu(), self.camera)

    def update_mesh(self):
        """Run (incremental) marching cubes on the TSDF volume. nvblox only
        re-meshes blocks touched since the last call, so this should stay roughly
        constant per submap regardless of total map size."""
        self.mapper.update_color_mesh()

    def get_color_mesh_raw(self):
        """Return the nvblox ColorMesh handle (GPU tensors, zero-copy views)."""
        return self.mapper.get_color_mesh()

    def extract_geometry(self):
        """Extract the reconstructed geometry from the TSDF volume.

        The point cloud we ultimately care about is simply the vertices of this
        structure; the connectivity is only retained so we can derive per-vertex
        normals (used by the colorizer) and optionally export a .glb.
        """
        self.update_mesh()
        return self.get_color_mesh_raw().to_open3d()

    def print_nvblox_timing(self):
        """Dump nvblox's internal C++ stage timers (integration, meshing, etc.)."""
        try:
            print(self.mapper.print_timing())
        except Exception as e:  # pragma: no cover - depends on nvblox build
            print(f"[TSDF] nvblox timing unavailable: {e}")