import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d

from nvblox_torch.mapper import Mapper, QueryType
from nvblox_torch.sensor import Sensor
from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams
from nvblox_torch.constants import constants

class NvbloxPanoTSDF:
    def __init__(self, voxel_size_m=0.02, max_depth=4.5, face_size=1024, crop_margin=24, device="cuda"):
        self.device = torch.device(device)
        self.voxel_size_m = voxel_size_m
        self.max_depth = max_depth 
        self.face_size = face_size
        self.crop_margin = crop_margin
        
        print(f"[TSDF] Initializing C++ Nvblox Mapper ({voxel_size_m}m Voxels, {max_depth}m Cap)...")
        
        proj_params = ProjectiveIntegratorParams()
        proj_params.projective_integrator_max_integration_distance_m = self.max_depth
        
        mapper_params = MapperParams()
        mapper_params.set_projective_integrator_params(proj_params)
        
        self.mapper = Mapper(
            voxel_sizes_m=self.voxel_size_m,
            mapper_parameters=mapper_params
        )
        
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

    def num_tsdf_blocks(self):
        """Number of allocated TSDF blocks (a proxy for nvblox GPU memory use)."""
        try:
            return int(self.mapper.tsdf_layer_view(0).num_blocks())
        except Exception:  # pragma: no cover - depends on nvblox build
            return 0

    # Resolved once: the decay method this nvblox build actually exposes.
    _decay_fn = None
    _decay_name = None

    def decay(self):
        """Decay the TSDF so voxels that stop being observed fade and are eventually
        carved (active removal of stale geometry / drift ghosts). Different nvblox_torch
        builds expose this under different names and on either the Python Mapper or the
        underlying C++ object, so we INTROSPECT once: prefer TSDF decay (this is a TSDF
        map), then generic, then occupancy. Raises if none found so the caller's guard
        can disable it cleanly and tell you what to look for. Call only on submaps where
        you ALSO integrated, or a 360° map will slowly erode itself."""
        if self._decay_fn is None:
            mapper = self.mapper
            cands = ["decay_tsdf", "decayTsdf", "decay", "decay_occupancy", "decayOccupancy"]
            objs = [mapper, getattr(mapper, "_c_mapper", None), getattr(mapper, "mapper", None)]
            for obj in objs:
                if obj is None:
                    continue
                for name in cands:
                    fn = getattr(obj, name, None)
                    if callable(fn):
                        self._decay_fn, self._decay_name = fn, name
                        print(f"[TSDF] nvblox decay → {type(obj).__name__}.{name}()")
                        break
                if self._decay_fn is not None:
                    break
            if self._decay_fn is None:
                have = [m for m in dir(self.mapper) if "decay" in m.lower()]
                raise AttributeError(
                    "no nvblox decay method found; methods containing 'decay' on "
                    f"self.tsdf.mapper = {have}. Set TSDF_DECAY=0 or wire the right one.")
        # Most builds take an optional mapper_id (default -1 = all layers); call bare.
        self._decay_fn()

    def update_esdf(self):
        """Recompute the Euclidean Signed Distance Field from the current TSDF.
        Only needed if you query the ESDF (collision distances) for planning."""
        self.mapper.update_esdf()

    def query_esdf(self, points_xyz):
        """Query ESDF distance (meters) at Nx3 world points.

        Mirrors nvblox's own ESDF example, which queries through
        ``query_differentiable_layer(QueryType.ESDF, ...)`` (the plain ``query_layer``
        ESDF path mis-allocates its output and fails with "Inputs do not have the
        required sizes").

        Args:
            points_xyz: (N, 3) CUDA tensor of world coordinates.
        Returns:
            (N,) tensor of signed distances in meters (negative = inside obstacles).
            Unobserved points are returned as NaN (nvblox's 'unknown' sentinel).
        """
        q = points_xyz.reshape(-1, 3).contiguous().float()
        # Run with grad enabled (as nvblox's own example does); the autograd-backed
        # ESDF query can misbehave under no_grad. We detach the result for the caller.
        with torch.enable_grad():
            sdf = self.mapper.query_differentiable_layer(QueryType.ESDF, q).reshape(-1)
        sdf = sdf.detach()
        # Mask nvblox's "unknown" sentinel (unobserved space) as NaN for clean viz.
        unknown = float(constants.esdf_unknown_distance())
        return torch.where(sdf >= unknown - 1e-3, torch.full_like(sdf, float("nan")), sdf)

    def print_nvblox_timing(self):
        """Dump nvblox's internal C++ stage timers (integration, meshing, etc.)."""
        try:
            print(self.mapper.print_timing())
        except Exception as e:  # pragma: no cover - depends on nvblox build
            print(f"[TSDF] nvblox timing unavailable: {e}")