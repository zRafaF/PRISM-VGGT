import os
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d

from nvblox_torch.mapper import Mapper, QueryType
from nvblox_torch.sensor import Sensor
from nvblox_torch.mapper_params import MapperParams, ProjectiveIntegratorParams
from nvblox_torch.constants import constants

# Anti-erosion depth-edge mask (env-tunable). The integrator drops pixels at depth
# discontinuities to avoid smeared "erosion" surfaces. Too aggressive a dilation also
# erases small/thin objects (a backpack on a table). Lower EDGE_DILATE_PX (e.g. 3) and/or
# raise EDGE_REJECT_THRESH to keep more of small DYNAMIC objects.
_EDGE_SMOOTH = float(os.environ.get("EDGE_SMOOTH_THRESH", "0.08"))   # keep pixels smoother than this (m)
_EDGE_REJECT = float(os.environ.get("EDGE_REJECT_THRESH", "0.15"))   # treat as a depth edge above this (m)
_EDGE_DILATE = int(os.environ.get("EDGE_DILATE_PX", "7")) | 1        # dilation kernel (forced odd)

# ── TSDF decay tuning (time/batch sliding window for nav) ────────────────────
# nvblox multiplies every voxel's weight by TSDF_DECAY_FACTOR on each decay() call
# and deallocates voxels whose weight falls below the threshold. Re-observed voxels
# recover their weight via integration, so calling decay() once per submap keeps what
# the robot currently sees and fades what it has left — a per-submap sliding window of
# length K ≈ ln(threshold/W)/ln(factor) submaps. nvblox's DEFAULT 0.95 fades over ~180
# submaps (≈ never) — the reason carving looked broken. 0.8 ≈ ~30 submaps; lower =
# shorter memory / more aggressive carving. TSDF_DECAY_SET_FREE makes a fully-decayed
# voxel read as FREE distance so the ESDF treats reclaimed space as traversable (nav).
_DECAY_FACTOR = float(os.environ.get("TSDF_DECAY_FACTOR", "0.8"))
_DECAY_SET_FREE = os.environ.get("TSDF_DECAY_SET_FREE", "1") == "1"


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

        # Decay tuning (see _DECAY_FACTOR notes above). nvblox's default factor (0.95) is
        # far too gentle to carve within a session; set a usable factor + deallocate +
        # free-distance-on-decayed so decay() actually reclaims space in the ESDF.
        try:
            _d = mapper_params.get_tsdf_decay_integrator_params()
            _d.tsdf_decay_factor = _DECAY_FACTOR
            _d.tsdf_set_free_distance_on_decayed = _DECAY_SET_FREE
            mapper_params.set_tsdf_decay_integrator_params(_d)
            _b = mapper_params.get_decay_integrator_base_params()
            _b.decay_integrator_deallocate_decayed_blocks = True
            mapper_params.set_decay_integrator_base_params(_b)
            import math
            _k = int(math.log(1e-3 / 10.0) / math.log(max(min(_DECAY_FACTOR, 0.999), 1e-3)))
            print(f"[TSDF] decay tuned: factor={_DECAY_FACTOR} (~{_k} submaps to carve), "
                  f"set_free_on_decayed={_DECAY_SET_FREE}, deallocate=True")
        except Exception as e:  # pragma: no cover - depends on nvblox build
            print(f"[TSDF] could not set decay params ({e}); using nvblox defaults "
                  f"(decay will be too gentle to carve — check the nvblox version)")

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
        mask = mask * ((diff_x < _EDGE_SMOOTH) & (diff_y < _EDGE_SMOOTH)).float()
        
        edges = ((diff_x > _EDGE_REJECT) | (diff_y > _EDGE_REJECT)).float().unsqueeze(0).unsqueeze(0)
        dilated_edges = F.max_pool2d(edges, kernel_size=_EDGE_DILATE, stride=1,
                                     padding=_EDGE_DILATE // 2).squeeze()
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

    # Resolved once (introspected): the decay + deallocate methods this nvblox build
    # actually exposes. TSDF maps MUST decay the TSDF layer — decaying only the
    # occupancy layer (unused here) does nothing visible, which is the usual reason
    # "old points never disappear".
    _decay_fn = None
    _decay_name = None
    _dealloc_fn = None
    # Radius-clear (egocentric local-map prune of the TSDF, so the ESDF used for
    # navigation only ever reflects a bounded, recent volume). Resolved once.
    _prune_fn = None
    _prune_name = None
    _prune_resolved = False
    _prune_warned = False

    @staticmethod
    def _call_any(fn):
        """Call an nvblox method that may take no args or an optional mapper_id."""
        for args in ((), (-1,), (0,)):
            try:
                fn(*args)
                return True
            except TypeError:
                continue
        fn()  # last resort: surface the real error
        return True

    def _resolve_decay(self):
        mapper = self.mapper
        objs = [("Mapper", mapper),
                ("c_mapper", getattr(mapper, "_c_mapper", None)),
                ("mapper", getattr(mapper, "mapper", None))]
        # one-time report of what this build exposes (so a wrong/missing API is obvious)
        found = {}
        for label, obj in objs:
            if obj is None:
                continue
            for m in dir(obj):
                ml = m.lower()
                if any(k in ml for k in ("decay", "deallocate")) and not m.startswith("__"):
                    found.setdefault(label, []).append(m)
        print(f"[TSDF] nvblox decay/deallocate methods available: {found or 'NONE'}")
        # prefer TSDF decay; generic decay; occupancy LAST (won't carve a TSDF mesh)
        for name in ("decay_tsdf", "decayTsdf", "decay", "decay_occupancy", "decayOccupancy"):
            for _label, obj in objs:
                fn = getattr(obj, name, None) if obj is not None else None
                if callable(fn):
                    self._decay_fn, self._decay_name = fn, name
                    break
            if self._decay_fn is not None:
                break
        if self._decay_name in ("decay_occupancy", "decayOccupancy"):
            print("[TSDF] WARNING: only OCCUPANCY decay found — this is a TSDF map, so it "
                  "will NOT carve the mesh. Old geometry won't disappear via decay; use the "
                  "sliding-window mode (PRISM_RESET_EACH_BATCH=1) for dynamic scenes.")
        elif self._decay_name:
            print(f"[TSDF] nvblox decay → {self._decay_name}() (TSDF carving active)")
        # optional: free fully-decayed blocks so they leave the mesh + the manifest
        for name in ("deallocate_fully_decayed_blocks", "deallocateFullyDecayedBlocks",
                     "update_hashmaps"):
            for _label, obj in objs:
                fn = getattr(obj, name, None) if obj is not None else None
                if callable(fn):
                    self._dealloc_fn = fn
                    print(f"[TSDF] nvblox deallocate → {name}()")
                    break
            if self._dealloc_fn is not None:
                break
        if self._decay_fn is None:
            raise AttributeError(
                "no nvblox decay method found. Methods seen: "
                f"{found}. Set TSDF_DECAY=0 (and use PRISM_RESET_EACH_BATCH=1 for dynamics).")

    def decay(self):
        """Decay the TSDF so voxels that stop being observed (or are contradicted by a
        new observation — e.g. a moved object) fade and are eventually carved. Resolved
        once via introspection (see _resolve_decay), then called every integrated submap.
        A 360° camera re-observes everything in range, so observed surfaces are refreshed
        and only stale geometry decays away."""
        if self._decay_fn is None:
            self._resolve_decay()
        self._call_any(self._decay_fn)
        if self._dealloc_fn is not None:
            try:
                self._call_any(self._dealloc_fn)
            except Exception:
                pass

    def _resolve_prune(self):
        """Find this nvblox build's radius-clear method (clears blocks outside a sphere
        — nvblox's egocentric/local-map primitive, e.g. isaac_ros_nvblox's
        `map_clearing_radius_m`). Falls back to decay+deallocate if the binding doesn't
        expose one. Reports what it found, like _resolve_decay."""
        self._prune_resolved = True
        mapper = self.mapper
        objs = [("Mapper", mapper),
                ("c_mapper", getattr(mapper, "_c_mapper", None)),
                ("mapper", getattr(mapper, "mapper", None))]
        found = {}
        for label, obj in objs:
            if obj is None:
                continue
            for m in dir(obj):
                ml = m.lower().replace("_", "")
                if "outsideradius" in ml or ("clear" in ml and "radius" in ml):
                    found.setdefault(label, []).append(m)
        print(f"[TSDF] nvblox radius-clear methods available: {found or 'NONE'}")
        for name in ("clear_outside_radius", "clearOutsideRadius",
                     "clear_blocks_outside_radius", "clearBlocksOutsideRadius"):
            for _label, obj in objs:
                fn = getattr(obj, name, None) if obj is not None else None
                if callable(fn):
                    self._prune_fn, self._prune_name = fn, name
                    print(f"[TSDF] nvblox TSDF prune → {name}() (ESDF will be bounded to the local sphere)")
                    return
        print("[TSDF] WARNING: no nvblox radius-clear in this build — TSDF prune falls "
              "back to decay+deallocate (weight-based). Build nvblox with clearOutsideRadius "
              "for a hard local-map bound, or set TSDF_PRUNE_RADIUS_M=0 for benchmarks.")

    def prune_outside_radius(self, center, radius):
        """Prune the nvblox TSDF to a sphere of ``radius`` m around ``center`` (the robot),
        so the ESDF used for navigation only sees a bounded, recent local volume — and the
        mesh-extraction cost stays ~constant as the robot travels (bounds the live latency).

        Uses nvblox's native radius-clear when the build exposes it (a hard spatial bound);
        otherwise falls back to decay+deallocate. The caller disables this entirely when
        radius<=0 (full accumulation — for SoTA benchmarks)."""
        if radius is None or radius <= 0:
            return
        if not self._prune_resolved:
            self._resolve_prune()
        c = (center.detach().cpu().numpy() if isinstance(center, torch.Tensor)
             else np.asarray(center))
        c = np.asarray(c, dtype=np.float32).reshape(3)
        if self._prune_fn is not None:
            # Tolerate the binding's exact signature: (array, r) / (tensor, r) / (x,y,z,r).
            for args in ((c, float(radius)),
                         (torch.from_numpy(c), float(radius)),
                         (float(c[0]), float(c[1]), float(c[2]), float(radius))):
                try:
                    self._prune_fn(*args)
                    return
                except TypeError:
                    continue
                except Exception as e:
                    if not self._prune_warned:
                        print(f"[TSDF] radius prune call failed ({e}); falling back to decay.")
                        self._prune_warned = True
                    break
        # Fallback: weight-based shrink (decay then free fully-decayed blocks → leaves ESDF).
        self.decay()

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