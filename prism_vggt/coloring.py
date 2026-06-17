"""Block color cache + best-view colorizer + point-cloud streaming.

This is the self-contained map-appearance layer used by ``StreamingWindowEngine``:

* It colorizes mesh/point-cloud vertices via best-view projection over the current
  window's keyframes, caching one color per coarse "color block" so per-submap cost
  stays bounded to the live region.
* It exposes the map as a block-granular, versioned, hashed point cloud for streaming
  to a remote client (full snapshot + per-version deltas).

It owns no nvblox / Open3D state -- just numpy/torch arrays -- so it can be unit
tested independently of the GPU mapper.
"""
from typing import Dict, List, Optional, Tuple
import hashlib

import numpy as np
import torch


class BlockColorCache:
    """Persistent per-block color store with best-view coloring and delta streaming."""

    # Pack an integer (x, y, z) block index into one int64 for vectorised numpy set
    # ops. Offset keeps negatives non-negative; range is +/- 2^19 blocks (~26 km @ 5cm).
    _KEY_OFFSET: int = 1 << 19
    _KEY_STRIDE: int = 1 << 20

    def __init__(self, voxel_size: float, color_block_mult: float = 2.5,
                 max_depth: float = 4.5, device: str = "cuda",
                 cam_batch: int = 8, point_chunk: int = 500_000) -> None:
        self.voxel_size = voxel_size
        self.color_block_mult = color_block_mult
        self.max_depth = max_depth
        self.device = device
        self.cam_batch = cam_batch
        self.point_chunk = point_chunk
        self.map_version: int = 0
        self.clear()

    # --- cache state ------------------------------------------------------- #
    def clear(self) -> None:
        self._cache_packed = np.empty((0,), dtype=np.int64)    # sorted block keys
        self._cache_color = np.empty((0, 3), dtype=np.float64)  # rgb in [0, 1]
        self._cache_score = np.empty((0,), dtype=np.float64)    # best-view score
        self._cache_version = np.empty((0,), dtype=np.int64)    # version last changed

    def begin_submap(self) -> int:
        """Bump and return the map version; call once per extracted submap before
        coloring so changed blocks are stamped with the new version."""
        self.map_version += 1
        return self.map_version

    def get_map_version(self) -> int:
        return int(self.map_version)

    # --- block key helpers ------------------------------------------------- #
    def _block_size(self) -> float:
        return self.voxel_size * self.color_block_mult

    def _pack_keys(self, keys_xyz: np.ndarray) -> np.ndarray:
        k = keys_xyz.astype(np.int64) + self._KEY_OFFSET
        s = self._KEY_STRIDE
        return (k[:, 0] * s + k[:, 1]) * s + k[:, 2]

    def _unpack_keys(self, packed: np.ndarray) -> np.ndarray:
        s, off = self._KEY_STRIDE, self._KEY_OFFSET
        kz = (packed % s) - off
        rem = packed // s
        ky = (rem % s) - off
        kx = (rem // s) - off
        return np.stack([kx, ky, kz], axis=1)

    def _block_centers(self, packed: np.ndarray) -> np.ndarray:
        if packed.size == 0:
            return np.zeros((0, 3), dtype=np.float64)
        return (self._unpack_keys(packed).astype(np.float64) + 0.5) * self._block_size()

    @staticmethod
    def _block_hashes(packed: np.ndarray, colors: np.ndarray) -> np.ndarray:
        """Per-block content hash (uint64) over the block id + quantized color."""
        if packed.size == 0:
            return np.zeros((0,), dtype=np.uint64)
        col8 = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint64)
        col_packed = (col8[:, 0] << np.uint64(16)) | (col8[:, 1] << np.uint64(8)) | col8[:, 2]
        k = packed.astype(np.uint64)
        return (k * np.uint64(1099511628211)) ^ (col_packed * np.uint64(2654435761))

    # --- cache lookup / update -------------------------------------------- #
    def _cache_lookup(self, query_packed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def _cache_update(self, keys_packed: np.ndarray, colors: np.ndarray, scores: np.ndarray) -> None:
        """Merge improved block colors into the cache (best-view wins). Changed/new
        blocks are stamped with the current map_version for delta streaming."""
        if keys_packed.shape[0] == 0:
            return
        if self._cache_packed.shape[0] > 0:
            pos = np.searchsorted(self._cache_packed, keys_packed)
            pos = np.clip(pos, 0, self._cache_packed.shape[0] - 1)
            exists = self._cache_packed[pos] == keys_packed
            ex_pos, ex_new = pos[exists], np.nonzero(exists)[0]
            better = scores[ex_new] > self._cache_score[ex_pos]
            upd = ex_pos[better]
            self._cache_color[upd] = colors[ex_new][better]
            self._cache_score[upd] = scores[ex_new][better]
            self._cache_version[upd] = self.map_version
            new = ~exists
        else:
            new = np.ones((keys_packed.shape[0],), dtype=bool)
        if new.any():
            n_new = int(new.sum())
            merged_packed = np.concatenate([self._cache_packed, keys_packed[new]])
            merged_color = np.concatenate([self._cache_color, colors[new]])
            merged_score = np.concatenate([self._cache_score, scores[new]])
            merged_version = np.concatenate([
                self._cache_version, np.full((n_new,), self.map_version, dtype=np.int64)])
            order = np.argsort(merged_packed, kind="stable")
            self._cache_packed = merged_packed[order]
            self._cache_color = merged_color[order]
            self._cache_score = merged_score[order]
            self._cache_version = merged_version[order]

    # --- coloring ---------------------------------------------------------- #
    @torch.no_grad()
    def vertex_normals_gpu(self, v_t: torch.Tensor, tri_t: torch.Tensor) -> torch.Tensor:
        """Area-weighted per-vertex normals on the GPU from raw vertex/triangle
        tensors (matches Open3D's convention; avoids building a TriangleMesh)."""
        v = v_t.float()
        tri = tri_t.long()
        v0, v1, v2 = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
        face_n = torch.cross(v1 - v0, v2 - v0, dim=1)
        n = torch.zeros_like(v)
        n.index_add_(0, tri[:, 0], face_n)
        n.index_add_(0, tri[:, 1], face_n)
        n.index_add_(0, tri[:, 2], face_n)
        return torch.nn.functional.normalize(n, dim=1, eps=1e-8)

    @torch.no_grad()
    def _project_blocks(self, pts_np: np.ndarray, nrm_np: np.ndarray,
                        rgbs: List[np.ndarray], depths: List[np.ndarray],
                        masks: List[np.ndarray], poses: List[np.ndarray]
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """Best-view color + score for a bounded set of block reps, projected over the
        current window's frames. Returns (colors [M,3] in [0,1], scores [M,])."""
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

        CAM_BATCH = max(1, int(self.cam_batch))
        CHUNK_SIZE = max(1, int(self.point_chunk))

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
                    torch.tensor(-1.0, device=device))

                batch_max, batch_best = torch.max(score, dim=0)

                update_mask = batch_max > best_score[start_idx:end_idx]
                if update_mask.any():
                    arange = torch.arange(C_size, device=device)
                    chosen_colors = sampled_colors[batch_best, :, arange]
                    final_colors[start_idx:end_idx][update_mask] = chosen_colors[update_mask]
                    best_score[start_idx:end_idx][update_mask] = batch_max[update_mask]

            del imgs_t, depths_t, masks_t, poses_t, R_w_c, t_w_c

        return final_colors.cpu().numpy(), best_score.cpu().numpy()

    @torch.no_grad()
    def color_vertices(self, vertices_np: np.ndarray, normals_np: np.ndarray,
                       rgbs: List[np.ndarray], depths: List[np.ndarray],
                       masks: List[np.ndarray], poses: List[np.ndarray]) -> np.ndarray:
        """Incremental best-view colorization of mesh vertices.

        Quantizes vertices to color blocks (own grid -> exact O(N) propagation, no
        KDTree). Only blocks near the current window's cameras are (re)projected, the
        rest read cached color. Pass empty frame lists to color purely from the cache.
        Returns (N, 3) rgb in [0, 1].
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

        colors, scores, _ = self._cache_lookup(uniq_packed)

        num_cams = len(rgbs)
        if num_cams > 0:
            cam_centers = np.stack([p[:3, 3] for p in poses])
            nearest = np.linalg.norm(rep_xyz[:, None, :] - cam_centers[None, :, :], axis=2).min(axis=1)
            active = nearest <= (self.max_depth + bsize)
            if active.any():
                a = np.nonzero(active)[0]
                new_cols, new_sco = self._project_blocks(rep_xyz[a], rep_nrm[a], rgbs, depths, masks, poses)
                better = new_sco > scores[a]
                ab = a[better]
                colors[ab] = new_cols[better]
                scores[ab] = new_sco[better]
                self._cache_update(uniq_packed[ab], new_cols[better], new_sco[better])

        return colors[inv]

    # --- streaming API ----------------------------------------------------- #
    def _pack_cloud(self, packed: np.ndarray, colors: np.ndarray,
                    since_version: Optional[int] = None) -> Dict:
        pts = self._block_centers(packed).astype(np.float32)
        bh = self._block_hashes(packed, colors)
        map_hash = 0
        if packed.size:
            map_hash = int(hashlib.blake2b(
                np.ascontiguousarray(packed).tobytes() + np.ascontiguousarray(bh).tobytes(),
                digest_size=8).hexdigest(), 16)
        out: Dict = {
            "points": pts,                                            # (K,3) float32 world
            "colors": np.ascontiguousarray(colors, dtype=np.float32),  # (K,3) rgb [0,1]
            "keys": packed.copy(),                                    # (K,) int64 block ids
            "block_hashes": bh,                                       # (K,) uint64
            "version": int(self.map_version),
            "map_hash": map_hash,                                     # whole-cloud hash
        }
        if since_version is not None:
            out["from_version"] = int(since_version)
        return out

    def get_point_cloud_snapshot(self) -> Dict:
        """Full streamable point cloud (one point per color block) + a map_hash for
        the client to detect drift and trigger a full resync."""
        return self._pack_cloud(self._cache_packed, self._cache_color)

    def get_point_cloud_delta(self, since_version: int) -> Dict:
        """Only blocks changed since ``since_version`` (the delta fast-path). On a
        drop/mismatch, re-request specific ``keys`` or fall back to the snapshot.
        Block removal (e.g. TSDF decay) is not tracked yet."""
        if self._cache_packed.size == 0:
            return self._pack_cloud(self._cache_packed, self._cache_color, since_version)
        changed = self._cache_version > int(since_version)
        return self._pack_cloud(self._cache_packed[changed], self._cache_color[changed], since_version)
