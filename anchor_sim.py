"""
Validate the anchor fix (SCALE WARM-UP + robust multi-frame first window).

Mirrors the real logs: VGGT's native metric scale is STABLE (steady-state floor
scale ~0.60), but a window's single-frame floor RANSAC can be an OUTLIER. The log
showed window 0 -> 0.70 (+16%). Locking the whole map to that one estimate puts
the CAMERAS at 0.70 while the per-window GEOMETRY sits at ~0.60 -> the same wall
point reconstructs at different world positions per window = ghosting/drift on the
live viewer. We measure that ghosting (spread of a fixed landmark's reconstruction).
"""
import numpy as np

S_TRUE, H_TRUE = 0.60, 1.15
N, WIN, OVL = 90, 16, 4
STRIDE = WIN - OVL
LANDMARK = np.array([6.0, 1.5, 0.8])
cam = np.stack([np.linspace(0,9,N), 0.2*np.sin(np.linspace(0,4,N)), np.full(N,H_TRUE)],1)
rng = np.random.default_rng(1)

def single_frame_floor(win_idx):
    # one frame's noisy RANSAC estimate; window 0's mid-frame is the known outlier
    return S_TRUE*1.16 if win_idx == 0 else S_TRUE*(1.0+rng.normal(0,0.03))

def robust_first_window():
    # median over several frames of window 0: only the mid frame is bad, the rest
    # are normal -> median is robust (this is the multi-frame first-window anchor)
    frames = [S_TRUE*1.16] + [S_TRUE*(1.0+rng.normal(0,0.03)) for _ in range(4)]
    return float(np.median(frames))

def run(mode):
    starts = list(range(0, N-WIN+1, STRIDE)); K = 3
    samples, cur = [], 1.0; recon = {}
    for wi, st in enumerate(starts):
        if wi == 0:
            s0 = robust_first_window() if mode == 'warmup+robust' else single_frame_floor(0)
            samples.append(s0); s_pose = s0
        else:
            samples.append(single_frame_floor(wi))
            if mode in ('warmup', 'warmup+robust') and len(samples) <= K:
                s_pose = float(np.median(samples[:K]) if len(samples) >= K else np.median(samples))
            else:
                s_pose = cur
        cur = s_pose
        for j in range(WIN):
            i = st+j
            cam_world = cam[i]*(s_pose/S_TRUE)      # pose-chain scale
            recon[(wi,i)] = cam_world + (LANDMARK - cam[i])   # +true-scale local geometry
    return recon

def ghost(recon):
    allp = np.array(list(recon.values()))
    return float(np.linalg.norm(allp-allp.mean(0), axis=1).max()*2)

for mode in ['lock_first', 'warmup', 'warmup+robust']:
    print(f"  {mode:<16} landmark ghost spread = {ghost(run(mode)):.3f} m")
