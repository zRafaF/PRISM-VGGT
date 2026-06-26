"""Replicates the exact scale-selection state machine from engine.py to confirm:
 (1) it commits to the robust MEDIAN, not the window-0 outlier;
 (2) once committed it stays locked (no per-window scale drift)."""
import numpy as np

def simulate(floor_scales, K=3, robust_first=None):
    samples, committed, cur = [], False, 1.0
    log = []
    for wi, fs in enumerate(floor_scales):
        if wi == 0:                                   # is_first_window branch
            s_anchor = robust_first if robust_first is not None else fs
            samples.append(s_anchor)
        else:
            if not committed:                         # collect until committed
                samples.append(fs)
            if not committed and len(samples) > 0:
                s_anchor = float(np.median(samples))
                if len(samples) >= K:
                    committed = True
            else:                                     # LOCK_SCALE_AFTER_FIRST
                s_anchor = cur
        cur = s_anchor
        log.append((wi, round(s_anchor, 4), committed))
    return log

# Window-0 mid-frame outlier 0.70 (from the real log), steady-state ~0.60 after.
floor = [0.7003, 0.602, 0.591, 0.604, 0.581, 0.608, 0.621, 0.605]

print("OLD behaviour (lock to window-0 single frame):")
print("   locked scale = 0.7003 forever  (=> ~16% map inflation, camera/geom mismatch)\n")

print("NEW: warm-up median, window-0 = single (outlier) frame:")
for wi, s, c in simulate(floor):
    print(f"   submap {wi}: scale={s}  {'LOCKED' if c else 'warming'}")

print("\nNEW: warm-up median + robust multi-frame window-0 (=0.60 median of its frames):")
for wi, s, c in simulate(floor, robust_first=0.600):
    print(f"   submap {wi}: scale={s}  {'LOCKED' if c else 'warming'}")
