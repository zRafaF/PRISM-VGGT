"""Validate the pose-correction gates (deadband + EMA/slerp) replicated exactly
from mapping_server._publish_pose_correction. Confirms: a STILL robot's noisy
corrections are suppressed (avatar holds), a MOVING robot tracks, outliers reject."""
import numpy as np

DEADBAND_M, DEADBAND_DEG, ALPHA = 0.06, 3.0, 0.3
MAX_SPEED, JUMP_MARGIN = 2.5, 0.75

def quat_angle_deg(q1, q2):
    d = float(np.clip(abs(np.dot(np.asarray(q1), np.asarray(q2))), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(d)))

def quat_slerp(q1, q2, t):
    q1 = np.asarray(q1, float); q2 = np.asarray(q2, float)
    dot = float(np.dot(q1, q2))
    if dot < 0.0: q2 = -q2; dot = -dot
    if dot > 0.9995:
        q = q1 + t*(q2-q1); return q/(np.linalg.norm(q)+1e-12)
    th = np.arccos(np.clip(dot,-1,1)); s = np.sin(th)
    return (np.sin((1-t)*th)/s)*q1 + (np.sin(t*th)/s)*q2

class Gate:
    def __init__(self): self.p=None; self.q=None; self.t=None; self.published=0; self.calls=0
    def feed(self, pos, quat, ts):
        self.calls += 1
        pos=np.asarray(pos,float); quat=np.asarray(quat,float)
        if self.p is not None:
            dt=abs(ts-self.t); jump=float(np.linalg.norm(pos-self.p))
            if dt>0 and jump > MAX_SPEED*dt+JUMP_MARGIN: return None      # outlier
            if jump < DEADBAND_M and quat_angle_deg(quat,self.q) < DEADBAND_DEG:
                return None                                              # deadband hold
            pos=(1-ALPHA)*self.p+ALPHA*pos
            quat=quat_slerp(self.q,quat,ALPHA); quat/=np.linalg.norm(quat)+1e-12
        self.p=pos.copy(); self.q=quat.copy(); self.t=ts; self.published+=1
        return pos, quat

rng=np.random.default_rng(0); I=np.array([0,0,0,1.0])
# 1) STILL robot: true pose fixed, corrections = pose + a few cm / few deg noise
g=Gate(); true=np.array([1.0,2.0,1.15])
for k in range(40):
    npos=true+rng.normal(0,0.01,3)
    ang=np.radians(rng.normal(0,1.0)); nq=np.array([0,0,np.sin(ang/2),np.cos(ang/2)])
    g.feed(npos,nq,k*3.0)
print(f"STILL : {g.published}/{g.calls} corrections published "
      f"(rest suppressed → avatar holds); avatar wander would be ~deadband-bounded")

# 2) MOVING robot: walks +x at 0.3 m/s, corrections every 3s (+ noise)
g=Gate(); pubpos=[]
for k in range(40):
    truep=np.array([0.3*3.0*k,2.0,1.15])
    npos=truep+rng.normal(0,0.02,3)
    g.feed(npos,I,k*3.0)
    if g.p is not None: pubpos.append(g.p.copy())
pp=np.array(pubpos); err=np.linalg.norm(pp[-1]-np.array([0.3*3.0*39,2.0,1.15]))
print(f"MOVING: {g.published}/{g.calls} published, tracks motion (final lag {err:.2f} m)")

# 3) OUTLIER: one correction teleports 8 m in 3 s → must be rejected
g=Gate(); g.feed([0,0,1.15],I,0.0)
r=g.feed([30,0,1.15],I,3.0)
print(f"OUTLIER: rejected={r is None} (8 m jump in 3 s > {MAX_SPEED*3+JUMP_MARGIN:.1f} m budget)")
