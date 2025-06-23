import numpy as np

traj = np.load("sb_trajectories.npy")          # shape (T+1, batch, dim)
x0   = traj[0]
x1   = traj[-1]
v    = traj[1:] - traj[:-1]         # finite-difference velocity
print(np.var(v, axis=(0,1)))        # if ~0 ⇒ deterministic constant
print(np.mean(v, axis=(0,1)))       # should be x1-x0 averaged
