import mujoco
from mujoco import viewer
import time
import numpy as np

def visualise_traj_generic(
        x, d: mujoco.MjData, m: mujoco.MjModel, sleep=0.01
):

    with viewer.launch_passive(m, d) as v:
        x = np.array(x)
        for b in range(x.shape[0]):
            for i in range(x.shape[1]):
                step_start = time.time()
                qpos = x[b, i, :m.nq]
                qvel = x[b, i, m.nq:m.nq + m.nv]
                d.qpos[:] = qpos
                d.qvel[:] = qvel
                mujoco.mj_forward(m, d)
                v.sync()
                time.sleep(sleep)
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

def visualise_trajectory(states, d: mujoco.MjData, m: mujoco.MjModel, sleep=0.01):

    states_np = [np.array(s) for s in states]

    with viewer.launch_passive(m, d) as v:
        for s in states_np:
            step_start = time.time()
            # Extract qpos and qvel from the state.
            nq = m.nq
            # We assume the state was produced as jnp.concatenate([qpos, qvel])
            qpos = s[:nq]
            qvel = s[nq:nq + m.nv]
            d.qpos[:] = qpos
            d.qvel[:] = qvel
            mujoco.mj_forward(m, d)
            v.sync()
            # Optionally sleep to mimic real time.
            time.sleep(sleep)
            # Optionally, adjust to match the simulation timestep:
            time_until_next = m.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)
