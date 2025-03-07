from mujoco import viewer
import time
import mujoco
import numpy as np

def visualise_traj_generic(x, d: mujoco.MjData, m: mujoco.MjModel, sleep=0.01):

    with viewer.launch_passive(m, d) as v:
        x = np.array(x)
        for i in range(x.shape[0]):
            step_start = time.time()
            qpos = x[i, :m.nq]
            qvel = x[i, m.nq:m.nq + m.nv]
            d.qpos[:] = qpos
            d.qvel[:] = qvel
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(sleep)
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def visualise_pusher(states, d: mujoco.MjData, m: mujoco.MjModel, sleep=0.01):
    """
    Visualizes a gym pusher trajectory.

    Assumes that each state in 'states' is a flat array where the first m.nq entries
    correspond to qpos and the next m.nv entries correspond to qvel.

    Parameters:
      states: list or array of states (each state is a JAX array; converted to NumPy).
      d: current mjx_data instance.
      m: mjx_model.
      sleep: sleep duration between frames.
    """
    # Convert states to NumPy arrays if they aren't already.
    # Here we assume states is a list of 1D arrays.
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


def visualise_finger(states, d: mujoco.MjData, m: mujoco.MjModel, sleep=0.01):
    """
    Visualizes a gym pusher trajectory.

    Assumes that each state in 'states' is a flat array where the first m.nq entries
    correspond to qpos and the next m.nv entries correspond to qvel.

    Parameters:
      states: list or array of states (each state is a JAX array; converted to NumPy).
      d: current mjx_data instance.
      m: mjx_model.
      sleep: sleep duration between frames.
    """
    # Convert states to NumPy arrays if they aren't already.
    # Here we assume states is a list of 1D arrays.
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