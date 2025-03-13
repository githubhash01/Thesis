import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx, viewer
import equinox
from pmp import PMP, make_loss, simulate_trajectory
import time
import numpy as np

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


if __name__ == "__main__":
    path = "/Users/hashim/Desktop/Thesis/mjx/diff_sim/xmls/cartpole.xml"
    model = mujoco.MjModel.from_xml_path(path)
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    qpos_init = jnp.array([0.0, 3.14])
    Nsteps, nu = 300, 1
    U0 = jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu))

    def set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    def running_cost(dx):
        u = dx.ctrl
        return 1e-3 * jnp.sum(u ** 2)

    def terminal_cost(dx):
        return 1 * jnp.sum(dx.qpos ** 2)

    loss_fn = make_loss(mx, qpos_init, set_control, running_cost, terminal_cost)
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))

    optimizer = PMP(loss=loss_fn, grad_loss=grad_loss_fn)
    optimal_U = optimizer.solve(U0, learning_rate=0.1, max_iter=500)

    d = mujoco.MjData(model)
    x, cost = simulate_trajectory(mx, qpos_init, set_control, running_cost, terminal_cost, optimal_U)
    visualise_trajectory(x, d, model)
