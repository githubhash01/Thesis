import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import mujoco
from mujoco import mjx, viewer
import equinox
from pmp import PMP, make_loss, simulate_trajectory
import numpy as np
import time

def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x


def visualise_trajectory(states, d: mujoco.MjData, m: mujoco.MjModel, sleep=0.01):
    states_np = [np.array(s) for s in states]
    with viewer.launch_passive(m, d) as v:
        for s in states_np:
            d.qpos[:] = s[:m.nq]
            d.qvel[:] = s[m.nq:m.nq + m.nv]
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(sleep)


if __name__ == "__main__":
    path = "/Users/hashim/Desktop/Thesis/experiments/xmls/finger.xml"
    model = mujoco.MjModel.from_xml_path(path)
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    mjx_data = jax.tree.map(upscale, dx)

    #qpos_init = jnp.array([.1, 0, -.8, 0, 0, 0])
    qpos_init = jnp.array([-1.57, -1.57, 1, 0, 0, 0])
    Nsteps, nu = 300, 2
    U0 = jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu)) * 2
    #U0 = jnp.zeros((Nsteps, nu))

    def set_control(dx, u):
        # u = jnp.tanh(u) * 0.5
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    def finger_cost(dx):
        pos_finger = dx.qpos[2]
        u = dx.ctrl
        return 0.01 * pos_finger ** 2 + 0.01 * jnp.sum(u ** 2)

    def spinner_cost(dx):
        """
        Goal: push the spinner to be vertical flat
        """
        distance = jnp.linalg.norm(dx.qpos[4] - 0.5)
        return jnp.square(distance)

    def running_cost(dx):
        alpha = 0.5
        pos_finger = dx.qpos[2]
        u = dx.ctrl
        return 0.01 * pos_finger ** 2 #+ 0.1 * jnp.sum(u ** 2) #+ alpha * spinner_cost(dx)

    def terminal_cost(dx):
        pos_finger = dx.qpos[2]
        return 1 * pos_finger ** 2

    loss_fn = make_loss(mx, qpos_init, set_control, running_cost, terminal_cost)
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))

    optimizer = PMP(loss=loss_fn, grad_loss=grad_loss_fn)
    optimal_U = optimizer.solve(U0, learning_rate=0.1, max_iter=200)
    #optimal_U = U0

    d = mujoco.MjData(model)
    states, _ = simulate_trajectory(mx, qpos_init, set_control, spinner_cost, terminal_cost, optimal_U)
    visualise_trajectory(states, d, model)