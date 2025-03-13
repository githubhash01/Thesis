import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--gradient_mode", type=str, default="autodiff", help="solver")
args = parser.parse_args()
# set the MuJoCo solver depending on the gradient mode
os.environ["MJX_SOLVER"] = args.gradient_mode

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import mujoco
from mujoco import mjx
import equinox
from viz import visualise_trajectory
from dataclasses import dataclass
from typing import Callable


def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

@equinox.filter_jit
def simulate_trajectory(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn, U):
    """
    Simulate a trajectory given a control sequence U.

    Args:
        mx: The MuJoCo model handle (static)
        qpos_init: initial positions (array)
        set_control_fn: fn(dx, u) -> dx to apply controls
        running_cost_fn: fn(dx, u) -> cost (float)
        terminal_cost_fn: fn(dx) -> cost (float)
        U: (N, nu) array of controls.

    Returns:
        states: (N, nq+nv) array of states
        total_cost: scalar total cost
    """
    def step_fn(dx, u):
        dx = set_control_fn(dx, u)
        dx = mjx.step(mx, dx)
        c = running_cost_fn(dx)
        state = jnp.concatenate([dx.qpos, dx.qvel])
        return dx, (state, c)

    dx0 = mjx.make_data(mx)
    dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))
    dx_final, (states, costs) = jax.lax.scan(step_fn, dx0, U)
    total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final)
    return states, total_cost


def make_loss(mx, qpos_init, set_control_fn, running_cost_fn, terminal_cost_fn):
    """
    Create a loss function that only takes U as input.
    """
    def loss(U):
        _, total_cost = simulate_trajectory(
            mx, qpos_init,
            set_control_fn, running_cost_fn, terminal_cost_fn,
            U
        )
        return total_cost
    return loss


@dataclass
class PMP:
    loss: Callable[[jnp.ndarray], float]
    grad_loss: Callable[[jnp.ndarray], jnp.ndarray]

    def solve(self, U0: jnp.ndarray, learning_rate=1e-2, tol=1e-6, max_iter=100):
        """
        Gradient descent on the control trajectory.

        U0: initial guess (N, nu)
        Returns: optimized U
        """
        U = U0
        for i in range(max_iter):
            g = self.grad_loss(U)
            U_new = U - learning_rate * g
            f_val = self.loss(U_new)
            print(f"Iteration {i}: cost={f_val}")
            if jnp.linalg.norm(U_new - U) < tol or jnp.isnan(g).any():
                return U_new
            U = U_new
        return U


if __name__ == "__main__":
    path = "/Users/hashim/Desktop/Thesis/experiments/xmls/finger.xml"
    model = mujoco.MjModel.from_xml_path(path)
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    dx = jax.tree.map(upscale, dx)
    qpos_init = jnp.array([.1, 0, -.8, 0.0, 0.0, 0.0])
    Nsteps, nu = 300, 2
    U0 = jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu)) * 2

    def set_control(dx, u):
        # u = jnp.tanh(u) * 0.5
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    def running_cost(dx):
        pos_finger = dx.qpos[2]
        u = dx.ctrl
        return 0.01 * pos_finger ** 2 + 0.01 * jnp.sum(u ** 2)

    def terminal_cost(dx):
        pos_finger = dx.qpos[2]
        return 1 * pos_finger ** 2

    loss_fn = make_loss(mx, qpos_init, set_control, running_cost, terminal_cost)
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))

    optimizer = PMP(loss=loss_fn, grad_loss=grad_loss_fn)
    optimal_U = optimizer.solve(U0, learning_rate=0.2, max_iter=150)
    #optimal_U = U0

    d = mujoco.MjData(model)

    states, _ = simulate_trajectory(mx, qpos_init, set_control, running_cost, terminal_cost, optimal_U)
    visualise_trajectory(states, d, model)
    #print the final cost
    #print(spinner_distance(dx_final))
    #jax.debug.print("Spinner distance", spinner_distance(dx_final))
