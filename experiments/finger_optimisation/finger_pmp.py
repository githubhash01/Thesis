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
import numpy as np

def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

path = "/Users/hashim/Desktop/Thesis/experiments/xmls/finger.xml"
model = mujoco.MjModel.from_xml_path(path)
mx = mjx.put_model(model)
dx = mjx.make_data(mx)
mjx_data = jax.tree.map(upscale, dx)


@equinox.filter_jit
def simulate_trajectory(mx, dx0, set_control_fn,  running_cost_fn, terminal_cost_fn, U):
    """
    Simulate a trajectory given a control sequence U, starting from an initial data object dx.
    This function returns the states along the trajectory, the total cost, and the final dx.
    """
    def step_fn(dx, u):
        # Set control input and perform a simulation step.
        dx = set_control_fn(dx, u)
        dx = mjx.step(mx, dx)
        c = running_cost_fn(dx)
        state = jnp.concatenate([dx.qpos, dx.qvel])
        return dx, (state, c)

    # Optionally update dx with any initial conditions (for example, setting velocity)
    dx_final, (states, costs) = jax.lax.scan(step_fn, dx0, U)
    total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final)
    return states, total_cost, dx_final


def make_loss(mx, dx, set_control, running_cost_fn, terminal_cost_fn):
    """
    Create a loss function that only takes U as input.
    """
    def loss(U):
        _, total_cost, _ = simulate_trajectory(mx, dx, set_control, running_cost_fn, terminal_cost_fn, U)
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
            if jnp.isnan(g).any():
                return U_new
            U = U_new
        return U


if __name__ == "__main__":

    qpos_init = jnp.array([-1.57079633, -1.57079633, 1.0, 0.0, 0.0, 0.0])
    qvel_init = jnp.array([0.8, 0.0, 0.0, 0.0, 0.0])
    mjx_data = mjx_data.replace(qpos=qpos_init, qvel=qvel_init)
    Nsteps, nu = 300, 2
    U0 = jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu)) * 2
    #U0 = jnp.zeros((Nsteps, nu)) # set U0 to all zeros

    def set_control(dx, u):
        dx = dx.replace(ctrl=dx.ctrl.at[:].set(u))
        return dx

    def running_cost(dx):
        pos_finger = dx.qpos[2]
        u = dx.ctrl
        return 0.01 * pos_finger ** 2 + 0.01 * jnp.sum(u ** 2)

    def spinner_cost(dx):
        """
        Goal: push the spinner to be vertical flat
        """
        goal_pose = jnp.array([.5, .5, .5])
        spinner_pose = dx.qpos[3:]
        distance = jnp.linalg.norm(goal_pose - spinner_pose)
        return jnp.square(distance)

    def terminal_cost(dx):
        pos_finger = dx.qpos[2]
        return 1 * pos_finger ** 2

    loss_fn = make_loss(mx, mjx_data, set_control, spinner_cost, spinner_cost)
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))

    optimizer = PMP(loss=loss_fn, grad_loss=grad_loss_fn)
    optimal_U = optimizer.solve(U0, learning_rate=0.2, max_iter=10)

    d = mujoco.MjData(model)
    states, _, dx_final = simulate_trajectory(mx, mjx_data, set_control, spinner_cost, terminal_cost, optimal_U)
    print(spinner_cost(dx_final))
    visualise_trajectory(states, d, model)