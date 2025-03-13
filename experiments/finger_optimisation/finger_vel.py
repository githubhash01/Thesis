import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--gradient_mode", type=str, default="autodiff", help="solver")
args = parser.parse_args()
os.environ["MJX_SOLVER"] = args.gradient_mode

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")
import mujoco
from mujoco import mjx
import equinox
from viz import visualise_trajectory
from dataclasses import dataclass
import numpy as np


def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, "dtype"):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x


# Load model and create mjx versions of model and data.
path = "/Users/hashim/Desktop/Thesis/experiments/xmls/finger.xml"
model = mujoco.MjModel.from_xml_path(path)
mx = mjx.put_model(model)
dx = mjx.make_data(mx)
mjx_data = jax.tree.map(upscale, dx)


# -------------------------------
# Define a simulation function parameterized by v₀.
# -------------------------------
@equinox.filter_jit
def simulate_trajectory_v0(mx, qpos, v0, running_cost_fn, terminal_cost_fn, U_fixed):
    """
    Simulate a trajectory with initial qpos and v0.

    Args:
        mx: MuJoCo model handle (static).
        qpos: initial positions.
        v0: initial velocity (the parameter we want to optimize).
        running_cost_fn: function(dx) -> cost at each step.
        terminal_cost_fn: function(dx) -> terminal cost.
        U_fixed: fixed control sequence (e.g. zeros) for all time steps.

    Returns:
        states: (Nsteps, nq+nv) array of states,
        total_cost: scalar cost,
        dx_final: final mjx data state.
    """
    # Create the initial state.
    dx = mjx.make_data(mx)
    dx = dx.replace(qpos=qpos, qvel=v0)

    def step_fn(dx, u):
        # Here, you might apply a (fixed) control if needed. For now we just use the dynamics.
        dx = mjx.step(mx, dx)
        c = running_cost_fn(dx)
        state = jnp.concatenate([dx.qpos, dx.qvel])
        return dx, (state, c)

    dx_final, (states, costs) = jax.lax.scan(step_fn, dx, U_fixed)
    total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final)
    return states, total_cost, dx_final


# -------------------------------
# Define loss functions (using running and terminal cost)
# -------------------------------
def running_cost(dx):
    # For example, we can penalize the position of a finger (qpos index 2)
    pos_finger = dx.qpos[2]
    return 0.01 * pos_finger ** 2


def terminal_cost(dx):
    pos_finger = dx.qpos[2]
    return 1.0 * pos_finger ** 2


# In this example, we use a cost that pushes the spinner toward a target configuration.
def spinner_cost(dx):
    # Let's say the goal is to bring the spinner's extra DOFs (qpos[3:]) toward [0.5, 0.5, 0.5].
    goal_pose = jnp.array([0.5, 0.5, 0.5])
    spinner_pose = dx.qpos[3:]
    distance = jnp.linalg.norm(goal_pose - spinner_pose)
    return jnp.square(distance)


# We'll use spinner_cost as both running and terminal cost for this example.
cost_fn = spinner_cost

# -------------------------------
# Set up optimization on the initial velocity v0.
# -------------------------------
# Define your fixed initial position (qpos) and an initial guess for v0.
qpos_init = jnp.array([-1.57079633, -1.57079633, 1.0, 0.0, 0.0, 0.0])
# For instance, if the model's velocity dimension is 5:
v0_init = jnp.array([0.8, 0.0, 0.0, 0.0, 0.0])

# Define a fixed control sequence (e.g., zeros) over Nsteps.
Nsteps = 300
# Here, nu is the control dimension. For example, if nu=2:
nu = 2
U_fixed = jnp.zeros((Nsteps, nu))


# Define the loss function that depends only on v0.
def loss_fn_v0(v0):
    # Run the simulation with the given v0.
    _, total_cost, _ = simulate_trajectory_v0(mx, qpos_init, v0, cost_fn, cost_fn, U_fixed)
    return total_cost


# -------------------------------
# Create an optimizer class for v0.
# -------------------------------
@dataclass
class Optimizer:
    loss: callable
    grad_loss: callable

    def solve(self, x0: jnp.ndarray, learning_rate=1e-2, max_iter=100):
        x = x0
        for i in range(max_iter):
            g = self.grad_loss(x)
            x_new = x - learning_rate * g
            cost = self.loss(x_new)
            print(f"Iteration {i}: cost={cost}")
            x = x_new
        return x


# JIT the loss and compute its gradient with respect to v0.
loss_fn_v0_jit = equinox.filter_jit(loss_fn_v0)
grad_loss_fn_v0 = equinox.filter_jit(jax.jacrev(loss_fn_v0))
optimizer = Optimizer(loss=loss_fn_v0_jit, grad_loss=grad_loss_fn_v0)

# Optimize v0.
optimal_v0 = optimizer.solve(v0_init, learning_rate=0.2, max_iter=10)

# -------------------------------
# Run the final simulation using the optimized v0 and visualize.
# -------------------------------
states, final_cost, dx_final = simulate_trajectory_v0(mx, qpos_init, optimal_v0, cost_fn, terminal_cost, U_fixed)
print("Final cost:", final_cost)

# Create a standard MjData for visualization (to update computed fields, etc.).
d = mujoco.MjData(model)
visualise_trajectory(states, d, model)