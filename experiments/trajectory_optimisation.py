from enum import Enum
import argparse
import os
import jax
import jax.numpy as jnp

class GradientMode(str, Enum):
    AUTODIFF = "autodiff"
    FD = "fd"
    IMPLICIT_JAXOPT = "implicit_jaxopt"
    IMPLICIT_LAX = "implicit_lax"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Parse the gradient mode argument.
parser = argparse.ArgumentParser()
parser.add_argument("--gradient_mode", type=str, default="autodiff", help="solver")
args = parser.parse_args()

# Set the MuJoCo solver depending on the gradient mode.
#os.environ["MJX_SOLVER"] = "fd" #hardcodig the solver to be finite difference
os.environ["MJX_SOLVER"] = args.gradient_mode

# Import MuJoCo modules after setting the solver.
import mujoco
from mujoco import mjx
from simulation import (
    make_step_fn,
    make_step_fn_fd,
    simulate,
    visualise_trajectory,

    simulate_,
    upscale,
    build_fd_cache,
    set_control,
    make_step_fn_fd_cache, # default step function using dx instead of state
)

import matplotlib.pyplot as plt

# ---------------------------------------- START OF CODE ---------------------------------------- #

# Load the two_bounce.xml environment.
xml_path = os.path.join(BASE_DIR, "xmls", "two_bounce.xml")
mj_model = mujoco.MjModel.from_xml_path(filename=xml_path)
mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# Choose the appropriate step function.
if args.gradient_mode == GradientMode.FD:
    #dx_template = mjx.make_data(mjx_model)
    #dx_template = jax.tree.map(upscale, dx_template)
    #fd_cache = build_fd_cache(dx_template)
    #step_fn = make_step_fn_fd_cache(mjx_model, set_control, fd_cache)
    #simulate = simulate_
    step_fn = make_step_fn_fd(mjx_model, mjx_data)
    simulate = simulate
else:
    step_fn = make_step_fn(mjx_model, mjx_data)
    simulate = simulate

# Define the target position.
TARGET_POS = jnp.array([0.1571356 , 0.0, 1.15331564, 1, 0.0, 0.0, 0.0])
TARGET_POS_CARTESIAN = jnp.array([TARGET_POS[0], TARGET_POS[2]])
INIT_VEL_STAR = jnp.array([2, 0.0, -0.5, 0.0, 0.0, 0.0])  # ground truth initial velocity


# Helper: Build the full 6D velocity from the free 2D parameters.
def build_full_velocity(free_velocity):
    # free_velocity[0] is v_x and free_velocity[1] is v_z.
    # The other components (v_y and angular velocities) are fixed to 0.
    return jnp.array([free_velocity[0], 0.0, free_velocity[1], 0.0, 0.0, 0.0])

#@jax.jit
def simulate_trajectory(dx, free_velocity):
    full_velocity = build_full_velocity(free_velocity)
    dx = dx.replace(qvel=full_velocity)
    states, _ = simulate(mjx_data=dx, num_steps=300, step_function=step_fn)
    final_pos = states[-1][:7]
    final_pos_cartesian = jnp.array([final_pos[0], final_pos[2]])
    cost = jnp.sum((final_pos_cartesian - TARGET_POS_CARTESIAN) ** 2)
    return states, cost

# Define a loss function that takes the free velocity parameters.
def make_loss(dx):
    def loss(free_velocity):
        _, total_cost = simulate_trajectory(dx, free_velocity)
        return total_cost
    return loss

# Optimization routine: Only update free_velocity (v_x and v_z).
def solve(free_velocity, learning_rate=1e-2, tol=1e-6, max_iter=10):
    loss = make_loss(dx=mjx_data)
    grad_loss = jax.jacrev(loss)
    loss_history = []  # Record the loss for each iteration.

    for i in range(max_iter):
        g = grad_loss(free_velocity)
        free_velocity = free_velocity + learning_rate * g
        f_val = loss(free_velocity)
        loss_history.append(-f_val)
        print(f"Iteration {i}: cost={f_val}, free_velocity={free_velocity}")
        if jnp.linalg.norm(g) < tol or jnp.isnan(g).any():
            return free_velocity

    return free_velocity, loss_history

def plot_loss(loss_history):
    # Plot the loss graph.
    plt.figure()
    plt.plot(loss_history, marker='o')
    plt.title(f"Trajectory Optimisation Loss - Gradient Mode: {args.gradient_mode}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

def main():
    #states, cost = simulate_trajectory(mjx_data, INIT_VEL_STAR)
    #jax.debug.print("Cost: {}", cost)
    #jax.debug.print("Final Position: {}", states[-1][:7])
    #visualise_trajectory(states, mj_data, mj_model)
    free_init = jnp.array([2.5, -0.3])
    optimal_free_velocity, loss_history = solve(free_init)

    # For visualization, convert free parameters to full velocity.
    optimal_velocity = build_full_velocity(optimal_free_velocity)
    states, _ = simulate_trajectory(mjx_data, optimal_free_velocity)
    #visualise_trajectory(states, mj_data, mj_model)
    plot_loss(loss_history)


if __name__ == "__main__":
    main()