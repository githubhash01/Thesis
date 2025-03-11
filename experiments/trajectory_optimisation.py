from enum import Enum
import argparse
import os
import jax
import jax.numpy as jnp
import optax # for optimisation

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Parse the gradient mode argument.
parser = argparse.ArgumentParser()
parser.add_argument("--gradient_mode", type=str, default="autodiff", help="solver")
args = parser.parse_args()

# Set the MuJoCo solver depending on the gradient mode.
os.environ["MJX_SOLVER"] = args.gradient_mode

# Import MuJoCo modules after setting the solver.
import mujoco
from mujoco import mjx
from simulation import (
    make_step_fn,
    make_step_fn_default,
    simulate,
    simulate_data,
simulate_data_lax,
    upscale,
    visualise_trajectory
)

import matplotlib.pyplot as plt

# ---------------------------------------- START OF CODE ---------------------------------------- #
xml_path = os.path.join(BASE_DIR, "xmls", "finger.xml")
mj_model = mujoco.MjModel.from_xml_path(filename=xml_path)
mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)
dx_template = mjx.make_data(mjx_model)
mjx_data = jax.tree.map(upscale, dx_template)

mjx_data = mjx_data.replace(qpos=jnp.array([-1.57079633, -1.57079633, 1, 0.0, 0.0, 0.0]))

spinner_tip_site = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "target_site")

TARGET_POS = jnp.array([-0.08, 0.0, -0.4])  # target final position of the spinner (x, y, z)
INIT_VEL = jnp.array([0.95, -0.2])  # there are many possible solutions

#step_fn = make_step_fn(mjx_model, mjx_data)
step_function = make_step_fn_default(mjx_model)

# Helper: Build the full 5D velocity from the free 2D parameters.
def build_full_velocity(free_velocity):
    return jnp.array([free_velocity[0], free_velocity[1], 0.0, 0.0, 0.0])

# get the spinner tip position from the data
def get_spinner_tip_position(dx):
    return dx.site_xpos[spinner_tip_site]

@jax.jit
def simulate_trajectory(dx, free_velocity):
    full_velocity = build_full_velocity(free_velocity)
    dx = dx.replace(qvel=full_velocity)
    states, dx = simulate_data_lax(mjx_data=dx, num_steps=300, step_fn=step_function)
    final_position = get_spinner_tip_position(dx)
    distance = jnp.linalg.norm(final_position - TARGET_POS)
    return states, distance

# Define a loss function that takes the free velocity parameters.
def make_loss(dx):
    def loss(free_velocity):
        _, total_cost = simulate_trajectory(dx, free_velocity)
        return total_cost
    return loss

def solve(free_velocity, learning_rate=10, tol=1e-6, max_iter=20):
    loss = make_loss(mjx_data)
    grad_loss = jax.jit(jax.jacrev(loss))
    loss_history = []  # Record the loss for each iteration.

    for i in range(max_iter):
        g = grad_loss(free_velocity)
        free_velocity = free_velocity - learning_rate * g
        learning_rate *= 0.8  # Decay the learning rate.
        f_val = loss(free_velocity)
        loss_history.append(f_val)
        jax.debug.print("Iteration {}: cost={}, free_velocity={}", i, f_val, free_velocity)
        if jnp.linalg.norm(g) < tol or jnp.isnan(g).any():
            return free_velocity

    return free_velocity, loss_history

def solve_with_optax(init_velocity, max_steps=20):
    # 1) Learning-rate schedule:
    # Starts at 10, decays by 0.9 every 5 steps (tweak as needed).
    lr_schedule = optax.exponential_decay(
        init_value=10.0,
        transition_steps=5,
        decay_rate=0.9,
        end_value=1e-3  # won't decay below this
    )
    # 2) Define an optimizer
    optimizer = optax.sgd(learning_rate=lr_schedule)
    # 3) Initialize optimizer state
    opt_state = optimizer.init(init_velocity)

    # We'll store the cost each iteration
    loss_history = []

    # 4) Per-iteration update function (JIT-ed)
    @jax.jit
    def train_step(params, opt_state):
        # compute grads
        grads = jax.grad(cost_fn)(params)
        # produce updates
        updates, opt_state = optimizer.update(grads, opt_state)
        # apply updates
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state

    velocity = init_velocity
    for i in range(max_steps):
        velocity, opt_state = train_step(velocity, opt_state)
        current_loss = cost_fn(velocity)
        loss_history.append(current_loss)
        print(f"Iteration {i}, cost={current_loss}, velocity={velocity}")

    return velocity, loss_history

# Visualise the converged trajectory.
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
    #states, distance = simulate_trajectory(mjx_data, INIT_VEL)
    #visualise_trajectory(states, mj_data, mj_model)
    #print(distance)
    #print(states[-1])
    optimal_velocity, loss_history = solve(INIT_VEL)
    states, _ = simulate_trajectory(mjx_data, optimal_velocity)
    #visualise_trajectory(states, mj_data, mj_model)
    plot_loss(loss_history)

if __name__ == "__main__":

    main()