import argparse
import os
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')

# Parse the gradient mode argument.
parser = argparse.ArgumentParser()
parser.add_argument("--gradient_mode", type=str, default="autodiff", help="solver")
args = parser.parse_args()

# Set the MuJoCo solver depending on the gradient mode.
os.environ["MJX_SOLVER"] = args.gradient_mode

# Import MuJoCo modules after setting the solver.
import mujoco
from mujoco import mjx
from simulation import make_step_fn, simulate_data, visualise_trajectory, upscale
import matplotlib.pyplot as plt

# ---------------------------------------- START OF CODE ---------------------------------------- #
xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xmls", "finger.xml")
mj_model = mujoco.MjModel.from_xml_path(filename=xml_path)
mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)
dx_template = mjx.make_data(mjx_model)
mjx_data = jax.tree.map(upscale, dx_template)

# set the start position of the finger
mjx_data = mjx_data.replace(qpos=jnp.array([-1.57079633, -1.57079633, 1, 0.0, 0.0, 0.0]))

# target position = [-0.08, 0.0, -0.4] (x, y, z)
target_site = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "target_decoration")
# keep track of the position of the spinner tip
spinner_tip_site = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "target_site")

step_function = make_step_fn(mjx_model)

def simulate_trajectory(dx, finger_vel):

    # update the finger's velocity
    dx = dx.replace(qvel=jnp.array([finger_vel[0], finger_vel[1], 0.0, 0.0, 0.0]))

    # simulate the trajectory with the updated velocity
    states, dx = simulate_data(mjx_data=dx, step_function=step_function, num_steps=300)

    # calculate the distance between the final position of the spinner and the target position
    final_position = dx.site_xpos[spinner_tip_site]
    target_position = dx.geom_xpos[target_site]
    distance = jnp.linalg.norm(final_position - target_position)

    return states, distance

# Define a loss function that takes the free velocity parameters.
def cost_function(free_velocity):
    _, distance = simulate_trajectory(mjx_data, free_velocity)
    return distance

def solve(init_finger_vel, learning_rate=5, tol=1e-4, max_iter=10):
    grad_loss = jax.jacrev(cost_function)
    loss_history = []  # Record the loss for each iteration.

    finger_vel = init_finger_vel
    for i in range(max_iter):
        g = grad_loss(finger_vel)
        finger_vel = finger_vel - learning_rate * g
        learning_rate *= 0.8  # Decay the learning rate.
        f_val = cost_function(finger_vel)
        loss_history.append(f_val)
        jax.debug.print("Iteration {}: cost={}, free_velocity={}", i, f_val, finger_vel)
        if jnp.linalg.norm(g) < tol or jnp.isnan(g).any():
            return finger_vel

    return finger_vel, loss_history

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
    init_finger_vel = jnp.array([0.95, -0.2])
    #states, distance = simulate_trajectory(mjx_data, init_finger_vel)
    #visualise_trajectory(states, mj_data, mj_model)

    optimal_velocity, loss_history = solve(init_finger_vel)
    states, _ = simulate_trajectory(mjx_data, optimal_velocity)
    visualise_trajectory(states, mj_data, mj_model)
    #plot_loss(loss_history)

if __name__ == "__main__":
    main()