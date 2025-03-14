import argparse
import os
import jax
import jax.numpy as jnp
import equinox

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
from simulation import make_step_fn, make_step_fn_fd_cache, build_fd_cache, upscale, simulate_trajectory
import matplotlib.pyplot as plt

# ---------------------------------------- START OF CODE ---------------------------------------- #
xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xmls", "rebound.xml")
mj_model = mujoco.MjModel.from_xml_path(filename=xml_path)
mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)
dx_template = mjx.make_data(mjx_model)
mjx_data = jax.tree.map(upscale, dx_template)

# set the start position of the finger
#mjx_data = mjx_data.replace(qpos=jnp.array([-1.57079633, -1.57079633, 1, 0.0, 0.0, 0.0, 0.0]))


# Define the target position.
TARGET_POS = jnp.array([0.1571356 , 0.0, 1.15331564, 1, 0.0, 0.0, 0.0])
TARGET_POS_CARTESIAN = jnp.array([TARGET_POS[0], TARGET_POS[2]])
INIT_VEL_STAR = jnp.array([2, 0.0, -0.5, 0.0, 0.0, 0.0])  # ground truth initial velocity

if args.gradient_mode == "fd":
    fd_cache = build_fd_cache(mjx_data)
    step_function = make_step_fn_fd_cache(mjx_model, fd_cache)
else:
    step_function = make_step_fn(mjx_model)


# Helper: Build the full 6D velocity from the free 2D parameters.
def build_qvel(ball_vel):
    return jnp.array([ball_vel[0], 0.0, ball_vel[1], 0.0, 0.0, 0.0])

def rollout_trajectory(mx, qpos_init, qvel_init, step_fn, U):

    # rollout the trajectory
    states, _, _, _ = simulate_trajectory(
        mx=mx,
        qpos_init=qpos_init,
        qvel_init=qvel_init,
        step_fn=step_fn,
        U=U
    )

    # calculate the final distance to the target
    final_pos = states[-1][:7]
    final_pos_cartesian = jnp.array([final_pos[0], final_pos[2]])
    final_distance = jnp.sum((final_pos_cartesian - TARGET_POS_CARTESIAN) ** 2)

    return states, final_distance

def make_loss(mx, init_pos, step_fn, U):
    def loss(ball_vel):
        _, total_cost = rollout_trajectory(mx, init_pos, build_qvel(ball_vel), step_fn, U)
        return total_cost
    return loss

def solve(ball_vel, learning_rate=1e-2, tol=1e-4, max_iter=30):

    loss_fn = make_loss(mjx_model, mjx_data.qpos, step_function, U=jnp.zeros((300, 1)))
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))

    loss_history = []  # Record the loss for each iteration.

    for i in range(max_iter):
        g = grad_loss_fn(ball_vel)
        ball_vel = ball_vel + learning_rate * g
        f_val = loss_fn(ball_vel)
        loss_history.append(f_val)
        print(f"Iteration {i}: cost={f_val}, ball initial velocity ={ball_vel}")
        if jnp.linalg.norm(g) < tol or jnp.isnan(g).any():
            return ball_vel

    return ball_vel, loss_history

# Visualise the converged trajectory.
def plot_loss(loss_history):
    # Plot the loss graph.
    plt.figure()
    plt.plot(loss_history, marker='o')
    plt.title(f"Rebound: Trajectory Optimisation Cost - Gradient Mode: {args.gradient_mode}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.show()


def main():
    init_vel = jnp.array([2.5, -0.3])
    U = jnp.zeros((300, 1))
    full_velocity = build_qvel(init_vel)
    states, _, _, dx = simulate_trajectory(
        mx=mjx_model,
        qpos_init=mjx_data.qpos,
        qvel_init=full_velocity,
        step_fn=step_function,
        U=U
    )

    #visualise_trajectory(states, mj_data, mj_model)

    optimal_velocity, loss_history = solve(init_vel)
    states, _, _, _= simulate_trajectory(
        mx=mjx_model,
        qpos_init=mjx_data.qpos,
        qvel_init=build_qvel(optimal_velocity),
        step_fn=step_function,
        U=U
    )
    #visualise_trajectory(states, mj_data, mj_model)
    print(f"Optimal velocity: {optimal_velocity}")
    plot_loss(loss_history)
    # print the final optimal velocity


if __name__ == "__main__":
    main()