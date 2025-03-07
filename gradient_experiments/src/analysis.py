"""
Main file for doing analysis on the data generated from the experiments
"""

import os
import numpy as np
from GradientExperiments.src.visualiser import visualise_traj_generic
from GradientExperiments.src.experiments.one_bounce.config import mj_data, mj_model
import pandas as pd
import matplotlib.pyplot as plt

def print_state_jacobian(jacobian_state, mujoco_model):

    nq = mujoco_model.nq  # expected to be 7
    nv = mujoco_model.nv  # expected to be 6
    # Define labels for qpos and qvel
    #qpos_labels = ['p_x', 'p_y', 'p_z', 'q_w', 'q_x', 'q_y', 'q_z']
    #qvel_labels = ['v_x', 'v_y', 'v_z', 'ω_x', 'ω_y', 'ω_z']

    # Extract blocks from the full Jacobian
    # dq_next/dq: top-left block (nq x nq)
    dq_dq = jacobian_state[:nq, :nq]
    # dq_next/dv: top-right block (nq x nv)
    dq_dv = jacobian_state[:nq, nq:]
    # dv_next/dq: bottom-left block (nv x nq)
    dv_dq = jacobian_state[nq:, :nq]
    # dv_next/dv: bottom-right block (nv x nv)
    dv_dv = jacobian_state[nq:, nq:]

    # Create DataFrames for better formatting in terminal output
    df_dq_dq = pd.DataFrame(dq_dq)
    df_dq_dv = pd.DataFrame(dq_dv)
    df_dv_dq = pd.DataFrame(dv_dq)
    df_dv_dv = pd.DataFrame(dv_dv)

    # Print the blocks with headers
    print("Jacobian Block: dq_next/dq (Position w.r.t. Position)")
    print(df_dq_dq)
    print("\nJacobian Block: dq_next/dv (Position w.r.t. Velocity)")
    print(df_dq_dv)
    print("\nJacobian Block: dv_next/dq (Velocity w.r.t. Position)")
    print(df_dv_dq)
    print("\nJacobian Block: dv_next/dv (Velocity w.r.t. Velocity)")
    print(df_dv_dv)


def build_ground_truth_jacobian(dt):
    """
    Construct a 13x13 Jacobian for a free joint in 3D,
    under the simplified rule:
      p' = p + dt * v
      q' = q   (ignore orientation update)
      v' = v
      w' = w
    Hence:
      dq'/dq = I7,  dq'/dv = [ dt*I3 ; 0 ],
      dv'/dq = 0,   dv'/dv = I6.
    """
    nq, nv = 7, 6  # free joint in 3D
    J_gt = np.zeros((nq + nv, nq + nv))

    # (1) dq'/dq -> top-left block: 7x7 identity
    J_gt[:nq, :nq] = np.eye(nq)

    # (2) dv'/dv -> bottom-right block: 6x6 identity
    J_gt[nq:, nq:] = np.eye(nv)

    # (3) dq'/dv -> top-right block:
    #   - for the first 3 DOFs of qpos (p_x, p_y, p_z),
    #     partial wrt. the first 3 DOFs of velocity (v_x, v_y, v_z) is dt.
    #     That corresponds to rows [0..2], columns [0..2] *within the sub-block*.
    #   - The sub-block itself sits at rows [0..6], columns [7..12] in the full matrix.
    #   - So for i in [0,1,2], j in [0,1,2]:
    for i in range(3):
        J_gt[i, nq + i] = dt

    # (4) dv'/dq -> bottom-left block: 6x7 zero (already zero from initialization)

    return J_gt


def build_ground_truth_states(T, collision_step):
    """
    Constructs the expected state trajectory with interpolation before and after collision.

    Parameters:
    - T: Total number of time steps
    - collision_step: Time step at which collision occurs

    Returns:
    - states_gt: (T, 13) array where each row is [qpos (7), qvel (6)]
    """

    # Initial state (before collision)
    init_state = np.array([-1, 0, 1, 1, 0, 0, 0, 2, 0, -2, 0, 0, 0])

    # Final state (after full trajectory ends)
    final_state = np.array([1, 0, 1, 1, 0, 0, 0, 2, 0, 2, 0, 0, 0])  # v_z flipped

    # Split into pre-collision and post-collision trajectories
    pre_collision = np.linspace(init_state, final_state, collision_step)  # Up to collision
    post_collision = np.linspace(init_state, final_state, T - collision_step)  # After collision

    # Flip velocity in z-direction at collision step
    pre_collision[-1, 9] = -pre_collision[-1, 9]  # Flip v_z at collision
    post_collision[:, 9] = np.abs(post_collision[:, 9])  # Ensure positive v_z

    # Combine full trajectory
    states_gt = np.vstack([pre_collision, post_collision])

    return states_gt


def plot_jacobian_difference_across_time(jacobians, ground_truth, boundaries, title="Jacobian difference across time"):
    """
    Plots the Jacobian difference for pre-collision, collision, and post-collision
    on a SINGLE graph with different colors.

    Parameters:
    - jacobians: list (or array) of shape (T, (nq+nv), (nq+nv))
    - ground_truth: array of shape ((nq+nv), (nq+nv)), analytical Jacobian
    - boundaries: tuple (start_collision, end_collision)
    - title: Plot title
    """

    # Convert jacobians to NumPy array
    jacobians = np.array(jacobians)
    T = jacobians.shape[0]
    time_steps = np.arange(T)

    # Compute Frobenius norm error at each time step
    errors = np.linalg.norm(jacobians - ground_truth, ord='fro', axis=(1,2))

    # Unpack boundaries
    start_collision, end_collision = boundaries

    # Split time steps into three phases
    pre_mask = time_steps < start_collision
    col_mask = (time_steps >= start_collision) & (time_steps < end_collision)
    post_mask = time_steps >= end_collision

    # Set overall y-limits
    y_min, y_max = np.min(errors), np.max(errors)

    # Create figure
    plt.figure(figsize=(10,6))

    # Plot all three phases on the same plot
    plt.plot(time_steps[pre_mask],  errors[pre_mask],  'bo-', label="Pre-Collision")
    plt.plot(time_steps[col_mask],  errors[col_mask],  'ro-', label="Collision")
    plt.plot(time_steps[post_mask], errors[post_mask], 'go-', label="Post-Collision")

    # Labels & Titles
    plt.xlabel("Time Step")
    plt.ylabel("Jacobian Difference (Frobenius Norm)")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # Optional: Adjust y-limits for better visibility
    plt.ylim(y_min, y_max if y_max < 3 else 3)  # Clip max to 3 if large

    plt.show()

    # Print max error stats for debugging
    print(f"Max error pre-collision:  {np.max(errors[pre_mask]):.6e}")
    print(f"Max error during collision:  {np.max(errors[col_mask]):.6e}")
    print(f"Max error post-collision: {np.max(errors[post_mask]):.6e}")

def plot_state_difference_across_time(states, states_gt, collision_step, title="State difference across time"):
    """
    Plots actual vs expected state values (position & velocity) over time.

    Parameters:
    - states: (T, 13) array where each row is [qpos (7), qvel (6)]
    - states_gt: (T, 13) array of expected [qpos, qvel] at each step
    - collision_step: int, time step where collision is expected to occur
    - title: title of the plot

    The first 7 elements represent position (qpos),
    The next 6 elements represent velocity (qvel).
    """

    states = np.array(states)
    states_gt = np.array(states_gt)
    time_steps = np.arange(len(states))

    # Extract position and velocity separately
    qpos_actual, qvel_actual = states[:, :7], states[:, 7:]
    qpos_gt, qvel_gt = states_gt[:, :7], states_gt[:, 7:]

    # Create figure with 2 subplots (qpos & qvel)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Position (qpos) plot
    for i in range(7):
        axs[0].plot(time_steps, qpos_actual[:, i], label=f"qpos {i} (actual)", linestyle='solid')
        axs[0].plot(time_steps, qpos_gt[:, i], label=f"qpos {i} (expected)", linestyle='dashed')

    axs[0].axvline(x=collision_step, color='r', linestyle='dotted', label="Collision")
    axs[0].set_ylabel("Position (qpos)")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title("Position Evolution Over Time")

    # Velocity (qvel) plot
    for i in range(6):
        axs[1].plot(time_steps, qvel_actual[:, i], label=f"qvel {i} (actual)", linestyle='solid')
        axs[1].plot(time_steps, qvel_gt[:, i], label=f"qvel {i} (expected)", linestyle='dashed')

    axs[1].axvline(x=collision_step, color='r', linestyle='dotted', label="Collision")
    axs[1].set_ylabel("Velocity (qvel)")
    axs[1].set_xlabel("Time Step")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title("Velocity Evolution Over Time")

    plt.suptitle(title)
    plt.show()


def main():

    print("Running analysis on the data ...")

    # Load the data
    stored_data_directory = "/Users/hashim/Desktop/Dissertation/GradientExperiments/src/experiments/collision/stored_data"
    #stored_data_directory ="/Users/hashim/Desktop/Dissertation/GradientExperiments/src/experiments/two_bounce/stored_data"

    fd_states = "states_fd.npy"
    fd_jacobians = "jacobians_fd.npy"

    autodiff_states = "states_autodiff.npy"
    autodiff_jacobians = "jacobians_autodiff.npy"

    implicit_states = "states_implicit.npy"
    implicit_jacobians = "jacobians_implicit.npy"

    states = np.load(os.path.join(stored_data_directory, implicit_states))
    jacobians = np.load(os.path.join(stored_data_directory, implicit_jacobians))

    # visualise the trajectory using the states data
    print("Visualising the trajectory ...")
    #visualise_traj_generic(states, mj_data, mj_model)

    # Perform analysis on the data
    print_state_jacobian(jacobians[300], mj_model)

    # Build up the ground truth Jacobian
    J_gt = build_ground_truth_jacobian(dt=0.01)
    # use the ground truth jacobian to plot the difference between the jacobians
    plot_jacobian_difference_across_time(jacobians, J_gt, boundaries=[400, 600], title="Jacobian difference across time")

    # Build up the ground truth states
    states_gt = build_ground_truth_states(T=1000, collision_step=500)
    # use the ground truth states to plot the difference between the states
    #plot_state_difference_across_time(states, states_gt, collision_step=500, title="State difference across time")

    print("Analysis completed")

if __name__ == "__main__":
    main()