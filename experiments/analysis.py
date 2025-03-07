import pandas as pd
import numpy as np
import os
from main import BASE_DIR, build_environment

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

def get_states_jacobians(experiment_name, gradient_method):
    saved_data_dir = str(os.path.join(BASE_DIR, "stored_data", experiment_name))
    jacobians_file = f"jacobians_{gradient_method}.npy"
    states_file = f"states_{gradient_method}.npy"
    states = np.load(os.path.join(saved_data_dir, states_file))
    jacobians = np.load(os.path.join(saved_data_dir, jacobians_file))
    return states, jacobians

def main():
    states, jacobians = get_states_jacobians("one_bounce", "fd")
    print(len(states), len(jacobians))
    # print a random jacobian
    model, _, _, _ = build_environment("one_bounce")

    print_state_jacobian(jacobians[0], model)

if __name__ == "__main__":
    main()