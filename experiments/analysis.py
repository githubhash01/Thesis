import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from main import BASE_DIR, ExperimentType, GradientMode
import mujoco


def get_experiment_data(experiment_name, gradient_method):
    saved_data_dir = str(os.path.join(BASE_DIR, "stored_data", experiment_name))
    jacobians_file = f"jacobians_{gradient_method}.npy"
    states_file = f"states_{gradient_method}.npy"
    states = np.load(os.path.join(saved_data_dir, states_file))
    jacobians = np.load(os.path.join(saved_data_dir, jacobians_file))
    return states, jacobians

def print_state_jacobian(jacobian_state, mujoco_model):

    nq = mujoco_model.nq  # expected to be 7
    nv = mujoco_model.nv  # expected to be 6

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

def compare_jacobians(jacobians_a, jacobians_b):
    """
    Compare two sets of Jacobians over time and visualize the error.

    :param jacobians_a: (T, nq+nv, nq+nv) numpy array of accurate Jacobians (e.g., autodiff)
    :param jacobians_b: (T, nq+nv, nq+nv) numpy array of experimental Jacobians (e.g., finite difference)
    """
    assert jacobians_a.shape == jacobians_b.shape, "Jacobians must have the same shape."

    num_timesteps = jacobians_a.shape[0]  # Number of timesteps

    # Compute absolute and Frobenius norm errors per timestep
    abs_error = np.abs(jacobians_a - jacobians_b)  # Element-wise absolute error
    frob_norms = np.linalg.norm(jacobians_a - jacobians_b, ord='fro', axis=(1, 2))  # Frobenius norm per timestep
    mse = np.mean((jacobians_a - jacobians_b) ** 2, axis=(1, 2))  # MSE per timestep

    # === Plot Frobenius Norm Over Time ===
    plt.figure(figsize=(10, 4))
    plt.plot(range(num_timesteps), frob_norms, label=f"Frobenius Norm Difference", color="red")
    plt.xlabel("Time Step")
    plt.ylabel("Error")
    plt.title(f"State Jacobian Error Over Time")
    plt.legend()
    plt.grid()
    plt.show()

    # Return error statistics
    return {
        "mean_absolute_error": np.mean(abs_error),
        "mean_squared_error": np.mean(mse),
        "mean_frobenius_norm": np.mean(frob_norms),
        "max_absolute_error": np.max(abs_error)
    }

def set_one_bounce_analytic(time_steps=1000, timestep_length=0.01):
    """
    Build the analytic Jacobians for the one bounce experiment.
    """
    # Initialize the Jacobians
    nq, nv = 7, 6

    """
    Constant jacobian 

    dq_next/dq: I 
    dq_next/dv: t 
    dv_next/dq: 0
    dv_next/dv: I

    """
    dq_next_dq = np.eye(nq)
    dq_next_dv = np.zeros((nq, nv))
    # edit the first 3 diagonals of dq_next_dv to be timestep_length
    dq_next_dv[0, 0] = timestep_length
    dq_next_dv[1, 1] = timestep_length
    dq_next_dv[2, 2] = timestep_length

    # Correct shape: dv_next_dq should be (nv, nq)
    dv_next_dq = np.zeros((nv, nq))
    dv_next_dv = np.eye(nv)

    # build out the full (nq+nv) x (nq+nv) Jacobian
    jacobian_constant = np.block([
        [dq_next_dq, dq_next_dv],
        [dv_next_dq, dv_next_dv]
    ])

    jacobians_array = np.array([jacobian_constant for _ in range(time_steps)])

    # pick out the 500th timestep and set the jacobian to 0
    jacobians_array[500] = np.zeros((nq + nv, nq + nv))


    # write the jacobians to stored data/one_bounce
    saved_data_dir = os.path.join(BASE_DIR, "stored_data", ExperimentType.ONE_BOUNCE)
    os.makedirs(saved_data_dir, exist_ok=True)  # Ensure the directory exists
    np.save(os.path.join(saved_data_dir, f"jacobians_analytic.npy"), jacobians_array)


def main():

    experiment = ExperimentType.FINGER # setting the experiment to one bounce

    xml_path = os.path.join(BASE_DIR, "xmls", f"{experiment}.xml")
    model = mujoco.MjModel.from_xml_path(filename=xml_path)
    states_fd, jacobians_fd = get_experiment_data(experiment, GradientMode.FD)
    states_ad, jacobians_ad = get_experiment_data(experiment, GradientMode.AUTODIFF)
    states_implicit_jaxopt, jacobians_implicit_jaxopt = get_experiment_data(experiment, GradientMode.IMPLICIT_JAXOPT)
    states_implicit_lax, jacobians_implicit_lax = get_experiment_data(experiment, GradientMode.IMPLICIT_LAX)

    # print a random jacobian
    #print_state_jacobian(jacobians_ad[600], model)

    # compare the jacobians
    error_stats = compare_jacobians(jacobians_ad, jacobians_implicit_jaxopt)
    print(error_stats)



if __name__ == "__main__":
    main()