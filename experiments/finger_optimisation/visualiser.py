"""
We should decide which solver to use before importing mujoco
"""
import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("--gradient_mode", type=str, default="autodiff", help="solver")
args = parser.parse_args()
# set the MuJoCo solver depending on the gradient mode
os.environ["MJX_SOLVER"] = args.gradient_mode

import mujoco
from mujoco import viewer
from mujoco import mjx
import time
import numpy as np
import jax
import equinox
from jax import numpy as jnp, config
config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import pandas as pd


"""
Helper Functions
"""
def upscale(x):
    if hasattr(x, "dtype"):
        # If x is a JAX array and has an integer type, cast it to float64.
        if jnp.issubdtype(x.dtype, jnp.integer):
            return x.astype(jnp.float64)
        # If x is float32, cast it to float64.
        elif x.dtype == jnp.float32:
            return x.astype(jnp.float64)
        else:
            return x
    elif isinstance(x, int):
        # Convert Python ints to a JAX array of type float64.
        return jnp.array(x, dtype=jnp.float64)



@equinox.filter_jit
def simulate_trajectory(mx, qpos_init, qvel_init, step_fn, U):
    """
    Simulate a trajectory given a control sequence U.

    Args:
        mx: The MuJoCo model handle (static).
        qpos_init: initial positions (array).
        qvel_init: initial velocities (array).
        step_fn: function (dx, u) -> dx_next to step the simulation (e.g. with FD or without)
        U: (N, nu) array of controls.

    Returns:
        states: (N, nq+nv) array of states.
        state_jacobians: (N, nq+nv, nq+nv) array of state transition Jacobians.
        control_jacobians: (N, nq+nv, nu) array of control Jacobians.
    """
    def step_state(dx, u):
        dx_next = step_fn(dx, u)
        # Reapply upscale so that every field is float.
        dx_next = jax.tree.map(upscale, dx_next)
        state = jnp.concatenate([dx_next.qpos, dx_next.qvel]).astype(jnp.float64)
        return dx_next, state

    # Helper function that returns just the differentiable output.
    def f(dx, u):
        return step_state(dx, u)[1]

    # Compute the Jacobians of f with respect to dx and u.
    J_f = jax.jacrev(f, argnums=(0, 1))

    def step_fn_jac(dx, u):
        dx, state = step_state(dx, u)
        jac_state, jac_control = J_f(dx, u)
        # Here, jac_state is a Data-like structure with the same fields as dx.
        # Extract the derivatives with respect to qpos and qvel and flatten them.
        flat_jac_state = jnp.concatenate([jac_state.qpos, jac_state.qvel], axis=-1)
        return dx, (state, flat_jac_state, jac_control)

    # Build the initial data structure and upscale all its fields.
    dx0 = jax.tree.map(upscale, mjx.make_data(mx))
    dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))
    dx0 = dx0.replace(qvel=dx0.qvel.at[:].set(qvel_init))

    dx_final, (states, state_jacobians, control_jacobians) = jax.lax.scan(step_fn_jac, dx0, U)
    return states, state_jacobians, control_jacobians, dx_final

def visualise_trajectory(states, d: mujoco.MjData, m: mujoco.MjModel, sleep=0.01):

    states_np = [np.array(s) for s in states]
    #spinner_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target_site")
    #target_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target_decoration")

    with viewer.launch_passive(m, d) as v:
        for s in states_np:
            step_start = time.time()
            # Extract qpos and qvel from the state.
            nq = m.nq
            # We assume the state was produced as jnp.concatenate([qpos, qvel])
            qpos = s[:nq]
            qvel = s[nq:nq + m.nv]
            d.qpos[:] = qpos
            d.qvel[:] = qvel
            mujoco.mj_forward(m, d)
            #print(d.site_xpos[spinner_site])
            # get the distance between the spinner and the target
            #distance = np.linalg.norm(d.site_xpos[spinner_site] - d.geom_xpos[target_site])
            #print(f"Distance in Viz: {distance}")
            v.sync()
            # Optionally sleep to mimic real time.
            time.sleep(sleep)
            # Optionally, adjust to match the simulation timestep:
            #time_until_next = m.opt.timestep - (time.time() - step_start)
            #if time_until_next > 0:
            #    time.sleep(time_until_next)

"""
Standard step function
"""
def set_control(dx, u):
    u = jnp.asarray(u, dtype=jnp.float64)  # Ensure JAX-tracked array
    return dx.replace(ctrl=dx.ctrl.at[:].set(u))

def make_step_fn(mx, set_control_fn=set_control):

    def step_fn(dx: mjx.Data, u: jnp.ndarray):
        """
        Forward pass:
          1) Writes 'u' into dx_init (or a copy thereof) via set_control_fn.
          2) Steps the simulation forward one step with MuJoCo.
        """
        dx_with_ctrl = set_control_fn(dx, u)
        dx_next = mjx.step(mx, dx_with_ctrl)
        return dx_next

    return step_fn

"""
Code for analysis
"""

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

def print_control_jacobian(jacobian_control, mujoco_model):
    pass

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

def compare_jacobians_all(
        jacobians_dict,
        baseline_key="fd"
):
    """
    Compare multiple sets of Jacobians over time against a baseline method.

    Args:
        jacobians_dict: dict mapping method_name -> (T, nq+nv, nq+nv) array
            of Jacobians. For example:
            {
              "autodiff": jac_autodiff,
              "finite_diff": jac_fd,
              "implicit_lax": jac_lax,
              "implicit_jaxopt": jac_jaxopt
            }
        baseline_key: the method name in jacobians_dict to treat as baseline
            for computing differences.

    Returns:
        stats_dict: a dictionary mapping each method -> summary stats,
            e.g. mean and max Frobenius-norm differences.
    """
    # -- 1. Extract the baseline Jacobians --
    baseline_jac = jacobians_dict[baseline_key]
    T = baseline_jac.shape[0]

    # -- 2. Prepare to plot --
    plt.figure(figsize=(10, 4))

    # We'll store stats for each method here:
    stats_dict = {}

    # -- 3. Compare each method to the baseline --
    for method_name, jacobian_array in jacobians_dict.items():
        if method_name == baseline_key:
            continue  # skip comparing baseline to itself

        # Check shapes match
        if jacobian_array.shape != baseline_jac.shape:
            raise ValueError(f"Shape mismatch between {baseline_key} and {method_name}")

        # Compute Frobenius norm difference at each timestep
        diff = baseline_jac - jacobian_array
        frob_norms = np.linalg.norm(diff, ord='fro', axis=(1, 2))

        # Plot the difference
        plt.plot(range(T), frob_norms, label=f"{method_name} vs {baseline_key}")

        # Record summary stats
        stats_dict[method_name] = {
            "mean_frobenius_diff": np.mean(frob_norms),
            "max_frobenius_diff": np.max(frob_norms),
        }

    # -- 4. Finalize the plot --
    plt.xlabel("Time Step")
    plt.ylabel("Frobenius Norm of Error")
    plt.title("Comparison of Jacobians vs Baseline")
    plt.legend()
    plt.grid(True)
    plt.show()

    return stats_dict

def main():

    model = mujoco.MjModel.from_xml_path("/Users/hashim/Desktop/Thesis/experiments/finger_optimisation/finger.xml")
    mjx_model = mjx.put_model(model)
    data = mujoco.MjData(model)
    dx = mjx.make_data(mjx_model)
    dx = jax.tree.map(upscale, dx)

    qpos_init = jnp.array([-.8, 0, -.8])

    # read the optimal U from the file
    optimal_U = np.load(f"U_fd.npy")

    if args.gradient_mode == "fd":
        from pmp import prepare_sensitivity, make_step_fn_fd
        unravel_dx, inner_dx, sensitivity_mask = prepare_sensitivity(dx, target_fields={'qpos', 'qvel', 'ctrl'})
        step_function = make_step_fn_fd(mjx_model, set_control, unravel_dx, inner_dx, sensitivity_mask)
    else:
        step_function = make_step_fn(mjx_model)

    # simulate the trajectory
    states, state_jacobians , control_jacobians, dx_final = simulate_trajectory(
        mjx_model,
        qpos_init,
        data.qvel,
        step_function,
        optimal_U
    )

    # save the state jacobians and the control jacobians as numpy arrays
    np.save(f"states_{args.gradient_mode}.npy", states)
    np.save(f"state_jacobians_{args.gradient_mode}.npy", state_jacobians)
    np.save(f"control_jacobians_{args.gradient_mode}.npy", control_jacobians)

    #visualise_trajectory(states, data, model)

def analyse():
    model = mujoco.MjModel.from_xml_path("/Users/hashim/Desktop/Thesis/experiments/finger_optimisation/finger.xml")
    # Load the state jacobians for both methods
    state_jacobians_fd = np.load("state_jacobians_fd.npy")
    state_jacobians_autodiff = np.load("state_jacobians_autodiff.npy")
    state_jacobians_implicit_jaxopt = np.load("state_jacobians_implicit_jaxopt.npy")
    state_jacobians_implicit_lax = np.load("state_jacobians_implicit_lax.npy")
    #compare_jacobians(state_jacobians_fd, state_jacobians_implicit_jaxopt)

    control_jacobians_fd = np.load("control_jacobians_fd.npy")
    control_jacobians_autodiff = np.load("control_jacobians_autodiff.npy")
    control_jacobians_implicit_jaxopt = np.load("control_jacobians_implicit_jaxopt.npy")
    control_jacobians_implicit_lax = np.load("control_jacobians_implicit_lax.npy")

    #compare_jacobians(control_jacobians_fd, control_jacobians_implicit_jaxopt)
    print(state_jacobians_implicit_jaxopt[200])
    print(control_jacobians_implicit_jaxopt[200])

if __name__ == "__main__":
    #main()
    analyse()

