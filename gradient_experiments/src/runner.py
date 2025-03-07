import argparse
import importlib
import os
import time
import numpy as np
from simulator import make_step_fn, make_step_fn_fd, simulate, simulate_scan, make_step_fn_rev
from visualiser import visualise_finger

def runner():
    """Run an experiment using the specified configuration and gradient mode."""
    parser = argparse.ArgumentParser(description="Run a specific experiment")
    parser.add_argument(
        '--experiment', type=str, required=True,
        help='Name of the experiment folder (e.g. one_bounce)'
    )
    parser.add_argument(
        '--gradient_mode', type=str, required=True, choices=['autodiff', 'fd', 'implicit'],
        help='Gradient mode: "autodiff" for automatic differentiation, "fd" for finite differences'
    )
    args = parser.parse_args()

    # Dynamically import the experiment configuration module.
    module_path = f"gradient_experiments.src.experiments.{args.experiment}.config"
    try:
        exp_config = importlib.import_module(module_path)
    except ModuleNotFoundError:
        parser.error(f"Experiment '{args.experiment}' not found")

    print(f"Running experiment: {args.experiment}")

    # Choose the step function based on the gradient mode.
    if args.gradient_mode == "autodiff":
        step_fn = make_step_fn(exp_config.mjx_model, exp_config.mjx_data)
        states, jacobians = simulate(
            mjx_data=exp_config.mjx_data,
            num_steps=exp_config.config.steps,
            step_function=step_fn
        )

    if args.gradient_mode == "fd":
        step_fn = make_step_fn_fd(exp_config.mjx_model, exp_config.mjx_data)
        states, jacobians = simulate(
            mjx_data=exp_config.mjx_data,
            num_steps=exp_config.config.steps,
            step_function=step_fn
        )
    elif args.gradient_mode == "implicit":
        step_fn = make_step_fn(exp_config.mjx_model, exp_config.mjx_data)
        states, jacobians = simulate(
            mjx_data=exp_config.mjx_data,
            num_steps=exp_config.config.steps,
            step_function=step_fn
        )
        # i need to also modify the solver or use a different solver

    # Generate a timestamp and ensure the stored_data directory exists.
    #timestamp = time.strftime("%Y_%m_%d_%H_%M")
    os.makedirs(exp_config.stored_data_directory, exist_ok=True)

    # hardcoding the gradient mode as just using the solverIFT
    gradient_mode = args.gradient_mode
    # Save the states and jacobians.
    np.save(os.path.join(exp_config.stored_data_directory, f"states_{gradient_mode}.npy"), states)
    np.save(os.path.join(exp_config.stored_data_directory, f"jacobians_{gradient_mode}.npy"), jacobians)

    jacobians = np.load(os.path.join(exp_config.stored_data_directory, f"jacobians_{gradient_mode}.npy"))

    # Print a sample jacobian.

    #for j in jacobians:
        #print_state_jacobian(jacobian_state=j, mujoco_model=exp_config.mj_model)

    # Visualise the trajectory.
    #print("Visualising the trajectory ...")
    visualise_finger(states, exp_config.mj_data, exp_config.mj_model)


if __name__ == "__main__":
    runner()