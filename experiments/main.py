import numpy as np
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import argparse
import os
import jax
from simulation import (
    upscale,
    set_control,
    make_step_fn,
    make_step_fn_default,
    make_step_fn_fd,
    build_fd_cache,
    simulate,
    visualise_trajectory
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def build_environment(experiment_name):
    models_dir = os.path.join(BASE_DIR, "xmls")
    xml_path = os.path.join(models_dir, f"{experiment_name}.xml")
    mj_model = mujoco.MjModel.from_xml_path(filename=xml_path)
    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

    if experiment_name == "one_bounce":
        """
        extremely simple experiment where the ball is dropped from a height of 1.0
        with a velocity of 2.0 downwards and to the right and bounces off the ground
        (no gravity in this experiment) - as it is for comparing against analytic solutions
        """
        mjx_data = mjx_data.replace(
            qpos=jnp.array([-1.0, 0.0, 1.0, 1, 0.0, 0.0, 0.0]),
            qvel=jnp.array([2.0, 0.0, -2.0, 0.0, 0.0, 0.0])
        )

        return mj_model, mj_data, mjx_model, mjx_data

    elif experiment_name == "two_cart":
        """
        left cart starts at -0.5, right cart starts at 0.5
        left cart has a velocity of 2.0 and collides with the right cart
        they end up bouncing off the right wall and coming to a stop
        """
        mjx_data = mjx_data.replace(
            qpos=jnp.array([-0.5, 0.0]),
            qvel=jnp.array([2.0, 0.0])
        )

        return mj_model, mj_data, mjx_model, mjx_data

    elif experiment_name == "finger":
        # update the position and velocity of the finger
        """
        finger starts on top of the spinner with the distal joint recoiled
        distal joint has a velocity placed on it so that it flicks the spinner
        """

        qpos = jnp.array([-1.57079633, -1.57079633, 1.0, 0.0, 0.0, 0.0])
        qvel = jnp.zeros(5)
        qvel = qvel.at[0].set(3) # flick the spinner
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        return mj_model, mj_data, mjx_model, mjx_data

    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

def run_experiment(experiment_name, gradient_mode, visualise=False):
    mj_model, mj_data, mjx_model, mjx_data = build_environment(experiment_name)

    # TODO - DANIEL'S FD IMPLEMENTATION
    """
    Code for Daniel's FD Implementation 
    if gradient_mode == "fd":
        dx_template = mjx.make_data(mjx_model)
        dx_template = jax.tree.map(upscale, dx_template)
        fd_cache = build_fd_cache(dx_template)
        step_function = make_step_fn_fd(mjx_model, set_control, fd_cache)
        
    """

    if gradient_mode == "fd":
        step_function = make_step_fn_fd(mjx_model, mjx_data)

    else:
        step_function = make_step_fn(mjx_model, mjx_data)

    states, jacobians = simulate(
        mjx_data=mjx_data,
        num_steps=1000,
        step_function=step_function
    )

    saved_data_dir = os.path.join(BASE_DIR, "stored_data", experiment_name)
    os.makedirs(saved_data_dir, exist_ok=True)  # Ensure the directory exists
    np.save(os.path.join(saved_data_dir, f"states_{gradient_mode}.npy"), states)
    np.save(os.path.join(saved_data_dir, f"jacobians_{gradient_mode}.npy"), jacobians)

    # visualise the trajectory
    if visualise:
        visualise_trajectory(states, mj_data, mj_model)

def main():

    # parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", type=str, default="default", help="Solver can be default or implicit")
    args = parser.parse_args()

    experiments = ["one_bounce", "two_cart", "finger"]
    standard_solver_gradient_modes = ["autodiff", "fd"]

    if args.solver == "default":
        for experiment in experiments:
            for gradient_mode in standard_solver_gradient_modes:
                jax.debug.print("Experiment: {} Gradient Mode: {} ", experiment, gradient_mode)
                run_experiment(experiment, gradient_mode)

    elif args.solver == "implicit_jaxopt":
        gradient_mode = "implicit_jaxopt"
        for experiment in experiments:
            jax.debug.print("Experiment: {} Gradient Mode: {} ", experiment, gradient_mode)
            run_experiment(experiment, gradient_mode)

    elif args.solver == "implicit_lax":
        gradient_mode = "implicit_lax"
        for experiment in experiments:
            jax.debug.print("Experiment: {} Gradient Mode: {} ", experiment, gradient_mode)
            run_experiment(experiment, gradient_mode)

if __name__ == "__main__":
    main()