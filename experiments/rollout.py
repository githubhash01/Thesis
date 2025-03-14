from enum import Enum
import argparse
import os
from jax import numpy as jnp
from jax import config
import jax
import pandas as pd
import numpy as np

config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)

class ExperimentType(str, Enum):
    ONE_BOUNCE = "one_bounce"
    TWO_CART = "two_cart"
    FINGER = "finger"

class GradientMode(str, Enum):
    AUTODIFF = "autodiff"
    FD = "fd"
    IMPLICIT_JAXOPT = "implicit_jaxopt"
    IMPLICIT_LAX = "implicit_lax"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

"""
Specify the gradient mode to use for the experiment
"""

parser = argparse.ArgumentParser()
parser.add_argument("--gradient_mode", type=str, default="autodiff", help="solver")
args = parser.parse_args()
# set the MuJoCo solver depending on the gradient mode
os.environ["MJX_SOLVER"] = args.gradient_mode
from simulation import (
    simulate_trajectory,
    visualise_trajectory,
    set_control,
    upscale,
    build_fd_cache,
    make_step_fn_fd_cache,
    make_step_fn
)

"""
Now we can import mujoco modules, having set the solver
"""
import mujoco
from mujoco import mjx


def build_environment(experiment):
    xml_path = os.path.join(BASE_DIR, "xmls", f"{experiment}.xml")
    mj_model = mujoco.MjModel.from_xml_path(filename=xml_path)

    if args.gradient_mode == GradientMode.AUTODIFF:
        mj_model.opt.iterations = 1    # autodiff limits the number of iterations to 1 so as not run out of memory
    else:
        mj_model.opt.iterations = 100

    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    dx_template = mjx.make_data(mj_model)
    mjx_data = jax.tree.map(upscale, dx_template)

    if experiment == ExperimentType.ONE_BOUNCE:
        """
        extremely simple experiment where the ball is dropped from a height of 1.0
        with a velocity of 2.0 downwards and to the right and bounces off the ground
        (no gravity in this experiment) - as it is for comparing against analytic solutions
        """
        mjx_data = mjx_data.replace(
            qpos=jnp.array([-1.0, 0.0, 1.0, 1, 0.0, 0.0, 0.0]),
            qvel=jnp.array([0.2, 0.0, -0.2, 0.0, 0.0, 0.0])
        )

        return mj_model, mj_data, mjx_model, mjx_data

    elif experiment == ExperimentType.TWO_CART:
        """
        left cart starts at -0.5, right cart starts at 0.5
        left cart has a velocity of 10.0 and collides with the right cart
        they end up bouncing off the right wall and coming to a stop
        """
        mjx_data = mjx_data.replace(
            qpos=jnp.array([-0.5, 0.0]),
            qvel=jnp.array([2.0, 0.0])
        )

        return mj_model, mj_data, mjx_model, mjx_data

    elif experiment == ExperimentType.FINGER:
        # update the position and velocity of the finger
        """
        finger starts on top of the spinner with the distal joint recoiled
        distal joint has a velocity placed on it so that it flicks the spinner
        """
        qpos = jnp.array([-1.57079633, -1.57079633, 1.0, 0.0, 0.0, 0.0])
        qvel = jnp.array([1.5, 0.0, 0.0, 0.0, 0.0])
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        return mj_model, mj_data, mjx_model, mjx_data

    else:
        raise ValueError(f"Unknown experiment: {experiment}")

def run_experiment(experiment, visualise=False, save_data=False):
    mj_model, mj_data, mjx_model, mjx_data = build_environment(experiment)

    if args.gradient_mode == GradientMode.FD:
        fd_cache = build_fd_cache(mjx_data)
        step_function = make_step_fn_fd_cache(mjx_model, fd_cache)
    else:
        step_function = make_step_fn(mjx_model)

    # TODO - allow for control inputs to be passed in
    states, state_jacobians, control_jacobians, _= simulate_trajectory(
        mx=mjx_model,
        qpos_init=mjx_data.qpos,
        qvel_init=mjx_data.qvel,
        step_fn=step_function,
        U=jnp.asarray(jnp.zeros((1000, mjx_model.nu)), dtype=jnp.float64))

    if save_data:
        saved_data_dir = os.path.join(BASE_DIR, "stored_data", experiment)
        os.makedirs(saved_data_dir, exist_ok=True)  # Ensure the directory exists
        np.save(os.path.join(saved_data_dir, f"states_{args.gradient_mode}.npy"), states)
        np.save(os.path.join(saved_data_dir, f"state_jacobians_{args.gradient_mode}.npy"), state_jacobians)

    # visualise the trajectory
    if visualise:
        visualise_trajectory(states, mj_data, mj_model)

def just_visualise(experiment):
    mj_model, mj_data, mjx_model, mjx_data = build_environment(experiment)
    states, state_jacobians, _, _ = simulate_trajectory(
        mx=mjx_model,
        qpos_init=mjx_data.qpos,
        qvel_init=mjx_data.qvel,
        set_control_fn=set_control,
        U=jnp.asarray(jnp.zeros((1000, mjx_model.nu)), dtype=jnp.float64))

    visualise_trajectory(states, mj_data, mj_model)


def main():

    #Run all experiments:
    """
    for experiment in ExperimentType:
        print(f"Running: {experiment} using: {args.gradient_mode}\n")
        run_experiment(experiment, visualise=False, save_data=True)
    """
    # Run a single experiment
    experiment = ExperimentType.FINGER
    run_experiment(experiment, visualise=False, save_data=True)

if __name__ == "__main__":
    main()