from enum import Enum
import argparse
import os
from jax import numpy as jnp
import numpy as np


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
"""
Now we can import mujoco modules, having set the solver
"""
import mujoco
from mujoco import mjx
from simulation import (
    make_step_fn,
    make_step_fn_fd,
    simulate,
    visualise_trajectory,

    upscale,
    build_fd_cache,
    set_control,
    make_step_fn_default, # default step function using dx instead of state
    make_step_fn_fd_cache,
    simulate_ # simulate function that uses the default step function
)


def build_environment(experiment):
    xml_path = os.path.join(BASE_DIR, "xmls", f"{experiment}.xml")
    #xml_path = "/Users/hashim/Desktop/Thesis/mjx/diff_sim/xmls/finger_mjx.xml"
    mj_model = mujoco.MjModel.from_xml_path(filename=xml_path)
    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mj_data)

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
        qvel = jnp.zeros(5)
        qvel = qvel.at[0].set(1.5) # flick the spinner
        mjx_data = mjx_data.replace(qpos=qpos, qvel=qvel)
        return mj_model, mj_data, mjx_model, mjx_data

    else:
        raise ValueError(f"Unknown experiment: {experiment}")

def run_experiment(experiment, visualise=False):
    mj_model, mj_data, mjx_model, mjx_data = build_environment(experiment)

    # TODO - DANIEL'S FD IMPLEMENTATION
    """
    if args.gradient_mode == GradientMode.FD:
        dx_template = mjx.make_data(mjx_model)
        dx_template = jax.tree.map(upscale, dx_template)
        fd_cache = build_fd_cache(dx_template)
        step_function = make_step_fn_fd_cache(mjx_model, set_control, fd_cache)
    """

    if args.gradient_mode == GradientMode.FD:
        # alternatively use the standard finite difference method
        step_function = make_step_fn_fd(mjx_model, mjx_data)

    else:
        step_function = make_step_fn(mjx_model, mjx_data)

    states, jacobians = simulate(
        mjx_data=mjx_data,
        num_steps=1000,
        step_function=step_function
    )

    saved_data_dir = os.path.join(BASE_DIR, "stored_data", experiment)
    os.makedirs(saved_data_dir, exist_ok=True)  # Ensure the directory exists
    np.save(os.path.join(saved_data_dir, f"states_{args.gradient_mode}.npy"), states)
    np.save(os.path.join(saved_data_dir, f"jacobians_{args.gradient_mode}.npy"), jacobians)

    # visualise the trajectory
    if visualise:
        visualise_trajectory(states, mj_data, mj_model)


def main():

    """
    Run all experiments:

    for experiment in ExperimentType:
        print(f"Running: {experiment} using: {args.gradient_mode}\n")
        run_experiment(experiment, visualise=False)

    """
    run_experiment(ExperimentType.FINGER, visualise=True)


if __name__ == "__main__":
    main()