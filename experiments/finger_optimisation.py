from enum import Enum
import argparse
import os
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')

class GradientMode(str, Enum):
    AUTODIFF = "autodiff"
    FD = "fd"
    IMPLICIT_JAXOPT = "implicit_jaxopt"
    IMPLICIT_LAX = "implicit_lax"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Parse the gradient mode argument.
parser = argparse.ArgumentParser()
parser.add_argument("--gradient_mode", type=str, default="autodiff", help="solver")
args = parser.parse_args()

# Set the MuJoCo solver depending on the gradient mode.
#os.environ["MJX_SOLVER"] = "fd" #hardcodig the solver to be finite difference
os.environ["MJX_SOLVER"] = args.gradient_mode

# Import MuJoCo modules after setting the solver.
import mujoco
from mujoco import mjx
from simulation import (
    make_step_fn_state,
    make_step_fn_fd,
    make_step_fn,
    simulate_with_jacobians,
    simulate_data,
    visualise_trajectory,

    simulate_,
    upscale,
    build_fd_cache,
    set_control,
    make_step_fn_fd_cache, # default step function using dx instead of state
)


# Load the finger.xml environment.
xml_path = os.path.join(BASE_DIR, "xmls", "finger.xml")
mj_model = mujoco.MjModel.from_xml_path(filename=xml_path)
mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)
dx_template = mjx.make_data(mj_model, 'cpu')
mjx_data = jax.tree.map(upscale, dx_template)

fingertip_site = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
target_site = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "target_decoration")

# Choose the appropriate step function.
if args.gradient_mode == GradientMode.FD:
    step_fn = make_step_fn_fd(mjx_model, mjx_data)
else:
    step_fn = make_step_fn(mjx_model, set_control)

def build_qvel(finger_velocity):
    qvel = jnp.zeros(5)
    qvel = qvel.at[0].set(finger_velocity[0])
    qvel = qvel.at[1].set(finger_velocity[1])
    return qvel

def calculate_cost(d: mujoco.MjData):
    # get the location of the fingertip and the target
    finger_pos = d.site_xpos[fingertip_site]
    target_pos = d.geom_xpos[target_site]
    # calculate the distance between the fingertip and the target
    distance = jnp.linalg.norm(finger_pos - target_pos)
    return distance

def simulate_trajectory(dx, finger_velocity):
    qvel = build_qvel(finger_velocity)
    dx = dx.replace(qvel=qvel)
    final_dx = simulate_data(mjx_data=dx, num_steps=300, step_function=step_fn)
    final_cost = calculate_cost(final_dx)
    return final_cost

def make_loss(dx):
    def loss(finger_velocity):
        return simulate_trajectory(dx, finger_velocity)
    return loss

def solve(finger_velocity, learning_rate=1e-2, tol=1e-6, max_iter=10):
    loss = make_loss(dx=mjx_data)
    loss_history = []
    grad_loss = jax.jacrev(loss)
    for i in range(max_iter):
        loss_value = loss(finger_velocity)
        grad_value = grad_loss(finger_velocity)
        finger_velocity = finger_velocity - learning_rate * grad_value
        print(f"Iteration {i}: cost={loss_value}, velocity={finger_velocity}")
        if loss_value < tol:
            break
        loss_history.append(loss_value)
    return finger_velocity, loss_history

def main():

    finger_velocity = jnp.array([0.0, 1.5])
    #build_qvel(finger_velocity)
    optimal_velocity, loss_history = solve(finger_velocity)

    print(f"Optimal finger velocity: {optimal_velocity}")

if __name__ == "__main__":
    main()