import jax
import argparse
from enum import Enum
import os
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
jax.config.update('jax_default_matmul_precision', 'high')
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

import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import time
from mujoco import viewer

from simulation import simulate, make_step_fn

"""
Helper Functions
"""
def upscale(x):
    if 'dtype' in dir(x):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

def load_environment(xml_path):
    mj_model = mujoco.MjModel.from_xml_path(filename=xml_path)
    mj_data = mujoco.MjData(mj_model)
    mjx_model = mjx.put_model(mj_model)
    dx_template = mjx.make_data(mj_model, 'cpu')
    mjx_data = jax.tree.map(upscale, dx_template)
    return mj_model, mj_data, mjx_model, mjx_data

#@jax.jit
def step_function(mx, dx, control):
    dx.replace(ctrl=dx.ctrl.at[:].set(control))
    dx_new = mjx.step(mx, dx)
    return dx_new


def visualise_trajectory(states, d, m, sleep=0.01):
    with viewer.launch_passive(m, d) as v:
        for s in states:
            nq = m.nq
            nv = m.nv
            qpos = s[:nq]
            qvel = s[nq:nq + m.nv]
            d.qpos[:] = qpos
            d.qvel[:] = qvel
            mujoco.mj_forward(m, d)
            v.sync()
            time.sleep(sleep)

        # now close the window
        v.close()
"""
Loading in the data globally
"""

mj_model, mj_data, mjx_model, mjx_data = load_environment("/Users/hashim/Desktop/Thesis/experiments/xmls/finger.xml")
qpos = jnp.array([-1.57079633, -1.57079633, 1, 0.0, 0.0, 0.0]) # initial position of the finger
# update the position of the finger
mjx_data = mjx_data.replace(qpos=qpos)
spinner_tip = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
target_geom = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "target_decoration")

TARGET_POS = jnp.array([-0.08, 0.0, -0.4])  # target position of the spinner
"""
Trajectory Optimisation
"""

def build_full_velocity(finger_vel):
    return jnp.array([finger_vel[0], finger_vel[1], 0.0, 0.0, 0.0])


def simulate_dx(dx, mx, num_steps: int):
    states = []
    for _ in range(num_steps):
        dx = step_function(mx, dx, dx.ctrl)
        state = jnp.concatenate([dx.qpos, dx.qvel])
        states.append(state)

    spinner_pos = dx.qpos[3:]
    jax.debug.print("Spinner position: {}", spinner_pos)
    final_cost = jnp.linalg.norm(spinner_pos - TARGET_POS)
    return states, final_cost

# simulate the trajectory of the finger with some given initial velocity
def simulate_trajectory(dx, mx, finger_vel:jnp.ndarray, steps:int):
    full_velocity = build_full_velocity(finger_vel)
    dx = dx.replace(qvel=full_velocity)
    states, _ = simulate(dx, steps, make_step_fn(mx, dx))
    final_state = states[-1]
    spinner_pos = final_state[3:]
    cost = jnp.linalg.norm(spinner_pos - TARGET_POS)
    return states, cost

# loss function for optimising the initial velocity
def make_loss(dx, mx):
    def loss(finger_velocity):
        _, cost = simulate_trajectory(dx, mx, finger_velocity, 500)
        return cost
    return loss

def solve(finger_velocity, learning_rate=1e-2, iterations=10):
    loss = make_loss(mjx_data, mjx_model)
    grad_loss = jax.jacrev(loss)
    loss_history = []

    for i in range(iterations):
        loss_val = loss(finger_velocity)
        loss_history.append(loss_val)
        gradient = grad_loss(finger_velocity)
        jax.debug.print("Gradient: {}", gradient)
        finger_velocity = finger_velocity - learning_rate * gradient
        print(f"Iteration {i}: cost={loss_val}, velocity={finger_velocity}")

    return finger_velocity, loss_history


def main():

    finger_velocity = jnp.array([1.2, -1.0]) # initial velocity of the finger
    #states, cost = simulate_trajectory(mjx_data, mjx_model, finger_velocity, 500)
    #visualise_trajectory(states, mj_data, mj_model)

    optimal_finger_velocity, loss_history = solve(finger_velocity)

    # simulate the trajectory with the optimal initial velocity
    states, cost = simulate_trajectory(mjx_data, mjx_model, optimal_finger_velocity, 500)
    # visualise the trajectory
    visualise_trajectory(states, mj_data, mj_model)

if __name__ == "__main__":
    main()
