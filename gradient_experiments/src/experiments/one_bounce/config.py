from dataclasses import dataclass
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import os

# Get the absolute path of the current file's directory.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the stored_data directory relative to this file.
stored_data_directory = os.path.join(BASE_DIR, "stored_data")

# --- Configuration ---
@dataclass
class Config():     
    simulation_time: float = 1.0    # Duration of simulation
    steps: int = 1000               # Number of simulation steps
    init_pos: jnp.ndarray = jnp.array([-1.0, 1.0])  # Initial position of the ball
    init_vel: jnp.ndarray = jnp.array([2.0, -2.0])  # Initial velocity of the ball
    ctrl_input: jnp.ndarray = jnp.array([0.0, 0.0])

# --- Load model and data ---
xml_path = "/Users/hashim/Desktop/Dissertation/GradientExperiments/src/experiments/one_bounce/ball.xml"
mj_model = mujoco.MjModel.from_xml_path(filename=xml_path)
mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

# --- Initialize state ---
config = Config()
mjx_data = mjx_data.replace(
    qpos=jnp.array([config.init_pos[0], 0.0, config.init_pos[1], 1, 0.0, 0.0, 0.0]),
    # p_x, p_y, p_z, quat_w, quat_x, quat_y, quat_z
    qvel=jnp.array([config.init_vel[0], 0.0, config.init_vel[1], 0.0, 0.0, 0.0])  # v_x, v_y, v_z, w_x, w_y, w_z
)
