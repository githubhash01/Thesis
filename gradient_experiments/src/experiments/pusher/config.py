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
class Config:
    simulation_time: float = 10.0  # Total simulation time (s)
    steps: int = 1000  # Number of time steps

    # Initial position (11 DoF: base + orientation + arm joints)
    init_pos = jnp.array([
        -1.0, 0.0, 0.0,  # Base position (x, y, z)
        1.0, 0.0, 0.0, 0.0,  # Base quaternion (w, x, y, z)
        0.0, 0.0, 0.0, 0.0  # Joint angles (4)
    ])

    # Initial velocity (corresponding to 11 DoF)
    init_vel = jnp.array([
        1.0, 1.0, 1.0,  # Base linear velocity (x, y, z)
        0.0, 0.0, 0.0,  # Base angular velocity (x, y, z)
        0.0, 0.0, 0.0, 0.0, 0.0 # Joint velocities (4)
    ])
    # Control input (torques for 7 actuated joints)
    ctrl_input = jnp.zeros(7)

xml_path = "/Users/hashim/Desktop/Dissertation/GradientExperiments/src/experiments/pusher/pusher.xml"
mj_model = mujoco.MjModel.from_xml_path(filename=xml_path)
mj_data = mujoco.MjData(mj_model)
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)

config = Config()

# --- Initialize state ---

mjx_data = mjx_data.replace(
    qpos=config.init_pos,
    qvel=config.init_vel
)