import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')

import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument("--gradient_mode", type=str, default="autodiff", help="solver")
args = parser.parse_args()
# set the MuJoCo solver depending on the gradient mode
os.environ["MJX_SOLVER"] = args.gradient_mode


import mujoco
from mujoco import mjx
import equinox
#from diff_sim.traj_opt.pmp import PMP, make_loss
from pmp import PMP, make_loss, simulate_trajectory
from viz import visualise_trajectory

def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

if __name__ == "__main__":
    path = "/Users/hashim/Desktop/Thesis/experiments/xmls/finger.xml"
    model = mujoco.MjModel.from_xml_path(path)
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    dx = jax.tree.map(upscale, dx)
    qpos_init = jnp.array([-1.57079633, -1.57079633, 1, 0.0, 0.0, 0.0])
    Nsteps, nu = 300, 2
    U0 = jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu)) * 2

    # target position = [-0.08, 0.0, -0.4] (x, y, z)
    target_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_decoration")
    # keep track of the position of the spinner tip
    spinner_tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
    # keep track of the fingertip position
    finger_tip_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fingertip_site")
    # keep track of the spinner position
    spinner_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "capsule_site")

    def set_control(dx, u):
        # u = jnp.tanh(u) * 0.5
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    def spinner_distance(dx):
        final_position = dx.site_xpos[spinner_tip_site]
        target_position = dx.geom_xpos[target_geom]
        return jnp.linalg.norm(final_position - target_position)


    def terminal_cost(dx):
        return spinner_distance(dx)

    loss_fn = make_loss(mx, qpos_init, set_control, terminal_cost)
    grad_loss_fn = equinox.filter_jit(jax.jacrev(loss_fn))

    optimizer = PMP(loss=loss_fn, grad_loss=grad_loss_fn)
    optimal_U = optimizer.solve(U0, learning_rate=0.5, max_iter=50)

    d = mujoco.MjData(model)

    states, _ = simulate_trajectory(mx, qpos_init, set_control, running_cost, terminal_cost, optimal_U)
    visualise_trajectory(states, d, model)
