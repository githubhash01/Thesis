import os
import argparse
# Parse the gradient mode argument.
parser = argparse.ArgumentParser()
parser.add_argument("--gradient_mode", type=str, default="autodiff", help="solver")
args = parser.parse_args()

# Set the MuJoCo solver depending on the gradient mode.
#os.environ["MJX_SOLVER"] = "fd" #hardcodig the solver to be finite difference
os.environ["MJX_SOLVER"] = args.gradient_mode

import jax
#jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
#jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import equinox
import optax
from diff_sim.traj_opt.policy import (
    simulate_trajectories, make_loss_multi_init, make_step_fn, build_fd_cache, make_step_fn_default, make_step_fn_vjp
)
from diff_sim.utils.math_helper import angle_axis_to_quaternion, quaternion_to_angle_axis


def upscale(x):
    if 'dtype' in dir(x):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x


def generate_inital_conditions(key: jnp.ndarray):
    # Solution of IK
    _, key = jax.random.split(key)
    sign = 2. * jax.random.bernoulli(key, 0.5) - 1.

    # Reference target position in spinner local frame R_s
    r_l = 0.22
    _, key = jax.random.split(key, num=2)
    theta_l = jax.random.uniform(key, (1,), minval=0.6, maxval=2.5)  # Polar Coord
    x_s, y_s = r_l * jnp.cos(theta_l), r_l * jnp.sin(theta_l)  # Cartesian in R_l

    # Reference target position in finger frame R_f
    # Inverse kinematic formula
    x, y = x_s, y_s - 0.39
    l1, l2 = 0.17, 0.161
    q1 = sign * jnp.arccos((x ** 2 + y ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2))
    q0 = jnp.arctan2(y, x) - jnp.arctan2(l2 * jnp.sin(q1), l1 + l2 * jnp.cos(q1))
    _, key = jax.random.split(key, num=2)
    theta = jax.random.uniform(key, (1,), minval=-0.9, maxval=0.9)

    return jnp.concatenate([q0, q1, theta])


# def generate_inital_conditions(key):
#     # Solution of IK
#     _, key = jax.random.split(key)
#     sign = 2.*jax.random.bernoulli(key, 0.5) - 1.

#     # Reference target position in spinner local frame R_s
#     _, key = jax.random.split(key, num=2)
#     theta_l = jax.random.uniform(key, (1,), minval=0.6, maxval=2.5) # Polar Coord
#     l_spinner = 0.22
#     # r = jax.random.uniform(key, (1,), minval=l_spinner + 0.01, maxval=l_spinner + 0.01)
#     r_l = l_spinner
#     x_s,y_s = r_l*jnp.cos(theta_l), r_l*jnp.sin(theta_l) # Cartesian in R_l

#     # Reference target position in finger frame R_f
#     x, y = x_s, y_s - 0.39

#     # Inverse kinematic formula
#     l1, l2 = 0.17, 0.161
#     q1 = sign * jnp.arccos( (x**2 + y**2 - l1**2 - l2**2)/(2*l1*l2) )
#     q0 = jnp.arctan2(y,x) - jnp.arctan2(l2 * jnp.sin(q1), l1 + l2*jnp.cos(q1))

#     # dx = dx.replace(qpos=dx.qpos.at[0].set(q0[0]))
#     # dx = dx.replace(qpos=dx.qpos.at[1].set(q1[0]))

#     # Set Mocap quaternion
#     # _, key = jax.random.split(key)
#     # theta_cap = jax.random.uniform(key, (1,), minval=0., maxval=3.14)
#     # axis = jnp.array([0, -theta_cap[0], 0])
#     # quat = angle_axis_to_quaternion(axis)
#     # # dx = dx.replace(mocap_quat=dx.m
#     # #                 ocap_quat.at[:].set(quat))
#     # pos = jnp.array([-0.2, 0., -0.4])
#     # # dx = dx.replace(mocap_pos=dx.mocap_pos.at[:].set(pos))

#     _, key = jax.random.split(key, num=2)
#     theta = jax.random.uniform(key, (1,), minval=-0.9, maxval=0.9)
#     # dx = dx.replace(qpos=dx.qpos.at[2].set(theta[0]))

#     return jnp.concatenate([q0,q1,theta])

class PolicyNet(equinox.Module):
    layers: list
    act: callable

    def __init__(self, dims: list, key):
        keys = jax.random.split(key, len(dims))
        self.layers = [equinox.nn.Linear(
            dims[i], dims[i + 1], key=keys[i], use_bias=True
        ) for i in range(len(dims) - 1)]
        self.act = jax.nn.relu

    def __call__(self, x, t):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x).squeeze()
        # x = jnp.tanh(x) * 1
        return x


if __name__ == "__main__":
    # Load MuJoCo model
    path = "/Users/hashim/Desktop/Thesis/experiments/finger_learning/finger.xml"
    model = mujoco.MjModel.from_xml_path(path)
    mx = mjx.put_model(model)
    dx_template = mjx.make_data(mx)
    dx_template = jax.tree.map(upscale, dx_template)

    init_key = jax.random.PRNGKey(0)
    n_batch = 50
    n_samples = 1
    Nsteps, nu = 75, 2
    keys = jax.random.split(init_key, n_batch)  # Generate 100 random keys
    qpos_inits0 = jax.vmap(generate_inital_conditions, in_axes=(0))(keys)

    # Repeat the initial conditions `n_samples` times
    # Method 1: Using jax.numpy.repeat
    qpos_inits = jnp.repeat(qpos_inits0, n_samples, axis=0)  # Shape: (n_batch * n_samples, ...)

    # Method 2: Using jax.numpy.tile (if you'd rather replicate the entire batch)
    # qpos_repeated = jnp.tile(qpos_inits, (n_samples, 1))  # Shape: (n_batch * n_samples, ...)
    qvel_inits = jnp.zeros_like(qpos_inits)  # or any distribution you like


    def running_cost(dx):
        cost = 0.0002 * jnp.sum(dx.ctrl ** 2) + 0.001 * dx.qpos[2] ** 2
        return cost


    def terminal_cost(dx):
        cost = 4 * dx.qpos[2] ** 2
        return cost


    def set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))


    # Create the new multi-initial-condition loss function
    loss_fn = make_loss_multi_init(
        mx,
        qpos_inits,  # shape (B*n_sample, n_qpos)
        qvel_inits,  # shape (B*n_sample, n_qpos) or (B, n_qvel)
        set_control_fn=set_control,
        running_cost_fn=running_cost,
        terminal_cost_fn=terminal_cost,
        length=Nsteps,
        batch_size=n_batch,
        sample_size=n_samples
    )

    # JIT the gradient

    # Create your FD cache (if you want it explicitly) or rely on the above
    # fd_cache = build_fd_cache(dx_template, jnp.zeros((mx.nu,)), ...)

    # Create your policy net, optimizer, and do gradient descent
    nn = PolicyNet([6, 64, 128, 128, 64, 2], key=jax.random.PRNGKey(0))
    adam = optax.adamw(3.e-3)
    opt_state = adam.init(equinox.filter(nn, equinox.is_array))

    # Same "Policy" class as before
    from diff_sim.traj_opt.policy import Policy

    optimizer = Policy(loss=loss_fn)
    optimal_nn = optimizer.solve(nn, adam, opt_state, batch_size=n_batch * n_samples, max_iter=50)

    fd_cache = build_fd_cache(dx_template)

    # use the finite difference step function if FD is enabled
    if os.environ.get('MJX_SOLVER') == 'fd':
        print("finger_learning: Using FD-based step function")
        step_fn = make_step_fn(mx, set_control, fd_cache)
    # otherwise use the default step function
    else:
        print("finger_learning: Using default step function")
        step_fn = make_step_fn_default(mx, set_control)
       # step_fn = make_step_fn_vjp(mx, set_control)


    # Evaluate final performance *on the entire batch*`
    params, static = equinox.partition(optimal_nn, equinox.is_array)
    _, subkey = jax.random.split(init_key, num=(2,))
    key_batch = jax.random.split(subkey, num=(n_batch * n_samples,))
    states_batched, _ = simulate_trajectories(
        mx, qpos_inits, qvel_inits,
        running_cost, terminal_cost,
        step_fn=step_fn,
        params=params,
        static=static,
        length=Nsteps,
        keys=key_batch
    )

    # visualize the trajectories
    from diff_sim.utils.mj_viewers import visualise_traj_generic

    data = mujoco.MjData(model)
    visualise_traj_generic(jnp.array(states_batched), data, model)

