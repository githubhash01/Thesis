import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import jax.numpy as jnp
import mujoco
from mujoco import mjx, viewer
import numpy as np
import equinox
import optax
from policy import (
    simulate_trajectories, make_loss_multi_init, make_step_fn, build_fd_cache
)
from base_nn import Network
from math_helper import angle_axis_to_quaternion, quaternion_to_angle_axis
import time # for visualisation

def visualise_traj_generic(
        x, d: mujoco.MjData, m: mujoco.MjModel
):

    with viewer.launch_passive(m, d) as v:
        x = np.array(x)
        for b in range(x.shape[0]):
            for i in range(x.shape[1]):
                step_start = time.time()
                qpos = x[b, i, :m.nq]
                qvel = x[b, i, m.nq:]
                d.qpos[:] = qpos
                d.qvel[:] = qvel
                mujoco.mj_forward(m, d)
                v.sync()
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

def upscale(x):
    if 'dtype' in dir(x):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x


def generate_inital_conditions(key):
    # Solution of IK
    _, key = jax.random.split(key)
    sign = 2. * jax.random.bernoulli(key, 0.5) - 1.

    # Reference target position in spinner local frame R_s
    _, key = jax.random.split(key, num=2)
    theta_l = jax.random.uniform(key, (1,), minval=0.55, maxval=2.55)  # Polar Coord
    l_spinner = 0.22
    # r = jax.random.uniform(key, (1,), minval=l_spinner + 0.01, maxval=l_spinner + 0.01)
    r_l = l_spinner
    x_s, y_s = r_l * jnp.cos(theta_l), r_l * jnp.sin(theta_l)  # Cartesian in R_l

    # Reference target position in finger frame R_f
    x, y = x_s, y_s - 0.39

    # Inverse kinematic formula
    l1, l2 = 0.17, 0.161
    q1 = sign * jnp.arccos((x ** 2 + y ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2))
    q0 = jnp.arctan2(y, x) - jnp.arctan2(l2 * jnp.sin(q1), l1 + l2 * jnp.cos(q1))

    # dx = dx.replace(qpos=dx.qpos.at[0].set(q0[0]))
    # dx = dx.replace(qpos=dx.qpos.at[1].set(q1[0]))

    # Set Mocap quaternion
    _, key = jax.random.split(key)
    theta_cap = jax.random.uniform(key, (1,), minval=0., maxval=3.14)
    axis = jnp.array([0, -theta_cap[0], 0])
    quat = angle_axis_to_quaternion(axis)
    # dx = dx.replace(mocap_quat=dx.m
    #                 ocap_quat.at[:].set(quat))
    pos = jnp.array([-0.2, 0., -0.4])
    # dx = dx.replace(mocap_pos=dx.mocap_pos.at[:].set(pos))

    _, key = jax.random.split(key, num=2)
    theta = jax.random.uniform(key, (1,), minval=-0.9, maxval=0.9)
    # dx = dx.replace(qpos=dx.qpos.at[2].set(theta[0]))

    return jnp.concatenate([q0, q1, theta])


if __name__ == "__main__":

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path("/Users/hashim/Desktop/Thesis/experiments/finger_learning/finger.xml")
    mx = mjx.put_model(model)
    dx_template = mjx.make_data(mx)
    dx_template = jax.tree.map(upscale, dx_template)

    # Suppose we define a batch of initial conditions
    # e.g. 5 different qpos, qvel
    # qpos_inits = jnp.array([
    #     [-0.8,  0.0, -0.8]
    # ])
    # # repeat qpos_inits to match to size 164
    # qpos_inits = jnp.repeat(qpos_inits, 64, axis=0)
    # qpos_inits += 0.01 * jax.random.normal(jax.random.PRNGKey(0), qpos_inits.shape)
    init_key = jax.random.PRNGKey(10)
    n_batch = 200
    keys = jax.random.split(init_key, n_batch)  # Generate 100 random keys
    qpos_inits = jax.vmap(generate_inital_conditions, in_axes=(0))(keys)
    qvel_inits = jnp.zeros_like(qpos_inits)  # or any distribution you like

    Nsteps, nu = 100, 2


    def running_cost(dx):
        pos_finger = dx.qpos[2]
        u = dx.ctrl
        return 0.002 * jnp.sum(u ** 2) + 0.001 * pos_finger ** 2


    def terminal_cost(dx):
        pos_finger = dx.qpos[2]
        return 4 * pos_finger ** 2


    def set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))


    class PolicyNet(Network):
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
            return x


    # Create the new multi-initial-condition loss function
    loss_fn = make_loss_multi_init(
        mx,
        qpos_inits,  # shape (B, n_qpos)
        qvel_inits,  # shape (B, n_qpos) or (B, n_qvel)
        set_control_fn=set_control,
        running_cost_fn=running_cost,
        terminal_cost_fn=terminal_cost,
        length=Nsteps
    )

    # JIT the gradient

    # Create your FD cache (if you want it explicitly) or rely on the above
    # fd_cache = build_fd_cache(dx_template, jnp.zeros((mx.nu,)), ...)

    # Create your policy net, optimizer, and do gradient descent
    nn = PolicyNet([6, 128, 256, 128, 2], key=jax.random.PRNGKey(0))
    adam = optax.adamw(1.5e-3)
    opt_state = adam.init(equinox.filter(nn, equinox.is_array))

    # Same "Policy" class as before
    from diff_sim.traj_opt.policy import Policy

    optimizer = Policy(loss=loss_fn)
    optimal_nn = optimizer.solve(nn, adam, opt_state, batch_size=n_batch, max_iter=100)

    fd_cache = build_fd_cache(dx_template)
    step_fn = make_step_fn(mx, set_control, fd_cache)
    # Evaluate final performance *on the entire batch*
    params, static = equinox.partition(optimal_nn, equinox.is_array)
    _, subkey = jax.random.split(init_key, num=(2,))
    key_batch = jax.random.split(subkey, num=(n_batch,))
    states_batched, cost_batched = simulate_trajectories(
        mx, qpos_inits, qvel_inits,
        running_cost, terminal_cost,
        step_fn=step_fn,
        params=params,
        static=static,
        length=Nsteps,
        keys=key_batch
    )

    # visualize the trajectories
    #from diff_sim.utils.mj import visualise_traj_generic

    data = mujoco.MjData(model)
    visualise_traj_generic(jnp.array(states_batched), data, model)