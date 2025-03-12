import jax
import jax.numpy as jnp
from mujoco import mjx
from jax import config
from dataclasses import dataclass
from typing import Callable, Optional, Set
import mujoco
from jax.flatten_util import ravel_pytree
import numpy as np
from jax._src.util import unzip2
import time
from mujoco import viewer

config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)


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

def set_control(dx, u):
    return dx.replace(ctrl=dx.ctrl.at[:].set(u))

"""
Hashim's Step Functions 
"""

# Default - Autodiff
def make_step_fn(model, mjx_data):

    @jax.jit
    def step_fn(state):
        nq = model.nq
        qpos, qvel = state[:nq], state[nq:]
        dx = mjx_data.replace(qpos=qpos, qvel=qvel)
        #dx_next = mjx.step(model, dx)
        dx_next = mjx.forward(model, dx) # apparently necessary to update the xpos
        next_state = jnp.concatenate([dx_next.qpos, dx_next.qvel])
        return next_state

    return step_fn

# --- Finite Differences ---
def make_step_fn_fd(mjx_model, mjx_data):
    epsilon = 1e-6

    # This function performs the actual step computation.
    @jax.jit
    def _step_fn(state):
        nq = mjx_model.nq
        qpos, qvel = state[:nq], state[nq:]
        dx = mjx_data.replace(qpos=qpos, qvel=qvel)
        dx_next = mjx.step(mjx_model, dx)
        next_state = jnp.concatenate([dx_next.qpos, dx_next.qvel])
        return next_state

    # Define the custom_vjp-decorated function.
    @jax.custom_vjp
    def step_fn(state):
        return _step_fn(state)

    # Forward pass: compute the output and return the input (or any auxiliary data)
    def step_fn_fwd(s):
        f_s = _step_fn(s)
        return f_s, s  # saving s for use in backward pass

    # Backward pass: given the saved s and the incoming cotangent,
    # compute the vector-Jacobian product using finite differences.
    def step_fn_bwd(s, cotangent):
        #jax.debug.print("Using finite differences for VJP")
        f_s = _step_fn(s)  # compute baseline output once
        grad = []
        for j in range(s.shape[0]):
            # Create a perturbation along the j-th coordinate.
            e_j = jnp.zeros_like(s).at[j].set(1.0)
            # Evaluate the function at the perturbed state.
            f_perturbed = _step_fn(s + epsilon * e_j)
            # Approximate the partial derivative for coordinate j.
            diff = (f_perturbed - f_s) / epsilon
            # Multiply by the cotangent to get the j-th component of the VJP.
            grad_j = jnp.vdot(cotangent, diff)
            grad.append(grad_j)
        grad = jnp.stack(grad)
        return (grad,)

    # Register the forward and backward functions with the custom_vjp mechanism.
    step_fn.defvjp(step_fn_fwd, step_fn_bwd)
    return step_fn

"""
Daniel's FD-based step function
"""

# -------------------------------------------------------------
# Finite-difference cache
# -------------------------------------------------------------

def make_step_fn_default(mjx_model):

    @jax.jit
    def step_fn(mjx_data: mjx.Data, u: jnp.ndarray):
        dx = mjx_data.replace(ctrl=u)
        dx = mjx.step(mjx_model, dx)
        dx = mjx.forward(mjx_model, dx)
        return dx

    return step_fn

@dataclass(frozen=True)
class FDCache:
    """Holds all the precomputed info needed by the custom FD-based backward pass."""
    unravel_dx: Callable[[jnp.ndarray], mjx.Data]
    sensitivity_mask: jnp.ndarray
    inner_idx: jnp.ndarray
    dx_size: int
    num_u_dims: int
    eps: float = 1e-6


def build_fd_cache(
        dx_ref: mjx.Data,
        target_fields: Optional[Set[str]] = None,
        eps: float = 1e-6
) -> FDCache:
    """
    Build a cache containing:
      - Flatten/unflatten for dx_ref
      - The mask for relevant FD indices (e.g. qpos, qvel, ctrl)
      - The shape info for control
    """
    if target_fields is None:
        target_fields = {"qpos", "qvel", "ctrl"}

    # Flatten dx
    dx_array, unravel_dx = ravel_pytree(dx_ref)
    dx_size = dx_array.shape[0]
    num_u_dims = dx_ref.ctrl.shape[0]

    # Gather leaves for qpos, qvel, ctrl
    leaves_with_path = list(jax.tree_util.tree_leaves_with_path(dx_ref))
    sizes, _ = unzip2((jnp.size(leaf), jnp.shape(leaf)) for (_, leaf) in leaves_with_path)
    indices = tuple(np.cumsum(sizes))

    idx_target_state = []
    for i, (path, leaf_val) in enumerate(leaves_with_path):
        # Check if any level in the path has a 'name' that is in target_fields
        name_matches = any(
            getattr(level, 'name', None) in target_fields
            for level in path
        )
        if name_matches:
            idx_target_state.append(i)

    def leaf_index_range(leaf_idx):
        start = 0 if leaf_idx == 0 else indices[leaf_idx - 1]
        end = indices[leaf_idx]
        return np.arange(start, end)

    # Combine all relevant leaf sub-ranges
    inner_idx_list = []
    for i in idx_target_state:
        inner_idx_list.append(leaf_index_range(i))
    inner_idx = np.concatenate(inner_idx_list, axis=0)
    inner_idx = jnp.array(inner_idx, dtype=jnp.int32)

    # Build the sensitivity mask
    sensitivity_mask = jnp.zeros_like(dx_array).at[inner_idx].set(1.0)

    return FDCache(
        unravel_dx=unravel_dx,
        sensitivity_mask=sensitivity_mask,
        inner_idx=inner_idx,
        dx_size=dx_size,
        num_u_dims=num_u_dims,
        eps=eps
    )


# -------------------------------------------------------------
# Step function with custom FD-based derivative
# -------------------------------------------------------------
def make_step_fn_fd_cache(
        mx,
        set_control_fn: Callable,
        fd_cache: FDCache
):
    """
    Create a custom_vjp step function that takes (dx, u) and returns dx_next.
    We do finite differences (FD) in the backward pass using the info in fd_cache.
    """


    @jax.custom_vjp
    def step_fn(dx: mjx.Data, u: jnp.ndarray):
        """
        Forward pass:
          1) Writes 'u' into dx_init (or a copy thereof) via set_control_fn.
          2) Steps the simulation forward one step with MuJoCo.
        """
        dx_with_ctrl = set_control_fn(dx, u)
        dx_next = mjx.step(mx, dx_with_ctrl)
        dx_next = mjx.forward(mx, dx)
        return dx_next

    def step_fn_fwd(dx, u):
        dx_next = step_fn(dx, u)
        return dx_next, (dx, u, dx_next)

    def step_fn_bwd(res, g):
        """
        FD-based backward pass. We approximate d(dx_next)/d(dx,u) and chain-rule with g.
        Uses the cached flatten/unflatten info in fd_cache.
        """
        dx_in, u_in, dx_out = res

        # Convert float0 leaves in 'g' to zeros
        def map_g_to_dinput(diff_tree, grad_tree):
            def fix_leaf(d_leaf, g_leaf):
                if jax.dtypes.result_type(g_leaf) == jax.dtypes.float0:
                    return jnp.zeros_like(d_leaf)
                else:
                    return g_leaf

            return jax.tree_map(fix_leaf, diff_tree, grad_tree)

        mapped_g = map_g_to_dinput(dx_in, g)
        # jax.debug.print(f"mapped_g: {mapped_g}")
        g_array, _ = ravel_pytree(mapped_g)

        # Flatten dx_in, dx_out, and controls
        dx_array, _ = ravel_pytree(dx_in)
        dx_out_array, _ = ravel_pytree(dx_out)
        u_in_flat = u_in.ravel()

        # Grab cached info
        unravel_dx = fd_cache.unravel_dx
        sensitivity_mask = fd_cache.sensitivity_mask
        inner_idx = fd_cache.inner_idx
        num_u_dims = fd_cache.num_u_dims
        eps = fd_cache.eps

        # =====================================================
        # =============== FD wrt control (u) ==================
        # =====================================================
        def fdu_plus(i):
            e = jnp.zeros_like(u_in_flat).at[i].set(eps)
            u_in_eps = (u_in_flat + e).reshape(u_in.shape)
            dx_perturbed = step_fn(dx_in, u_in_eps)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            # Only keep relevant dims
            return sensitivity_mask * (dx_perturbed_array - dx_out_array) / eps

        # shape = (num_u_dims, dx_dim)
        Ju_array = jax.vmap(fdu_plus)(jnp.arange(num_u_dims))

        # =====================================================
        # ================ FD wrt state (dx) ==================
        # =====================================================
        # We only FD over "inner_idx" (subset of the state: qpos, qvel, ctrl, etc.)
        def fdx_for_index(idx):
            perturbation = jnp.zeros_like(dx_array).at[idx].set(eps)
            dx_in_perturbed = unravel_dx(dx_array + perturbation)
            dx_perturbed = step_fn(dx_in_perturbed, u_in)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            # Only keep relevant dims
            return sensitivity_mask * (dx_perturbed_array - dx_out_array) / eps

        # shape = (len(inner_idx), dx_dim)
        Jx_rows = jax.vmap(fdx_for_index)(inner_idx)

        # -----------------------------------------------------
        # Instead of scattering rows into a (dx_dim, dx_dim) matrix,
        # multiply Jx_rows directly with g_array[inner_idx].
        # This avoids building a large dense Jacobian in memory.
        # -----------------------------------------------------
        # Jx_rows[i, :] is derivative w.r.t. dx_array[inner_idx[i]].
        # We want sum_i [ Jx_rows[i] * g_array[inner_idx[i]] ].
        # => shape (dx_dim,)
        # Scatter those rows back to a full (dx_dim, dx_dim) matrix
        def scatter_rows(subset_rows, subset_indices, full_shape):
            base = jnp.zeros(full_shape, dtype=subset_rows.dtype)
            return base.at[subset_indices].set(subset_rows)

        dx_dim = dx_array.size

        # Solution 2 : Reduced size multiplication (inner_idx, inner_idx) @ (inner_idx,)
        d_x_flat_sub = Jx_rows[:, inner_idx] @ g_array[inner_idx]
        d_x_flat = scatter_rows(d_x_flat_sub, inner_idx, (dx_dim,))

        d_u = Ju_array[:, inner_idx] @ g_array[inner_idx]
        d_x = unravel_dx(d_x_flat)
        return (d_x, d_u)

    step_fn.defvjp(step_fn_fwd, step_fn_bwd)
    return step_fn


"""
Simulation Loop 
"""

# simulate states
def simulate(mjx_data, num_steps, step_function):
    state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
    states = [state]
    state_jacobians = []

    #jac_fn_rev = jax.jit(jax.jacrev(step_function))

    for _ in range(num_steps):
        #J_s = jac_fn_rev(state)
        #state_jacobians.append(J_s)
        state = step_function(state)
        states.append(jnp.array(state))

    states, state_jacobians = jnp.array(states), jnp.array(state_jacobians)
    return states, state_jacobians

def simulate_data(mjx_data, num_steps, step_function):
    state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
    states = [state]

    for _ in range(num_steps):
        mjx_data = step_function(mjx_data, u=jnp.zeros(mjx_data.ctrl.shape))
        state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
        states.append(state)

    return states, mjx_data

import jax
import jax.numpy as jnp

def simulate_data_lax(mjx_data, num_steps, step_fn):
    """Simulate forward 'num_steps' using a JAX-compatible step_fn with lax.scan."""
    # We'll store qpos+qvel as states. If you want to store the entire mjx_data
    # each step, we can do that, but it's more memory-intensive.

    def body_fn(carry, _):
        data = carry
        # Apply one simulation step with zero controls (replace with your controls if needed)
        data = step_fn(data, u=jnp.zeros_like(data.ctrl))
        # Extract (qpos, qvel) as a single vector
        state = jnp.concatenate([data.qpos, data.qvel])
        return data, state

    # initial state to store in the log
    init_state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])

    # run the scan over 'num_steps'
    final_data, state_log = jax.lax.scan(body_fn, mjx_data, xs=None, length=num_steps)

    # Prepend the initial state to the log so it has length = num_steps + 1
    state_log = jnp.concatenate([init_state[None, :], state_log], axis=0)
    return state_log, final_data

# simulate dx data structures
def simulate_(mjx_data, num_steps, step_function):
    # Initial state as a concatenated vector of qpos and qvel.
    state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
    states = [state]
    state_jacobians = []

    # Wrapper that converts state vector -> mjx.Data, calls step_function, and returns a state vector.
    def state_wrapper(state):
        # Assuming mjx_data has nq elements in qpos.
        nq = mjx_data.qpos.shape[0]
        # Create a new mjx.Data with the current state.c
        dx = mjx_data.replace(qpos=state[:nq], qvel=state[nq:])
        # Call the step function with no control (u=None).
        dx_next = step_function(dx, u=jnp.zeros(mjx_data.ctrl.shape))
        # Convert the resulting mjx.Data back into a state vector.
        return jnp.concatenate([dx_next.qpos, dx_next.qvel])

    # JIT compile the Jacobian function of the state wrapper.
    #jac_fn_rev = jax.jit(jax.jacrev(state_wrapper))

    for _ in range(num_steps):
        # Compute the Jacobian of the state transition.
        #J_s = jac_fn_rev(state)
        #state_jacobians.append(J_s)
        # Update the state using the wrapped step function.
        state = state_wrapper(state)
        states.append(state)

    # Convert lists to JAX arrays.
    states = jnp.array(states)
    state_jacobians = jnp.array(state_jacobians)
    return states, state_jacobians

"""
Visualisation
"""

def visualise_trajectory(states, d: mujoco.MjData, m: mujoco.MjModel, sleep=0.01):

    states_np = [np.array(s) for s in states]

    spinner_tip_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target_site")
    target_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "target_decoration")

    with viewer.launch_passive(m, d) as v:
        for s in states_np:

            spinner_tip = d.site_xpos[spinner_tip_site]
            target = d.geom_xpos[target_site]

            print(f"spinner tip: {spinner_tip}")
            #print(f"target*: {target}")
            distance = np.linalg.norm(spinner_tip - target)
            print(f"Distance between spinner tip and target: {distance:.4f}")
            step_start = time.time()
            # Extract qpos and qvel from the state.
            nq = m.nq
            # We assume the state was produced as jnp.concatenate([qpos, qvel])
            qpos = s[:nq]
            qvel = s[nq:nq + m.nv]
            #print(qvel)
            d.qpos[:] = qpos
            d.qvel[:] = qvel
            mujoco.mj_forward(m, d)
            v.sync()
            # Optionally sleep to mimic real time.

            #print(qpos)
            # print the position of the spinner only
            #print(qpos[3:])
            time.sleep(sleep)
            # Optionally, adjust to match the simulation timestep:
            #time_until_next = m.opt.timestep - (time.time() - step_start)
            #if time_until_next > 0:
            #    time.sleep(time_until_next)

