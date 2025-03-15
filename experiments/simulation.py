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
import equinox

config.update('jax_default_matmul_precision', 'high')
config.update("jax_enable_x64", True)


"""
Helper Functions
"""
def upscale(x):
    if hasattr(x, "dtype"):
        # If x is a JAX array and has an integer type, cast it to float64.
        if jnp.issubdtype(x.dtype, jnp.integer):
            return x.astype(jnp.float64)
        # If x is float32, cast it to float64.
        elif x.dtype == jnp.float32:
            return x.astype(jnp.float64)
        else:
            return x
    elif isinstance(x, int):
        # Convert Python ints to a JAX array of type float64.
        return jnp.array(x, dtype=jnp.float64)

def set_control(dx, u):
    return dx.replace(ctrl=dx.ctrl.at[:].set(u))

"""
Standard step function
"""

def make_step_fn(mx, set_control_fn=set_control):

    def step_fn(dx: mjx.Data, u: jnp.ndarray):
        """
        Forward pass:
          1) Writes 'u' into dx_init (or a copy thereof) via set_control_fn.
          2) Steps the simulation forward one step with MuJoCo.
        """
        dx_with_ctrl = set_control_fn(dx, u)
        dx_next = mjx.step(mx, dx_with_ctrl)
        return dx_next

    return step_fn


import jax
import jax.numpy as jnp
from jax import custom_vjp



"""
Simulation Loop 
"""

# just simulate states using mjx_data and standard step_function
def simulate_data(mjx_data, num_steps, step_function):
    state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
    states = [state]

    for _ in range(num_steps):
        # TODO - No Control
        mjx_data = step_function(mjx_data, u=jnp.zeros(mjx_data.ctrl.shape))
        state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
        states.append(state)

    return states, mjx_data

@equinox.filter_jit
def simulate_trajectory(mx, qpos_init, qvel_init, step_fn, U):
    """
    Simulate a trajectory given a control sequence U.

    Args:
        mx: The MuJoCo model handle (static).
        qpos_init: initial positions (array).
        qvel_init: initial velocities (array).
        step_fn: function (dx, u) -> dx_next to step the simulation (e.g. with FD or without)
        U: (N, nu) array of controls.

    Returns:
        states: (N, nq+nv) array of states.
        state_jacobians: (N, nq+nv, nq+nv) array of state transition Jacobians.
        control_jacobians: (N, nq+nv, nu) array of control Jacobians.
    """
    def step_state(dx, u):
        dx_next = step_fn(dx, u)
        # Reapply upscale so that every field is float.
        dx_next = jax.tree_map(upscale, dx_next)
        state = jnp.concatenate([dx_next.qpos, dx_next.qvel]).astype(jnp.float64)
        return dx_next, state

    # Helper function that returns just the differentiable output.
    def f(dx, u):
        return step_state(dx, u)[1]

    # Compute the Jacobians of f with respect to dx and u.
    J_f = jax.jacrev(f, argnums=(0, 1), allow_int=True)

    def step_fn_jac(dx, u):
        dx, state = step_state(dx, u)
        jac_state, jac_control = J_f(dx, u)
        # Here, jac_state is a Data-like structure with the same fields as dx.
        # Extract the derivatives with respect to qpos and qvel and flatten them.
        flat_jac_state = jnp.concatenate([jac_state.qpos, jac_state.qvel], axis=-1)
        return dx, (state, flat_jac_state, jac_control)

    # Build the initial data structure and upscale all its fields.
    dx0 = jax.tree_map(upscale, mjx.make_data(mx))
    dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))
    dx0 = dx0.replace(qvel=dx0.qvel.at[:].set(qvel_init))

    dx_final, (states, state_jacobians, control_jacobians) = jax.lax.scan(step_fn_jac, dx0, U)
    return states, state_jacobians, control_jacobians, dx_final


"""
Visualisation
"""

def visualise_trajectory(states, d: mujoco.MjData, m, sleep=0.01):
    print("visualise_trajectory: mj_model =", m)
    states_np = [np.array(s) for s in states]

    with viewer.launch_passive(m, d) as v:
        for s in states_np:
            step_start = time.time()
            # Extract qpos and qvel from the state.
            nq = m.nq
            # We assume the state was produced as jnp.concatenate([qpos, qvel])
            qpos = s[:nq]
            qvel = s[nq:nq + m.nv]
            d.qpos[:] = qpos
            d.qvel[:] = qvel
            mujoco.mj_forward(m, d)
            v.sync()
            # Optionally sleep to mimic real time.
            time.sleep(sleep)
            # Optionally, adjust to match the simulation timestep:
            time_until_next = m.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)


"""
Daniel's FD-based step function
"""

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
        fd_cache: FDCache,
        set_control_fn: Callable = set_control,
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
        #jax.debug.print("Using finite differences in the backward pass.")
        dx_with_ctrl = set_control_fn(dx, u)
        dx_next = mjx.step(mx, dx_with_ctrl)
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