import jax
import jax.numpy as jnp
from jax._src.util import safe_zip, unzip2
from jax.flatten_util import ravel_pytree
import numpy as np
from typing import Callable
import equinox
from dataclasses import dataclass
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')

"""
We should decide which solver to use before importing mujoco
"""
import argparse, os
parser = argparse.ArgumentParser()
parser.add_argument("--gradient_mode", type=str, default="implicit_jaxopt", help="solver")
args = parser.parse_args()
# set the MuJoCo solver depending on the gradient mode
os.environ["MJX_SOLVER"] = args.gradient_mode
import mujoco
from mujoco import mjx

def upscale(x):
    """Convert data to 64-bit precision."""
    if hasattr(x, 'dtype'):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x

def prepare_sensitivity(dx_template, target_fields=('qpos','qvel','ctrl')):
    """
    Precompute flatten/unflatten, plus which elements of 'dx_template' are relevant
    for finite differences.
    """
    # Flatten the template. We'll reuse `unravel_dx` to unflatten any new arrays.
    dx_array, unravel_dx = ravel_pytree(dx_template)

    # Leaves-with-path: allows matching leaf names like 'qpos', 'qvel', etc.
    leaves_with_path = list(jax.tree_util.tree_leaves_with_path(dx_template))

    # We'll gather the size and shape for each leaf
    sizes, _ = unzip2(
        (jnp.size(leaf), jnp.shape(leaf)) for _, leaf in leaves_with_path
    )
    indices = tuple(np.cumsum(sizes))

    # For each leaf, check if any path component matches something in `target_fields`
    def leaf_index_range(leaf_idx):
        # For leaf i, the flattened slice is [start_i : end_i]
        start = 0 if leaf_idx == 0 else indices[leaf_idx - 1]
        end = indices[leaf_idx]
        return np.arange(start, end)

    idx_target_state = []
    for i, (path, leaf_val) in enumerate(leaves_with_path):
        # If any level in `path` has `level.name` in `target_fields`, we keep it
        if any(getattr(level, 'name', None) in target_fields for level in path):
            idx_target_state.append(i)

    # Collect the actual flattened indices we care about
    inner_idx_list = []
    for i in idx_target_state:
        inner_idx_list.append(leaf_index_range(i))
    inner_idx = np.concatenate(inner_idx_list, axis=0)
    inner_idx = jnp.array(inner_idx, dtype=jnp.int32)

    # Build a mask of shape = dx_array.shape
    sensitivity_mask = jnp.zeros_like(dx_array).at[inner_idx].set(1.0)

    return unravel_dx, inner_idx, sensitivity_mask


@equinox.filter_jit
def step_fn_core(mx, dx, u, set_control_fn):
    """
    A jittable core that sets control and then does one MuJoCo step.
    """
    dx_with_ctrl = set_control_fn(dx, u)
    return mjx.step(mx, dx_with_ctrl)

def make_step_fn_fd(
    mx,
    set_control_fn: Callable,
    unravel_dx,            # from prepare_sensitivity
    inner_idx,             # from prepare_sensitivity
    sensitivity_mask,      # from prepare_sensitivity
    eps: float = 1e-6
):
    """
    Create a custom_vjp step function that does FD w.r.t. dx and u, using
    precomputed flatten/unflatten metadata, plus a jitted core for stepping.
    """

    @jax.custom_vjp
    def step_fn(dx: jnp.ndarray, u: jnp.ndarray):
        """
        Single-argument step function:
        1) Writes 'u' into dx (via set_control_fn).
        2) Steps the simulation forward one step with MuJoCo.
        3) Returns dx_next (which is also a pytree).
        """
        jax.debug.print("Using FD-based VJP")
        return step_fn_core(mx, dx, u, set_control_fn)

    def step_fn_fwd(dx, u):
        dx_next = step_fn_core(mx, dx, u, set_control_fn)
        return dx_next, (dx, u, dx_next)

    def step_fn_bwd(res, g):
        """
        FD-based backward pass. Approximates ∂dx_next/∂u and ∂dx_next/∂dx
        via forward differences, then does chain rule with cotangent g.
        """
        dx_in, u_in, dx_out = res

        # Flatten the relevant things only once
        dx_array, _ = ravel_pytree(dx_in)
        dx_out_array, _ = ravel_pytree(dx_out)

        # Convert float0 leaves of g to zeros (standard JAX trick)
        def map_g_to_dinput(diff_tree, grad_tree):
            def fix_leaf(d_leaf, g_leaf):
                if jax.dtypes.result_type(g_leaf) == jax.dtypes.float0:
                    return jnp.zeros_like(d_leaf)
                else:
                    return g_leaf
            return jax.tree.map(fix_leaf, diff_tree, grad_tree)

        mapped_g = map_g_to_dinput(dx_in, g)
        g_array, _ = ravel_pytree(mapped_g)

        # Flatten the input control
        u_in_flat = u_in.ravel()
        num_u_dims = u_in_flat.shape[0]

        # ====== Derivative wrt. control U (finite differences) ======
        @jax.vmap
        def fdu_plus(i):
            """Compute [dx_next(u + e_i*eps) - dx_next(u)] / eps in flattened form."""
            e = jnp.zeros_like(u_in_flat).at[i].set(eps)
            u_in_eps = (u_in_flat + e).reshape(u_in.shape)
            dx_perturbed = step_fn_core(mx, dx_in, u_in_eps, set_control_fn)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            return sensitivity_mask * (dx_perturbed_array - dx_out_array) / eps

        Ju_array = fdu_plus(jnp.arange(num_u_dims))  # shape = (num_u_dims, dx_dim)

        # ====== Derivative wrt. state dx_in (finite differences) ======
        def fdx_for_index(idx):
            """Compute [dx_next(dx_in + e_idx) - dx_next(dx_in)] / eps, flattened."""
            perturbation = jnp.zeros_like(dx_array).at[idx].set(eps)
            dx_in_perturbed = unravel_dx(dx_array + perturbation)
            dx_perturbed = step_fn_core(mx, dx_in_perturbed, u_in, set_control_fn)
            dx_perturbed_array, _ = ravel_pytree(dx_perturbed)
            return sensitivity_mask * (dx_perturbed_array - dx_out_array) / eps

        # Only vmap over the relevant subset "inner_idx"
        Jx_rows = jax.vmap(fdx_for_index)(inner_idx)  # shape=(#relevant, dx_dim)

        # Scatter these rows back into a full (dx_dim, dx_dim)
        def scatter_rows(subset_rows, subset_indices, full_shape):
            base = jnp.zeros(full_shape, dtype=subset_rows.dtype)
            return base.at[subset_indices].set(subset_rows)

        Jx_array = scatter_rows(Jx_rows, inner_idx, (dx_array.size, dx_array.size))

        # ============== Combine with the cotangent ==================
        # dL/du = g_array^T @ Ju_array  => shape = (num_u_dims,)
        d_u_flat = Ju_array @ g_array
        # dL/dx = g_array^T @ Jx_array => shape = (dx_dim,)
        d_x_flat = Jx_array @ g_array

        d_x = unravel_dx(d_x_flat)
        d_u = d_u_flat.reshape(u_in.shape)

        return (d_x, d_u)

    step_fn.defvjp(step_fn_fwd, step_fn_bwd)
    return step_fn

"""
Default step function, no custom VJP backwards pass 
"""
def make_step_fn(mx, set_control_fn: Callable):

    # Define the step function
    def step_fn(dx: jnp.ndarray, u: jnp.ndarray):
        return step_fn_core(mx, dx, u, set_control_fn)

    return step_fn

def make_loss_fn(
        mx,
        qpos_init: jnp.ndarray,
        set_ctrl_fn: Callable[[mjx.Data, jnp.ndarray], mjx.Data],
        running_cost_fn: Callable[[mjx.Data], float],
        terminal_cost_fn: Callable[[mjx.Data], float],
        unravel_dx,
        inner_idx,
        sensitivity_mask,
        eps: float = 1e-6
):
    # Build the step function with custom_vjp + FD
    if args.gradient_mode == "fd":
        single_arg_step_fn = make_step_fn_fd(
            mx=mx,
            set_control_fn=set_ctrl_fn,
            unravel_dx=unravel_dx,
            inner_idx=inner_idx,
            sensitivity_mask=sensitivity_mask,
            eps=eps
        )
    else:
        single_arg_step_fn = make_step_fn(mx, set_ctrl_fn)

    @equinox.filter_jit
    def simulate_trajectory(U: jnp.ndarray):
        dx0 = mjx.make_data(mx)
        dx0 = jax.tree.map(upscale, dx0)
        dx0 = dx0.replace(qpos=dx0.qpos.at[:].set(qpos_init))
        dx0 = mjx.step(mx, dx0)  # initial sync

        def scan_body(dx, u):
            dx_next = single_arg_step_fn(dx, u)
            cost_t = running_cost_fn(dx_next)
            state_t = jnp.concatenate([dx_next.qpos, dx_next.qvel])
            return dx_next, (state_t, cost_t)

        dx_final, (states, costs) = jax.lax.scan(scan_body, dx0, U)
        total_cost = jnp.sum(costs) + terminal_cost_fn(dx_final)
        return states, total_cost

    def loss(U: jnp.ndarray):
        _, total_cost = simulate_trajectory(U)
        return total_cost

    return loss


@dataclass
class PMP:
    """
    A gradient-based optimizer for the FD-based MuJoCo trajectory problem.
    """
    loss: Callable[[jnp.ndarray], float]


    def solve(
            self,
            U0: jnp.ndarray,
            learning_rate: float = 1e-2,
            tol: float = 1e-6,
            max_iter: int = 100
    ):
        grad_loss = equinox.filter_jit(jax.jacrev(self.loss))
        loss_history = []
        U = U0
        for i in range(max_iter):
            g = grad_loss(U)
            U_new = U - learning_rate * g
            cost_val = self.loss(U_new)
            print(f"\n--- Iteration {i} ---")
            print(f"Cost={cost_val}")
            print(f"||grad||={jnp.linalg.norm(g)}")
            loss_history.append(cost_val)
            # Check for convergence
            if jnp.linalg.norm(U_new - U) < tol or jnp.isnan(g).any():
                print(f"Converged at iteration {i}.")
                return U_new
            U = U_new
        return U, loss_history


# Visualise the converged trajectory.
def plot_loss(loss_history):
    # Plot the loss graph.
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_history, marker='o')
    plt.title(f"Finger: Trajectory Optimisation Cost - Gradient Mode: {args.gradient_mode}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path("/Users/hashim/Desktop/Thesis/experiments/finger_optimisation/finger.xml")
    mx = mjx.put_model(model)
    dx = mjx.make_data(mx)
    dx = jax.tree.map(upscale, dx)
    qpos_init = jnp.array([-.8, 0, -.8])
    Nsteps, nu = 300, 2
    U0 = 0.2*jax.random.normal(jax.random.PRNGKey(0), (Nsteps, nu)) * 2

    def running_cost(dx):
        pos_finger = dx.qpos[2]
        u = dx.ctrl
        return 0.002 * jnp.sum(u ** 2) + 0.001 * pos_finger ** 2

    def terminal_cost(dx):
        pos_finger = dx.qpos[2]
        return 4 * pos_finger ** 2

    def set_control(dx, u):
        return dx.replace(ctrl=dx.ctrl.at[:].set(u))

    # 2) Precompute flatten/unflatten + sensitivity info
    unravel_dx, inner_idx, sensitivity_mask = prepare_sensitivity(
        dx,
        target_fields={'qpos','qvel','ctrl'}
    )

    # 3) Build the loss function with the new step fn
    loss_fn = make_loss_fn(
        mx=mx,
        qpos_init=qpos_init,
        set_ctrl_fn=set_control,
        running_cost_fn=running_cost,
        terminal_cost_fn=terminal_cost,
        unravel_dx=unravel_dx,
        inner_idx=inner_idx,
        sensitivity_mask=sensitivity_mask,
        eps=1e-6
    )

    # 4) Optimize
    #pmp = PMP(loss=loss_fn)

    # initialise a PMP object
    pmp = PMP(loss=loss_fn)
    optimal_U, loss_history = pmp.solve(U0=jnp.zeros((Nsteps, nu)), learning_rate=0.5, max_iter=30)

    # write the optimal control sequence to a numpy file
    np.save(f"U_{args.gradient_mode}.npy", optimal_U)

    # save the loss history to a numpy file
    np.save(f"loss_history_{args.gradient_mode}.npy", loss_history)

    # plot the loss history
    plot_loss(loss_history)