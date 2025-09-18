"""This module contains code for variational optimization."""

from functools import partial
import warnings
import numpy as np

import jax
from jax import numpy as jnp
import optax
from jax_tqdm import loop_tqdm
from .matrix import matrix_v2_partial
from .universal_ansatze import ansatz_specs
from scipy.stats import unitary_group, ortho_group, special_ortho_group


def make_cost_fn(matrix_fn):
    """Produce a cost function for compilation for a given matrix function."""

    def cost_fn(params, target_dag):
        r"""Cost function for compilation.
        It computes the matrix :math:`M` for given parameters and returns the infidelity

        .. math::

            I = 1 - \frac{1}{2^n} |\operatorname{tr}(V^\dagger U)|

        where :math:`n` is the qubit count and :math:`V` is the target unitary to be compiled.

        Args:
            params (np.ndarray): Parameters for the matrix function of an ansatz.
            target_dag (np.ndarray): Adjoint of the target unitary :math:`V` to be compiled.

        Returns:
            float: Evaluated cost function.

        """
        U = matrix_fn(params)
        return 1 - jnp.abs(jnp.trace(target_dag @ U)) / len(target_dag)

    return cost_fn


def make_optimization_run(cost_fn, optimizer):
    """Create a full optimization workflow executable, with tailored syntax for
    compilation tasks.

    Args:
        cost_fn (callable): Compilation cost function. Should have signature
            ``(jnp.ndarray, jnp.ndarray)->float`` where the first input contains parameters to be
            optimized and the second input is the target unitary.
        optimizer: Instance of an ``optax`` optimizer.
    """

    value_and_grad = jax.jit(jax.value_and_grad(cost_fn, argnums=0))
    compiled_cost_fn = jax.jit(cost_fn)

    @jax.jit
    def partial_step(opt_state, theta, last_val, target_dag):
        """Closure variables:
        value_and_grad
        optimizer
        compiled_cost_fn
        """
        val, grad_circuit = value_and_grad(theta, target_dag)
        updates, opt_state = optimizer.update(
            grad_circuit,
            opt_state,
            theta,
            target_dag=target_dag,
            value=val,
            grad=grad_circuit,
            value_fn=compiled_cost_fn,
        )
        theta = optax.apply_updates(theta, updates)
        return opt_state, theta, val

    @jax.jit
    def static_step(opt_state, theta, last_val, target_dag):
        return opt_state, theta, last_val

    def _or(a, b):
        """Compute logical or in a JIT compatible manner."""
        return jax.lax.cond(a, lambda b: True, (lambda b: b), b)

    def _and(a, b):
        """Compute logical and in a JIT compatible manner."""
        return jax.lax.cond(a, lambda b: b, lambda b: False, b)

    def optimization_step(i, values, tol, max_const, target_dag, record_mod=1):
        """Perform a step of an optimization process.

        Args:
            i (int): Iteration variable
            values (tuple[tuple, jnp.ndarray]): Optimizer state, variables, and array with cost
                history, in that order. The latter needs to be of constant size and
                ``optimization_step`` overwrites the entry at ``i``.
            tol (float): Optimization tolerance. If the cost fall below this value, the
                optimization is effectively interrupted.
            max_const (int): Maximal number of iterations for which a constant cost value is
                allowed. Afterwards, the optimization is effectively interrupted.
            target_dag (jnp.ndarray): Adjoint of the compilation target.
            record_mod (int): Interval at which to record the cost function values.

        Returns:
            tuple[tuple,jnp.ndarray]: Optimizer state, optimized variables, and array with cost
            history.

        The logic of this function in Python reads

        opt_state, theta, cost, last_val, rec_idx = values
        if (last_val < tol) or (i>max_const>0 and allclose(cost[i-max_const:i-1], last_val)):
            opt_state, theta, val = static_step(opt_state, theta, last_val, target_dag)
        else:
            opt_state, theta, val = partial_step(opt_state, theta, last_val, target_dag)

        if i % record_mod == 0:
            cost[rec_idx] = val
            rec_idx += 1
        return opt_state, theta, cost, val, rec_idx

        """
        opt_state, theta, cost, last_val, rec_idx = values

        opt_state, theta, val = jax.lax.cond(
            _or(
                last_val < tol,
                _and(
                    _and(max_const > 0, i > max_const),
                    jnp.allclose(
                        jax.lax.dynamic_slice(cost, [i - max_const, 1], (abs(max_const - 1), 1)),
                        last_val,
                    ),
                ),
            ),
            static_step,
            partial_step,
            opt_state,
            theta,
            last_val,
            target_dag,
        )
        cost, rec_idx = jax.lax.cond(
            i % record_mod == 0,
            lambda i, val, cost, rec_idx: (cost.at[rec_idx].set([i, val]), rec_idx + 1),
            lambda i, val, cost, rec_idx: (cost, rec_idx),
            i,
            val,
            cost,
            rec_idx,
        )
        return opt_state, theta, cost, val, rec_idx

    def run_optimization(
        init_params, target_dag, n_epochs, tol, max_const, progress_bar, num_records=None
    ):
        """Run optimization workflow based on ``cost_fn`` and ``optimizer``.

        Args:
            init_params (jnp.ndarray): Initial parameters to be optimized.
            target_dag (jnp.ndarray): Adjoint of target unitary to be compiled.
            n_epochs (int): Number of epochs to optimize for.
            tol (float): Tolerance below which the optimization is effectively interrupted.
            max_const (int): Number of stagnating cost **recordings** (not optimization steps)
                after which the optimization is effectively interrupted.
            progress_bar (bool): Whether to print a progress bar during the optimization run.
            num_records (int): Number of cost evaluations to record across ``n_epochs`` epochs.

        Returns:
            tuple[jnp.ndarray]: Optimized parameters and recording of ``num_records`` steps
            and the attained cost values at those steps.
        """
        if num_records is None:
            num_records = n_epochs
        init_params = init_params.copy()
        opt_state = optimizer.init(init_params)
        values = (opt_state, init_params, 4 * jnp.ones((num_records + 1, 2)), 4.0, 0)
        assert n_epochs % num_records == 0
        record_mod = n_epochs // num_records
        step_fn = partial(
            optimization_step,
            tol=tol,
            max_const=max_const,
            target_dag=target_dag,
            record_mod=record_mod,
        )
        if progress_bar:
            step_fn = loop_tqdm(n_epochs)(step_fn)
        _, theta, cost, last_val, rec_idx = jax.lax.fori_loop(0, n_epochs, step_fn, values)
        cost = cost.at[num_records].set([n_epochs, last_val])
        return theta, cost

    return run_optimization


def compile(dim, optimization_run, key=None, tol=1e-10, max_attempts=10):
    """Repeatedly execute an optimization run until a convergence tolerance is hit.

    Args:
        dim (int): Dimension of the parameters to be optimized.
        optimization_run (callable): Optimization workflow to be executed.
        key (Union[None, int, jaxlib.xla_extension.ArrayImpl]): Random key to produce initial
            parameters for the optimizations from. May be ``None`` (random key), an integer
            (produces a JAX random key from the int) or a JAX random key.
        tol (float): Cost function tolerance under which the optimization is considered
            successful
        max_attempts (int): Maximal number of executions of ``optimization_run``.

    Returns:
        tuple[list[jnp.ndarray], jnp.ndarray, bool]: Cost recordings for all executions of
            ``optimization_run``, optimized parameters after final execution, and a Boolean
            whether the optimization was successful.

    """
    shape = (dim,)

    successful = False
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(24125))
    elif isinstance(key, int):
        key = jax.random.PRNGKey(key)

    costs = []
    thetas = []
    while not successful and len(thetas) < max_attempts:
        key, use_key = jax.random.split(key)
        # 0.2 is a good scaling factor for SU on n<=5 qubits
        theta = jax.random.normal(use_key, shape) * 0.2
        # theta = jax.random.normal(use_key, shape)
        # theta = jax.random.uniform(use_key, shape) * (4 * np.pi) - 2 * np.pi
        theta, cost = optimization_run(theta)
        costs.append(cost)
        thetas.append(theta)
        if tol is not None and cost[-1, 1] <= tol:
            successful = True

    return costs, thetas, successful


def compile_adapt(
    target,
    group,
    optimizer,
    n_epochs,
    num_czs=None,
    max_attempts_per_num_cz=1,
    tol=1e-10,
    max_const=0,
    progress_bar=False,
    num_records=500,
    seed=None,
):
    N = len(target)
    n = int(np.round(np.log2(N)))

    dim, params_init, params_per_cz, univ_num_cz, _ = ansatz_specs(n, group)
    if num_czs is None:
        num_czs = list(range(1, univ_num_cz + 1))

    target_dag = target.conj().T
    for num_cz in num_czs:
        print(f"Trying with {num_cz=} / {max(num_czs)}")
        matrix_fn = matrix_v2_partial(n, group, num_cz=num_cz)
        cost_fn = make_cost_fn(matrix_fn)
        run_optimization = make_optimization_run(cost_fn, optimizer)

        opt_run = jax.jit(
            partial(
                run_optimization,
                n_epochs=n_epochs,
                tol=tol,
                max_const=max_const,
                progress_bar=progress_bar,
                num_records=num_records,
            )
        )
        num_params = min((params_init + params_per_cz * num_cz, dim))

        energies, thetas, successful = compile(
            num_params,
            partial(opt_run, target_dag=target_dag),
            key=seed,
            tol=tol,
            max_attempts=max_attempts_per_num_cz,
        )
        if successful:
            break
    else:
        warnings.warn("Compilation failed.")
    return energies, thetas, num_cz


def sample_from_group(n, n_samples, group, seed):
    """Randomly sample matrices from a given classical Lie group acting on qubits.

    Args:
        n (int): Number of qubits.
        n_samples (int): Number of matrices to sample.
        group (str): Group to sample from. Must be one of ``("SU", "SO", "O")``.
        seed (None, int, np.random.RandomState, np.random.Generator): Random seed for sampling.
            May be anything that can be passed to ``scipy.unitary_group.rvs`` as ``random_state``.

    Returns:
        np.ndarray: Batch of randomly sampled matrices from the requested group. The
        returned object has shape ``(n_samples, 2**n, 2**n)``.

    """
    N = 2**n
    if group == "SU":
        samples = unitary_group.rvs(N, size=n_samples, random_state=seed)
        if n_samples == 1:
            samples = samples[None]

    elif group == "O":
        samples = special_ortho_group.rvs(N, size=n_samples, random_state=seed)
        if n_samples == 1:
            samples = samples[None]

    elif group == "SO":
        samples = special_ortho_group.rvs(N, size=n_samples, random_state=seed)
        if n_samples == 1:
            samples = samples[None]

    else:
        raise NotImplementedError

    return samples
