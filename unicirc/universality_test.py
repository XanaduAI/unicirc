import re
from functools import partial
from itertools import product
import numpy as np
import jax
from jax import numpy as jnp
from typing import Iterable
from tqdm.auto import tqdm
from .universal_ansatze import ansatz_specs
from .matrix import matrix_v1, matrix_v2, matrix_v3


def jac_rank(mat_jac_fn, params, tol):
    """Compute the rank of the Jacobian computed via ``mat_jac_fn`` at ``params``.

    Args:
        mat_jac_fn (callable): Function computing the Jacobian of a square matrix, taking
            parameters as a single argument.
        params (jnp.ndarray): Parameters at which to check the Jacobian.
        tol (float): Tolerance for eigenvalues to be considered non-zero.

    Returns:
        int, np.ndarray: rank of the Jacobian and Jacobian reshaped as ``(dim**2, len(params))``

    """
    jac = mat_jac_fn(params)
    shape = jac.shape
    flat_jac = jac.reshape((jac.shape[0] ** 2, jac.shape[2]))
    rank = jnp.linalg.matrix_rank(flat_jac, tol=tol)
    return rank, flat_jac


def assert_rank(mat_jac_fn, params, expected_rank, tol):
    """Assert that the Jacobian computed by ``mat_jac_fn`` at ``params``
    has the rank ``expected_rank``.

    Args:
        mat_jac_fn (callable): Function computing the Jacobian of a square matrix, taking
            parameters as a single argument.
        params (jnp.ndarray): Parameters at which to check the Jacobian.
        expected_rank (int): Expected rank of the Jacobian.
        tol (float): Tolerance for eigenvalues to be considered non-zero.

    """
    rank, flat_jac = jac_rank(mat_jac_fn, params, tol=tol)
    if rank != expected_rank:
        gram = flat_jac @ flat_jac.T.conj()
        eigvals = jnp.linalg.eigvalsh(gram)
        rank = int(rank)
        raise AssertionError(f"{rank=}, {expected_rank=}\n{eigvals}")


def rank_test(n, group, num_samples=1, key=None, tol=None, use_v2=True):
    """Test whether the Jacobian of the (conjectured to be) universal ansatz for the given
    ``group`` acting on ``n`` qubits has full rank at a number of parameter positions.

    Args:
        n (int): Number of qubits
        group (str): Group for which the ansatz is tested
        num_samples (int): Number of parameter positions at which to test the ansatz
        key (None or int or jaxlib.xla_extension.ArrayImpl): Random seed key for parameter sampling

    Returns:
        callable: JITted function to compute the Jacobian of the ansatz matrix. Holomorphic
        Jacobian for ``group=="SU"`` or ``group="Sp"``.

    """
    if use_v2:
        mat_fn = matrix_v2(n, group)
    else:
        ansatz = make_ansatz(n, group)
        mat_fn = matrix_v1(ansatz, interface="jax")
    holomorphic = group != "SO"
    dtype = jnp.complex128 if holomorphic else jnp.float64
    mat_jac_fn = jax.jit(jax.jacobian(mat_fn, holomorphic=holomorphic))
    dim, *_ = ansatz_specs(n, group)
    if key is None:
        key = jax.random.PRNGKey(np.random.randint(2516))
    elif isinstance(key, int):
        key = jax.random.PRNGKey(key)

    params = jax.random.uniform(key, (num_samples, dim)).astype(dtype) * (4 * np.pi) - 2 * np.pi
    [assert_rank(mat_jac_fn, p, dim, tol=tol) for p in tqdm(params)]
    return mat_jac_fn


def all_candidates_linear(num_wires: int, num_cz: int) -> tuple[Iterable, int]:
    """Produce all sequences of pairs ``(i, i+1)`` with length ``num_cz`` where
    ``i<num_wires-1``. This corresponds to all unique circuit ansatze on ``num_wires`` qubits
    under linear connectivity.
    """
    all_connections = [(i, i + 1) for i in range(num_wires - 1)]
    return product(all_connections, repeat=num_cz), (num_wires - 1) ** num_cz


def filtered_candidates_linear(num_wires: int, num_cz: int) -> tuple[Iterable, int]:
    assert num_wires < 11
    r = num_wires - 1
    all_connections = [(i, i + 1) for i in range(num_wires - 1)]

    def int_to_base_r(i, r):
        """Convert an integer ``i`` to a string in base ``r``
        with length ``num_cz``."""
        return np.base_repr(i, base=r, padding=num_cz - int(np.ceil(np.log(max(1, i)) / np.log(r))))

    filtered = [
        tuple(all_connections[int(j)] for j in rep)
        for i in range(r**num_cz)
        if not re.match(r"\d*(\d)\1{3}", rep := int_to_base_r(i, r))
    ]
    return filtered, len(filtered)


@jax.jit
def _four_equals_test(connections):
    return jnp.allclose(connections[0], connections[1:])


def search_ansatze(
    num_wires,
    num_cz,
    reduction_test=False,
    all_solutions=False,
    candidates_fn=all_candidates_linear,
    key=None,
    rank_tol=1e-8,
    mat_jac_fn=None,
):
    """Search through all possible circuit ansatze to find a universal ansatz for SU(2^n),
     as determined by the Jacobian rank test.

    Args:
        num_wires (int): Number of qubits
        num_cz (int): Number of two-qubit gates (``CZ``) in the ansatze.
        reduction_test (bool): Whether or not to pre-screen the candidate ansatze for groups of
            at least four equal two-qubit gate targets. If ``num_cz`` is set to a lower bound,
            this test can be run to quickly exclude ansatze that can't be universal due to local
            redundancy.
        all_solutions (bool): Whether or not to return all solutions, i.e., all universal ansatze,
            as opposed to just returning the first one that is found.
        q (int): Simulation parameter reducing ``for`` loop unrolling.
        candidates_fn (callable): A function that creates sequences of two-qubit gate target pairs,
            with each sequence representing a unique ansatz for the given number of qubits and
            two-qubit gates. Should have signature
            ``(num_cz: int, num_wires: int) -> Sequence[Sequence[tuple[int]]], int`` where
            the first is the sequence of qubit pair sequences and the latter is the number of
            candidates produced overall.
            By default, ``all_candidates_linear`` is used, which produces all possible sequences under
            linear connectivity.
        key (None or int or jaxlib.xla_extension.ArrayImpl): Random seed key to generate parameters
            at which the rank test is performed.
        rank_tol (float): Tolerance for eigenvalues of the Jacobian to be considered non-zero in
            the Jacobian rank test for universality.

    Returns:
        Sequence[tuple[int]] or Sequence[Sequence[tuple[int]]]: Either
        a single sequence or all sequences of qubit pairs that encode universal circuits as
        determined by the Jacobian rank test.

    Note that this function only is intended to work for :math:`SU(2^n)`, not for other groups.

    Note that currently only three or four qubits are supported.
    """
    assert num_wires in [
        3,
        4,
    ], f"Currently only three or four qubits are supported. Got {num_wires=}"

    num_params = 3 * num_wires + 4 * num_cz
    dim = 4**num_wires - 1

    if all_solutions:
        solutions = []

    if key is None:
        key = jax.random.PRNGKey(np.random.randint(2516))
    elif isinstance(key, int):
        key = jax.random.PRNGKey(key)
    params = jax.random.uniform(key, num_params).astype(jnp.complex128) * (4 * np.pi) - 2 * np.pi

    if mat_jac_fn is None:
        mat_fn = matrix_v3(num_wires, group="SU", num_cz=num_cz)
        mat_jac_fn = jax.jit(jax.jacobian(mat_fn, holomorphic=True))

    candidates, num_candidates = candidates_fn(num_wires, num_cz)
    discarded_four_equals = 0
    discarded_rank_test = 0
    for connections in tqdm(candidates, total=num_candidates):
        _connections = jnp.array(connections)
        if reduction_test:
            if any(_four_equals_test(_connections[i : i + 4]) for i in range(num_cz - 3)):
                discarded_four_equals += 1
                continue

        rank, _ = jac_rank(partial(mat_jac_fn, connections=_connections), params, tol=rank_tol)
        if rank == dim:
            if not all_solutions:
                print(
                    f"Discarded {discarded_four_equals}/{num_candidates} candidates based on the "
                    f"reduction test and discarded {discarded_rank_test}/{num_candidates} "
                    "candidates based on the rank test before finding a universal ansatz"
                )
                return connections
            solutions.append(connections)
        else:
            discarded_rank_test += 1

    if all_solutions:
        print(
            f"Discarded {discarded_four_equals}/{num_candidates} candidates based on the "
            f"reduction test, discarded {discarded_rank_test}/{num_candidates} "
            f"candidates based on the rank test, and found {len(solutions)} universal ansatze."
        )
        return solutions
    print(
        f"Discarded {discarded_four_equals}/{num_candidates} candidates based on the reduction "
        f"test and {discarded_rank_test}/{num_candidates} candidates based on the rank test. Did "
        f"not find any universal circuit ansatze with {num_cz=} for {num_wires} qubits."
    )
    return None
