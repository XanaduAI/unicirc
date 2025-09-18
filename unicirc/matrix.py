"""This file contains a function to compute the unitary matrix corresponding to(num_wires, "SU",
quantum circuits made up of a few specific gate types, with the aim to speed up computational
efficiency over ``qml.matrix``."""

from string import ascii_letters as alphabet
from functools import partial
import numpy as np
from pennylane import transform, RZ, RY, CZ, CY, S
import jax
from jax import numpy as jnp
from .universal_ansatze import make_ansatz, ansatz_specs
from tqdm.auto import tqdm
import pennylane as qml


def matrix(n, group, connections=None):
    """Convenience function that reroutes to the fastest matrix function currently available."""

    if n == 3 and group == "SU":
        connections = [(0, 1), (1, 2)] * 7

        @jax.jit
        def fn(params, connections):
            increm = q * 4
            mat = jnp.eye(8, dtype=complex)
            mat = _rot_3(params[:9:3], 0, mat)
            mat = _rot_3(params[1:9:3], 1, mat)
            mat = _rot_3(params[2:9:3], 2, mat)
            i = 9
            for j in range(A):
                mat = _block_q_3_su(
                    params[i : (i := i + increm)], connections[q * j : q * (j + 1)], mat
                )
                # i += increm
            for connec in connections[q * A : -1]:
                mat = _block_3_su(params[i : (i := i + 4)], connec, mat)
                # i += 4
            mat = _cz_3((1, 2), mat)
            mat = _ry_3(params[i], 1, mat)
            mat = _ry_3(params[i + 1], 2, mat)
            return mat

        return fn

    elif connections is not None:
        return matrix_v3(n, group, connections, q=5, num_cz=len(connections))

    return matrix_v2(n, group)


@partial(transform, is_informative=True)
def matrix_v1(tape, interface="numpy"):
    """Compute the matrix of a PennyLane tape. This function is faster than
    ``qml.matrix`` due to customized kernels and tensorization.
    However, it is not as fast as ``matrix_v2``.

    Args:
        tape (qml.QuantumScript): The tape of operations, ignores measurements.
            Assumes only `qml.RY`, `qml.RZ` and `qml.CZ` to be present.
        interface="numpy" (str): math interface with which this matrix will be used.
            Make sure to pass `interface="jax"` for JIT compatibility and better performance.
    """
    num_wires = len(tape.wires)
    assert list(tape.wires) == list(range(num_wires))
    result_indices = alphabet[: 2 * num_wires]
    dim = 2**num_wires
    eye = np.eye(dim).reshape((2,) * (2 * num_wires))

    if interface == "numpy":
        cz_diag = np.array([[1, 1], [1, -1]])
        s_diag = np.array([1, 1j])
        cy_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]).reshape(
            (2, 2, 2, 2)
        )

        def processing_fn(res):
            result = eye

            for op in res[0].operations:
                if isinstance(op, RZ):
                    diag = np.exp(0.5j * op.data[0] * np.array([-1, 1]))
                    indices = f"{alphabet[op.wires[0]]},{result_indices}->{result_indices}"
                    result = np.einsum(indices, diag, result)
                elif isinstance(op, RY):
                    angle = op.data[0] / 2
                    op_mat = np.array(
                        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
                    )
                    op_letter = alphabet[op.wires[0]]
                    result_indices_out = result_indices.replace(op_letter, alphabet[-1])
                    indices = f"{alphabet[-1]}{op_letter},{result_indices}->{result_indices_out}"
                    result = np.einsum(indices, op_mat, result)
                elif isinstance(op, CZ):
                    indices = f"{alphabet[op.wires[0]]}{alphabet[op.wires[1]]},{result_indices}->{result_indices}"
                    result = np.einsum(indices, cz_diag, result)
                elif isinstance(op, CY):
                    op_letter0 = alphabet[op.wires[0]]
                    op_letter1 = alphabet[op.wires[1]]
                    result_indices_out = result_indices.replace(op_letter0, alphabet[-1])
                    result_indices_out = result_indices_out.replace(op_letter1, alphabet[-2])
                    indices = f"{alphabet[-1]}{alphabet[-2]}{op_letter0}{op_letter1},{result_indices}->{result_indices_out}"
                    result = np.einsum(indices, cy_mat, result)
                elif isinstance(op, S):
                    indices = f"{alphabet[op.wires[0]]},{result_indices}->{result_indices}"
                    result = np.einsum(indices, s_diag, result)
                else:
                    raise NotImplementedError(f"Gate {op} not supported.")
            return result.reshape((dim, dim))

    elif interface == "jax":

        def processing_fn(res):
            result = eye

            for op in res[0].operations:
                if isinstance(op, RZ):
                    result = _apply_RZ(op.data[0], op.wires[0], result, num_wires)
                elif isinstance(op, RY):
                    result = _apply_RY(op.data[0], op.wires[0], result, num_wires)
                elif isinstance(op, CZ):
                    result = _apply_CZ(op.wires, result, num_wires)
                elif isinstance(op, CY):
                    result = _apply_CY(op.wires, result, num_wires)
                elif isinstance(op, S):
                    result = _apply_S(op.wires[0], result, num_wires)
                else:
                    raise NotImplementedError(f"Gate {op} not supported.")
            return result.reshape((dim, dim))

    else:
        raise NotImplementedError(f"Interface {interface} not supported.")

    return [tape], processing_fn


@partial(jax.jit, static_argnames=("wire", "num_wires"))
def _apply_RZ(theta, wire, result, num_wires):
    """This is a private method for ``matrix_v1``."""
    result_indices = alphabet[: 2 * num_wires]
    diag = jnp.exp(0.5j * theta * jnp.array([-1, 1]))
    indices = f"{alphabet[wire]},{result_indices}->{result_indices}"
    return jnp.einsum(indices, diag, result)


@partial(jax.jit, static_argnames=("wire", "num_wires"))
def _apply_S(wire, result, num_wires):
    """This is a private method for ``matrix_v1``."""
    result_indices = alphabet[: 2 * num_wires]
    diag = jnp.array([1, 1j])
    indices = f"{alphabet[wire]},{result_indices}->{result_indices}"
    return jnp.einsum(indices, diag, result)


@partial(jax.jit, static_argnames=("wire", "num_wires"))
def _apply_RY(theta, wire, result, num_wires):
    """This is a private method for ``matrix_v1``."""
    result_indices = alphabet[: 2 * num_wires]
    s, c = jnp.sin(theta / 2), jnp.cos(theta / 2)
    op_mat = jnp.array([[c, -s], [s, c]])
    op_letter = alphabet[wire]
    result_indices_out = result_indices.replace(op_letter, alphabet[-1])
    indices = f"{alphabet[-1]}{op_letter},{result_indices}->{result_indices_out}"
    return jnp.einsum(indices, op_mat, result)


cz_diag_matrix_v1 = jnp.array([[1, 1], [1, -1]])


@partial(jax.jit, static_argnames=("wires", "num_wires"))
def _apply_CZ(wires, result, num_wires):
    """This is a private method for ``matrix_v1``."""
    result_indices = alphabet[: 2 * num_wires]
    indices = f"{alphabet[wires[0]]}{alphabet[wires[1]]},{result_indices}->{result_indices}"
    return jnp.einsum(indices, cz_diag_matrix_v1, result)


cy_matrix_v1 = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]).reshape(
    (2, 2, 2, 2)
)


@partial(jax.jit, static_argnames=("wires", "num_wires"))
def _apply_CY(wires, result, num_wires):
    """This is a private method for ``matrix_v1``."""
    result_indices = alphabet[: 2 * num_wires]
    # indices = f"{alphabet[wires[0]]}{alphabet[wires[1]]},{result_indices}->{result_indices}"
    op_letter0 = alphabet[wires[0]]
    op_letter1 = alphabet[wires[1]]
    result_indices_out = result_indices.replace(op_letter0, alphabet[-1])
    result_indices_out = result_indices_out.replace(op_letter1, alphabet[-2])
    indices = f"{alphabet[-1]}{alphabet[-2]}{op_letter0}{op_letter1},{result_indices}->{result_indices_out}"
    return jnp.einsum(indices, cy_matrix_v1, result)


@jax.jit
def make_su_block(params):
    """Create the 4x4 matrix corresponding to a block of the universal SU ansatz, reshaped
    into ``(2, 2, 2, 2)``.

    Args:
        params (jnp.ndarray): Four parameters for the block

    Returns:
        jnp.ndarray: Matrix of the block, reshaped as 4-dimensional tensor with bonds of size 2.
    """
    c0, s0 = jnp.cos(params[0] / 2), jnp.sin(params[0] / 2)
    c1, s1 = jnp.cos(params[1] / 2), jnp.sin(params[1] / 2)
    mat = jnp.kron(jnp.array([[c0, -s0], [s0, c0]]), jnp.array([[c1, -s1], [s1, c1]]))
    mat = mat.at[:, -1].multiply(-1)
    mat = (
        jnp.exp(
            jnp.array(
                [
                    -0.5j * (params[2] + params[3]),
                    -0.5j * (params[2] - params[3]),
                    0.5j * (params[2] - params[3]),
                    0.5j * (params[2] + params[3]),
                ]
            )
        )[:, None]
        * mat
    )
    return mat.reshape((2, 2, 2, 2))


@jax.jit
def make_so_block(params):
    """Create the 4x4 matrix corresponding to a block of the universal SO ansatz, reshaped
    into ``(2, 2, 2, 2)``.

    Args:
        params (jnp.ndarray): Two parameters for the block

    Returns:
        jnp.ndarray: Matrix of the block, reshaped as 4-dimensional tensor with bonds of size 2.
    """
    c0, s0 = jnp.cos(params[0] / 2), jnp.sin(params[0] / 2)
    c1, s1 = jnp.cos(params[1] / 2), jnp.sin(params[1] / 2)
    mat = jnp.kron(jnp.array([[c0, -s0], [s0, c0]]), jnp.array([[c1, -s1], [s1, c1]]))
    mat = mat.at[:, -1].multiply(-1)
    return mat.reshape((2, 2, 2, 2))
    """ # Hardcoded version of this block's matrix
    return jnp.array([
        [c0 * c1, -c0 * s1, -s0 * c1, -s0 * s1],
        [c0 * s1, c0 * c1, -s0 * s1, s0 * c1],
        [s0 * c1, -s0 * s1, c0 * c1, c0 * s1],
        [s0 * s1, s0 * c1, c0 * s1, -c0 * c1],
    ]).reshape((2, 2, 2, 2))
    """


custom_cY = jnp.array(
    [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0]], dtype=jnp.complex128
)


@jax.jit
def make_sp_block(params):
    """Create the 4x4 matrix corresponding to a block of the universal Sp ansatz, reshaped
    into ``(2, 2, 2, 2)``.

    Args:
        params (jnp.ndarray): Four parameters for the block

    Returns:
        jnp.ndarray: Matrix of the block, reshaped as 4-dimensional tensor with bonds of size 2.
    """
    phase = 0.5j * params[0]
    mat = jnp.exp(jnp.array([-phase, -phase, phase, phase]))[:, None] * custom_cY
    c0, s0 = jnp.cos(params[1] / 2), jnp.sin(params[1] / 2)
    c1, s1 = jnp.cos(params[2] / 2), jnp.sin(params[2] / 2)
    mat = (
        jnp.kron(jnp.array([[c0, -s0], [s0, c0]]), jnp.array([[c1, -s1], [s1, c1]])).astype(
            jnp.complex128
        )
        @ mat
    )
    return mat.reshape((2, 2, 2, 2))


@partial(jax.jit, static_argnames=("k", "n", "fixed_trailing", "fixed_leading", "backwards"))
def roll_axes(array, k, n, fixed_trailing, fixed_leading=0, backwards=False):
    """Roll the first axes of a tensor to the back, possibly except for some fixed trailing axes.
    Offers the option to instead roll last axes to the front (``backwards`` mode, again except
    for some trailing axes).

    Args:
        array (jnp.ndarray): Tensor to roll the axes of.
        k (int): Number of axes to roll.
        n (int): Total number of axes that are affected by the rolling. That is, potentially
            fixed trailing/leading axes are *not* included in ``n``.
        fixed_trailing (int): Number of trailing axes to keep in place
        fixed_leading (int): Number of trailing axes to keep in place
        backwards (bool): Whether to roll backwards. By default ``backwards=False``, corresponding
            to the first ``k`` axes ending up in the last ``k`` slots, up to fixed
            trailing/leading axes.

    Returns:
        jnp.ndarray: Tensor with rolled axes.
    """
    if backwards:
        k = n - k
    n = n + fixed_leading
    k = k + fixed_leading
    new_indices = (
        list(range(fixed_leading))
        + list(range(k, n))
        + list(range(fixed_leading, k))
        + list(range(n, n + fixed_trailing))
    )
    return array.transpose(new_indices)


def _body0(i, diag_and_params, n):
    """Private method for ``first_layer``, which is used in ``matrix_v2``.
    Effectively applies an ``RZ`` gate to a diagonal matrix, and rolls
    axes so that the next ``RZ`` can be applied in the same way.

    Args:
        i (int): Qubit index to which an ``RZ`` gate is applied.
        diag_and_params (tuple[jnp.ndarray]): Diagonal of the diagonal matrix to which the ``RZ``
            gate is applied, and parameters for all ``RZ`` rotations applied in the layer.
        n (int): Number of qubits.

    Returns:
        tuple[jnp.ndarray]: New diagonal of diagonal matrix to which the ``RZ`` gate was applied,
        and unchanged array of parameters for all ``RZ`` rotations.
    """
    diag, params = diag_and_params
    phase = 0.5j * params[n - 1 - i]
    factor = jnp.exp(jnp.array([-phase, phase]))
    diag = roll_axes(factor * diag, k=1, n=n, fixed_trailing=0, backwards=True)
    return diag, params


@partial(jax.jit, static_argnames=("n",))
def first_layer(params, n):
    """Create the matrix for the first layer of the universal SU ansatz from scratch.
    (This function allocates the matrix in the first place.)

    Args:
        params (jnp.ndarray): Parameters for all ``RZ`` rotations in the first layer. The length
            should match ``n``.
        n (int): Number of qubits (and parameters).

    Returns:
        jnp.ndarray: matrix created by the first layer of the universal SU ansatz.
        It has shape ``(2,) * n + (2**n,)`` for ``n`` qubits. That is, the row index is expanded,
        or tensorized, into ``n`` axes of size ``2`` whereas the column index is left in
        "standard matrix" form, with a single axis of size ``2**n``.

    """
    diag = np.ones(2**n, dtype=jnp.complex128).reshape((2,) * n)
    diag_and_params = (diag, params)
    diag, params = jax.lax.fori_loop(0, n, partial(_body0, n=n), diag_and_params)
    return jnp.diag(diag.reshape(2**n)).reshape((2,) * n + (2**n,))


@partial(jax.jit, static_argnames=("n",))
def first_layer_sp(param, n):
    diag = np.ones(2**n, dtype=jnp.complex128).reshape((2,) * n)
    phase = 0.5j * param
    factor = jnp.exp(jnp.array([-phase, phase]))
    diag = roll_axes(diag, k=1, n=n, fixed_trailing=0, backwards=False)
    diag = roll_axes(factor * diag, k=1, n=n, fixed_trailing=0, backwards=True)
    return jnp.diag(diag.reshape(2**n)).reshape((2,) * n + (2**n,))


def _body1(i, mat_and_params, n, indices):
    """Private method for ``second_layer``, which is used in ``matrix_v2``.
    Effectively applies an ``RY`` gate to a matrix in special shape format, and rolls
    axes so that the next ``RY`` can be applied using the same contraction.

    Args:
        i (int): Qubit index to which an ``RY`` gate is applied.
        mat_and_params (tuple[jnp.ndarray]): Matrix to which the ``RY`` gate is applied (in shape
            ``(2,) * n + (2**n,)``, and parameters for all ``RY`` rotations applied in the layer.
        n (int): Number of qubits.
        indices (str): Contraction rule for the einsum call that implements the gate application.

    Returns:
        tuple[jnp.ndarray]: New matrix to which the ``RY`` gate was applied,
        and unchanged array of parameters for all ``RY`` rotations.
    """
    mat, params = mat_and_params
    c, s = jnp.cos(params[i] / 2), jnp.sin(params[i] / 2)
    mat = jnp.einsum(indices, jnp.array([[c, -s], [s, c]], dtype=mat.dtype), mat)
    mat = roll_axes(mat, k=1, n=n, fixed_trailing=1)
    return mat, params


@partial(jax.jit, static_argnames=("n",))
def apply_second_layer(params, mat, n):
    """Apply a layer of ``RY`` rotations to a matrix.

    Args:
        params (jnp.ndarray): Parameters for all ``RY`` rotations in the layer. The length
            should match ``n``.
        mat (jnp.ndarray): Matrix to apply the rotations to, with shape ``(2,) * n + (2**n,)``.
        n (int): Number of qubits (and parameters).

    Returns:
        jnp.ndarray: matrix with layer of ``RY`` rotations applied to it. The output shape
        matches the input shape.

    """
    mat_and_params = (mat, params)
    indices = f"Aa,{alphabet[:n+1]}->A{alphabet[1:n+1]}"
    mat, params = jax.lax.fori_loop(0, n, partial(_body1, n=n, indices=indices), mat_and_params)
    return mat


def _body2(i, mat_and_params, n):
    """Private method for ``third_layer``, which is used in ``matrix_v2``.
    Effectively applies an ``RZ`` gate to a matrix in special shape format, and rolls
    axes so that the next block can be applied using the same multiplication rule.

    Args:
        i (int): Qubit index to which an ``RZ`` gate is applied.
        mat_and_params (tuple[jnp.ndarray]): Matrix to which the ``RZ`` gate is applied (in shape
            ``(2,) * n + (2**n,)``, and parameters for all ``RZ`` rotations applied in the layer.
        n (int): Number of qubits.

    Returns:
        tuple[jnp.ndarray]: New matrix to which the ``RZ`` gate was applied,
        and unchanged array of parameters for all ``RZ`` rotations.
    """
    mat, params = mat_and_params
    phase = 0.5j * params[n - 1 - i]
    factor = jnp.exp(jnp.array([-phase, phase]))[:, jnp.newaxis]
    mat = roll_axes(factor * mat, k=1, n=n, fixed_trailing=1, backwards=True)
    return mat, params


@partial(jax.jit, static_argnames=("n",))
def apply_third_layer(params, mat, n):
    """Apply a layer of ``RZ`` rotations to a matrix.

    Args:
        params (jnp.ndarray): Parameters for all ``RZ`` rotations in the layer. The length
            should match ``n``.
        mat (jnp.ndarray): Matrix to apply the rotations to, with shape ``(2,) * n + (2**n,)``.
        n (int): Number of qubits (and parameters).

    Returns:
        jnp.ndarray: matrix with layer of ``RZ`` rotations applied to it. The output shape
        matches the input shape.

    """
    mat_and_params = (mat, params)
    mat, params = jax.lax.fori_loop(0, n, partial(_body2, n=n), mat_and_params)
    return mat


@partial(jax.jit, static_argnames=("n",))
def apply_third_layer_sp(param, mat, n):
    """ """
    phase = 0.5j * param
    factor = jnp.exp(jnp.array([-phase, phase]))[:, jnp.newaxis]
    mat = roll_axes(mat, k=1, n=n, fixed_trailing=1, backwards=False)
    mat = roll_axes(factor * mat, k=1, n=n, fixed_trailing=1, backwards=True)
    return mat


def _body3(i, mat_and_params_and_K, n, indices, make_block_matrix):
    """Private method for ``apply_blocks``, which is used in ``matrix_v2``.
    Effectively applies a two-qubit block of gates to a matrix in special shape format, and rolls
    axes so that the next block can be applied.

    Args:
        i (int): Block index determining which parameters are used in the block of gates.
        mat_and_params_and_K (tuple[jnp.ndarray, int]): Matrix to which the block of gates is
            applied (in shape ``(2,) * n + (2**n,)``, parameters for all blocks applied in
            the circuit, and the current axis roll position ``K``.
        n (int): Number of qubits.
        indices (str): Contraction rule for the einsum call that implements the gate application.

    Returns:
        tuple[jnp.ndarray, int]: New matrix to which the block of gates was applied,
        unchanged array of parameters for the full circuit, and updated axis roll position ``K``.

    The shape of ``mat`` is ``(2,) * n + (2**n,)`` throughout, and the shape of ``params``
    is ``(L, 4)`` throughout, where ``L`` is the number of blocks to be applied in total.
    """
    mat, params, K = mat_and_params_and_K
    # Create matrix for block with current parameters
    block_matrix = make_block_matrix(params[i])
    # Apply the block to the matrix
    mat = jnp.einsum(indices, block_matrix, mat)
    if n == 2:
        # No rolling for n=2
        return mat, params, K

    # roll axes by two, so that the next block can be applied to
    # the first two axes again. Update roll position K accordingly.
    mat = roll_axes(mat, k=2, n=n, fixed_trailing=1)
    K = K + 2

    # Roll axes additionally if we are close to ``K=n``, because then the blocks wrap around
    # and we need to shift between layers of blocks with even-odd target qubits and odd-even
    # target qubits, respectively. Update roll position K accordingly.
    mat, K = jax.lax.cond(
        K <= n - 2,
        lambda mat, K: (mat, K),
        lambda mat, K: jax.lax.cond(
            K == n - 1,
            lambda mat, K: (roll_axes(mat, k=1 + n % 2, n=n, fixed_trailing=1), n % 2),
            lambda mat, K: (
                mat if n % 2 == 1 else roll_axes(mat, k=1, n=n, fixed_trailing=1),
                1 - n % 2,
            ),
            mat,
            K,
        ),
        mat,
        K,
    )

    return mat, params, K


@partial(jax.jit, static_argnames=("n", "num_cz", "final_rolls", "make_block_matrix"))
def apply_blocks(params, mat, n, num_cz, final_rolls, make_block_matrix):
    """Apply all blocks of the universal SU(N) or SO(N) circuit to a matrix.

    Args:
        params (jnp.ndarray): Parameters for all blocks of gates in the circuit. Should have
            shape ``(num_cz, num_params_per_cz)`` where ``num_cz`` is the number of blocks
            that are applied and ``num_params_per_cz`` is ``4`` for SU and ``2`` for SO.
        mat (jnp.ndarray): Matrix to apply the gate blocks to, with shape ``(2,) * n + (2**n,)``.
        n (int): Number of qubits.
        num_cz (int): Number of blocks of gates, or entanglers, that will be applied.
        final_rolls (int): Amount of axis rolls that are necessary after applying all blocks to
            obtain correctly-ordered axes in the output matrix.
        make_block_matrix (callable): Function to create the matrix for one circuit block.

    Returns:
        jnp.ndarray: matrix with all blocks of gates applied to it. The output shape
        and qubit ordering match those of the input.

    """
    mat_and_params_and_K = (mat, params, 0)
    # Contraction indices: Two row indices and column indices each for the ``block_matrix``,
    # n indices for axes of size 2 and one index for axis of size 2**n for ``mat``,
    # and the same for the output ``mat``, with "ab" replaced by "AB".
    indices = f"ABab,{alphabet[:n+1]}->AB{alphabet[2:n+1]}"
    body = partial(_body3, n=n, indices=indices, make_block_matrix=make_block_matrix)
    # Apply all blocks
    mat, _, K = jax.lax.fori_loop(0, num_cz, body, mat_and_params_and_K)
    # Perform final axes roll
    return roll_axes(mat, k=final_rolls, n=n, fixed_trailing=1)


def _body4(i, mat_and_params_and_K, n, indices):
    """Private method for ``apply_blocks_sp``, which is used in ``matrix_v2``.
    Effectively applies a two-qubit block of gates to a matrix in special shape format, and rolls
    axes so that the next block can be applied.

    Args:
        i (int): Block index determining which parameters are used in the block of gates.
        mat_and_params_and_K (tuple[jnp.ndarray, int]): Matrix to which the block of gates is
            applied (in shape ``(2,) * n + (2**n,)``, parameters for all blocks applied in
            the circuit, and the current axis roll position ``K``.
        n (int): Number of qubits.
        indices (str): Contraction rule for the einsum call that implements the gate application.

    Returns:
        tuple[jnp.ndarray, int]: New matrix to which the block of gates was applied,
        unchanged array of parameters for the full circuit, and updated axis roll position ``K``.

    The shape of ``mat`` is ``(2,) * n + (2**n,)`` throughout, and the shape of ``params``
    is ``(L, 3)`` throughout, where ``L`` is the number of blocks to be applied in total.
    """
    mat, params, K = mat_and_params_and_K
    # Create matrix for block with current parameters
    block_matrix = make_sp_block(params[i])
    # Apply the block to the matrix
    mat = jnp.einsum(indices, block_matrix, mat)
    if n == 2:
        # No rolling for n=2
        return mat, params, K

    # Roll non-symplectic axes by 1, so that the next block
    # can be applied to the first two axes again. Update roll position K accordingly.
    mat = roll_axes(mat, k=1, n=n - 1, fixed_trailing=1, fixed_leading=1)
    K = K + 1

    return mat, params, K


@partial(jax.jit, static_argnames=("n", "num_cy", "final_rolls"))
def apply_blocks_sp(params, mat, n, num_cy, final_rolls):
    """Apply all blocks of the universal Sp(N) circuit to a matrix.

    Args:
        params (jnp.ndarray): Parameters for all blocks of gates in the circuit. Should have
            shape ``(num_cy, 3)`` where ``num_cy`` is the number of blocks that are applied.
        mat (jnp.ndarray): Matrix to apply the gate blocks to, with shape ``(2,) * n + (2**n,)``.
        n (int): Number of qubits.
        num_cy (int): Number of blocks of gates, or entanglers, that will be applied.
        final_rolls (int): Amount of axis rolls that are necessary after applying all blocks to
            obtain correctly-ordered axes in the output matrix.

    Returns:
        jnp.ndarray: matrix with all blocks of gates applied to it. The output shape
        and qubit ordering match those of the input.

    """
    mat_and_params_and_K = (mat, params, 0)
    # Contraction indices: Two row indices and column indices each for the ``block_matrix``,
    # n indices for axes of size 2 and one index for axis of size 2**n for ``mat``,
    # and the same for the output ``mat``, with "ab" replaced by "AB".
    indices = f"ABab,{alphabet[:n+1]}->AB{alphabet[2:n+1]}"
    body = partial(_body4, n=n, indices=indices)
    # Apply all blocks
    mat, _, K = jax.lax.fori_loop(0, num_cy, body, mat_and_params_and_K)
    # Perform final axes roll
    return roll_axes(mat, k=final_rolls, n=n - 1, fixed_trailing=1, fixed_leading=1)


def matrix_v2(n, group):
    """Create a function that computes the matrix for the universal circuit ansatz for the
    provided group acting on ``n`` qubits.

    Args:
        n (int): Number of qubits the circuit acts on.
        group (str): Group for which the circuit is universal. Must be one of ``"SU", "SO"``.

    Returns:
        callable: Function that takes ``4**n-1`` parameters and returns the ``(2**n, 2**n)``
        matrix corresponding to the universal ``SU(2**n)`` ansatz at those parameters.

    """
    if group not in ("SU", "SO", "Sp"):
        raise NotImplementedError("Only 'SU', 'SO' and 'Sp' are supported at the moment.")
    # Compute useful facts about the group and the ansatz
    dim, params_init, params_per_cz, num_cz, num_final_params = ansatz_specs(n, group)
    zeros = jnp.zeros(params_per_cz - num_final_params)

    if n == 1:
        if group == "SU":
            return jax.jit(lambda params: qml.Rot.compute_matrix(*params))
        elif group == "SO":
            return jax.jit(lambda params: qml.RY.compute_matrix(params[0]))
        elif group == "Sp":
            return jax.jit(lambda params: qml.Rot.compute_matrix(*params))

    elif n == 2:
        final_rolls = 0
    elif group == "Sp":
        final_rolls = n - 1 - num_cz % (n - 1)
    else:
        # Compute number of final roll moves
        num_cz_per_two_layers = n // 2 + (n - 1) // 2
        num_cz_in_rest_block = num_cz % num_cz_per_two_layers
        last_layer_starts_at_1, last_layer_num_cz = divmod(num_cz_in_rest_block, n // 2)
        final_rolls = (n - 2 * last_layer_num_cz - last_layer_starts_at_1) % n

    block_shape = (num_cz, params_per_cz)

    if group == "SU":

        @jax.jit
        def fn(params):
            # Layer of RZ rotations
            mat = first_layer(params[:n], n)
            # Layer of RY rotations
            mat = apply_second_layer(params[n : 2 * n], mat, n)
            # Layer of RZ rotations
            mat = apply_third_layer(params[2 * n : 3 * n], mat, n)
            # Two-qubit blocks
            block_params = jnp.concatenate([params[params_init:], zeros]).reshape(block_shape)
            mat = apply_blocks(block_params, mat, n, num_cz, final_rolls, make_su_block)
            # Reshape
            return mat.reshape((2**n, 2**n))

    elif group == "SO":

        @jax.jit
        def fn(params):
            mat = jnp.eye(2**n, dtype=jnp.float64).reshape((2,) * n + (2**n,))
            # Layer of RY rotations
            mat = apply_second_layer(params[:n], mat, n)
            # Two-qubit blocks
            block_params = jnp.concatenate([params[params_init:], zeros]).reshape(block_shape)
            mat = apply_blocks(block_params, mat, n, num_cz, final_rolls, make_so_block)
            # Reshape
            return mat.reshape((2**n, 2**n))

    elif group == "Sp":

        @jax.jit
        def fn(params):
            # "Layer" of a single RZ rotation on first qubit
            mat = first_layer_sp(params[0], n)
            # Layer of RY rotations
            mat = apply_second_layer(params[1 : n + 1], mat, n)
            # "Layer" of a single RZ rotation on first qubit
            mat = apply_third_layer_sp(params[n + 1], mat, n)
            # Two-qubit blocks
            block_params = jnp.concatenate([params[params_init:], zeros]).reshape(block_shape)
            mat = apply_blocks_sp(block_params, mat, n, num_cz, final_rolls)
            # Reshape
            return mat.reshape((2**n, 2**n))

    fn.__doc__ = f"""Compute the matrix of the universal circuit ansatz for {group}({2**n}).

Args:
    params (np.ndarray): Parameters for the single-qubit rotation gates in the ansatz.
        Should have length {dim}.

Returns:
    jnp.ndarray: Matrix of the universal ansatz at the given parameters.

This function is differentiable and optimized for JIT compatibility.
"""

    return fn


def matrix_v2_partial(n, group, num_cz):
    """Create a function that computes the matrix for the universal circuit ansatz for SU(N).

    Args:
        n (int): Number of qubits the circuit acts on.
        group (str): Group to which the circuit is tailored. Restricted to ``"SU"`` for now.
        num_cz (int): Number of CZ gates, and two-qubit blocks in general, to use.

    Returns:
        callable: Function that takes ``4**n-1`` parameters and returns the ``(2**n, 2**n)``
        matrix corresponding to the universal ``SU(2**n)`` ansatz at those parameters.

    """
    assert group == "SU"
    # Compute useful facts about the group and the ansatz
    *_, params_per_cz, univ_num_cz, _ = ansatz_specs(n, "SU")
    if num_cz == univ_num_cz:
        return matrix_v2(n, group)

    # Compute number of final roll moves
    num_cz_per_two_layers = n // 2 + (n - 1) // 2
    last_meta_block_cz = num_cz % num_cz_per_two_layers
    last_layer_starts_at_1, last_layer_num_cz = divmod(last_meta_block_cz, n // 2)
    final_rolls = (n - 2 * last_layer_num_cz - last_layer_starts_at_1) % n
    num_params = 3 * n + num_cz * params_per_cz

    @jax.jit
    def fn(params):
        # Layer of RZ rotations
        mat = first_layer(params[:n], n)
        # Layer of RY rotations
        mat = apply_second_layer(params[n : 2 * n], mat, n)
        # Layer of RZ rotations
        mat = apply_third_layer(params[2 * n : 3 * n], mat, n)
        if num_cz > 0:
            # Two-qubit blocks
            block_params = params[3 * n :].reshape((num_cz, params_per_cz))
            mat = apply_blocks(block_params, mat, n, num_cz, final_rolls, make_su_block)
        # Reshape
        return mat.reshape((2**n, 2**n))

    fn.__doc__ = f"""Compute the matrix of a partial circuit ansatz for {group}({2**n}).

Args:
    params (np.ndarray): Parameters for the single-qubit rotation gates in the ansatz.
        Should have length {num_params}.

Returns:
    jnp.ndarray: Matrix of the ansatz at the given parameters.

This function is differentiable and optimized for JIT compatibility.
"""

    return fn


@jax.jit
def _two_qubit_factors_4(wires):
    return jnp.abs(
        jnp.array(
            [
                (2 - wires[0]) * (3 - wires[0]) * (2 - wires[1]) * (3 - wires[1]) / 12,
                (1 - wires[0]) * (3 - wires[0]) * (1 - wires[1]) * (3 - wires[1]) / 3,
                (1 - wires[0]) * (2 - wires[0]) * (1 - wires[1]) * (2 - wires[1]) / 4,
                wires[0] * (3 - wires[0]) * wires[1] * (3 - wires[1]) / 4,
                wires[0] * (2 - wires[0]) * wires[1] * (2 - wires[1]) / 3,
                wires[0] * (1 - wires[0]) * wires[1] * (1 - wires[1]) / 12,
            ]
        )
    )


def _rot_3(params, wire, mat):
    """For matrix_v3."""
    return qml.Rot(*params, wire).matrix(wire_order=(0, 1, 2)) @ mat


@jax.jit
def _ry_3(param, wire, mat):
    mats = jnp.array([qml.RY(param, w).matrix(wire_order=(0, 1, 2)) for w in (0, 1, 2)])
    return jnp.einsum("f,fab,bc->ac", jnp.eye(3)[wire], mats, mat)


@jax.jit
def _ry_3_real(param, wire, mat):
    mats = jnp.array([qml.RY(param, w).matrix(wire_order=(0, 1, 2)) for w in (0, 1, 2)]).real
    return jnp.einsum("f,fab,bc->ac", jnp.eye(3)[wire], mats, mat)


@jax.jit
def _rz_3(param, wire, mat):
    mats = jnp.array([jnp.diag(qml.RZ(param, w).matrix(wire_order=(0, 1, 2))) for w in (0, 1, 2)])
    return jnp.einsum("f,fb,bc->bc", jnp.eye(3)[wire], mats, mat)


def _rot_4(params, wire, mat):
    """For matrix_v3."""
    return qml.Rot(*params, wire).matrix(wire_order=(0, 1, 2, 3)) @ mat


@jax.jit
def _ry_4(param, wire, mat):
    mats = jnp.array([qml.RY(param, w).matrix(wire_order=(0, 1, 2, 3)) for w in (0, 1, 2, 3)])
    return jnp.einsum("f,fab,bc->ac", jnp.eye(4)[wire], mats, mat)


@jax.jit
def _rz_4(param, wire, mat):
    mats = jnp.array([qml.RZ(param, w).matrix(wire_order=(0, 1, 2, 3)) for w in (0, 1, 2, 3)])
    return jnp.einsum("f,fab,bc->ac", jnp.eye(4)[wire], mats, mat)


_cz_mats_3 = jnp.array(
    [jnp.diag(qml.CZ(wires).matrix(wire_order=(0, 1, 2))) for wires in [(0, 1), (0, 2), (1, 2)]]
)


@jax.jit
def _cz_3(wires, mat):
    return jnp.einsum("f,fb,bc->bc", jnp.eye(3)[wires[0] + wires[1] - 1], _cz_mats_3, mat)


@jax.jit
def _cz_3_real(wires, mat):
    return jnp.einsum("f,fb,bc->bc", jnp.eye(3)[wires[0] + wires[1] - 1], _cz_mats_3, mat).real


_cz_mats_4 = jnp.array(
    [
        jnp.diag(qml.CZ(wires).matrix(wire_order=(0, 1, 2, 3)))
        for wires in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    ]
)


@jax.jit
def _cz_4(wires, mat):
    return jnp.einsum(
        "f,fb,bc->bc", jnp.eye(6)[wires[0] + wires[1] - (wires[0] < 1)], _cz_mats_4, mat
    )


@jax.jit
def _block_3_su(params, connec, mat):
    w1, w2 = connec
    mat = _cz_3((w1, w2), mat)
    mat = _ry_3(params[0], w1, mat)
    mat = _ry_3(params[1], w2, mat)
    mat = _rz_3(params[2], w1, mat)
    mat = _rz_3(params[3], w2, mat)
    return mat


@jax.jit
def _block_3_so(params, connec, mat):
    w1, w2 = connec
    mat = _cz_3_real((w1, w2), mat)
    mat = _ry_3_real(params[0], w1, mat)
    mat = _ry_3_real(params[1], w2, mat)
    return mat


@jax.jit
def _block_4_su(params, connec, mat):
    w1, w2 = connec
    mat = _cz_4((w1, w2), mat)
    mat = _ry_4(params[0], w1, mat)
    mat = _ry_4(params[1], w2, mat)
    mat = _rz_4(params[2], w1, mat)
    mat = _rz_4(params[3], w2, mat)
    return mat


def _block_body_3_su(i, data):
    params, mat, connections = data
    mat = _block_3_su(params[i], connections[i], mat)
    return params, mat, connections


def _block_body_3_so(i, data):
    params, mat, connections = data
    mat = _block_3_so(params[i], connections[i], mat)
    return params, mat, connections


def _block_body_4_su(i, data):
    params, mat, connections = data
    mat = _block_4_su(params[i], connections[i], mat)
    return params, mat, connections


def matrix_v3(n, group, num_cz):
    """Create a function that computes the matrix for the given ``group`` on ``n`` qubits
    with ``num_cz`` two-qubit blocks. Not suited to include partial two-qubit blocks.
    The pairs of qubits on which the two-qubit blocks act are passed as second input to
    the created function."""

    if (n, group) not in [(3, "SU"), (4, "SU"), (3, "SO")]:
        raise NotImplementedError

    dim, params_init, params_per_cz, *_ = ansatz_specs(n, group)
    N = 2**n
    if group == "SU":
        dtype = jnp.complex128
        if n == 3:
            single_rot = _rot_3
            block_body = _block_body_3_su
        elif n == 4:
            single_rot = _rot_4
            block_body = _block_body_4_su
    elif group == "SO":
        dtype = jnp.float64
        if n == 3:
            single_rot = _ry_3_real
            block_body = _block_body_3_so

    @jax.jit
    def fn(params, connections):
        mat = jnp.eye(N, dtype=dtype)
        for i in range(n):
            mat = single_rot(params[i:params_init:n], i, mat)
        block_params = params[params_init:].reshape((num_cz, params_per_cz))
        data = (block_params, mat, connections)
        _, mat, _ = jax.lax.fori_loop(0, num_cz, block_body, data)
        return mat

    return fn
