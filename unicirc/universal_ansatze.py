"""This file contains quantum circuit ansatze for the special unitary, special
orthogonal, and unitary symplectic groups, made up only of CZ, RZ and RY gates."""

import warnings
from functools import partial
import numpy as np
import pennylane as qml


def ansatz_specs(n, group):
    """Compute useful specs of the universal circuit ansatz for a given group on ``n``
    qubits.

    Args:
        n (int): Number of qubits.
        group (str): Group for which the ansatz is universal.

    Returns:
        tuple(int): The dimension of the group (number of parameters), the number of parameters
        in the initial layer of the ansatz, the number of parameters per CZ, the number of CZs,
        and the number of parameters in the final, potentially incomplete, circuit block.

    .. warning::

        This function is possibly flawed for ``group="Sp"``.

    """
    if group == "SU":
        dim = 4**n - 1
        params_init = 3 * n
        params_per_cz = 4
    elif group == "SO":
        dim = (4**n - 2**n) // 2
        params_init = n
        params_per_cz = 2
    elif group == "Sp":
        dim = (4**n + 2**n) // 2
        params_init = n + 2
        params_per_cz = 3
    else:
        raise ValueError(f"Unknown group: {group}")
    num_cz = int(np.ceil((dim - params_init) / params_per_cz))
    num_final_params = dim - params_init - (num_cz - 1) * params_per_cz
    return dim, params_init, params_per_cz, num_cz, num_final_params


def _init_layer_su(params, wires):
    """Initial layer of a universal ansatz for SU(N).

    Args:
        params (np.ndarray): Parameters for the initial layer gates. Should have length ``3n``
            for ``n`` wires.
        wires (Sequence[int]): Wires on which the initial layer acts.

    Returns:

    Queues:
        Single qubit rotations (``qml.RZ`` and ``qml.RZ``) for the initial ansatz layer.

    This function increases the parameter index across qubits (along the circuit width) first,
    and across time (along the circuit depth) second.

    This function supports broadcasting along the *last* axis of ``params``.
    """
    n = len(wires)
    [qml.RZ(p, w) for p, w, in zip(params[:n], wires, strict=True)]
    [qml.RY(p, w) for p, w, in zip(params[n : 2 * n], wires, strict=True)]
    [qml.RZ(p, w) for p, w, in zip(params[2 * n :], wires, strict=True)]


def _init_layer_so(params, wires):
    """Initial layer of a universal ansatz for SO(N).

    Args:
        params (np.ndarray): Parameters for the initial layer gates. Should have length ``n``
            for ``n`` wires.
        wires (Sequence[int]): Wires on which the initial layer acts.

    Returns:

    Queues:
        Single qubit rotations (``qml.RY``) for the initial ansatz layer.

    This function increases the parameter index across qubits (along the circuit width).

    This function supports broadcasting along the *last* axis of ``params``.
    """
    [qml.RY(p, w) for p, w in zip(params, wires, strict=True)]


def _init_layer_sp(params, wires):
    """Initial layer of a universal ansatz for Sp(N).

    Args:
        params (np.ndarray): Parameters for the initial layer gates. Should have length ``n+2``
            for ``n`` wires.
        wires (Sequence[int]): Wires on which the initial layer acts.

    Returns:

    Queues:
        Single qubit rotations (``qml.RZ`` and ``qml.RY``) for the initial ansatz layer.

    This function increases the parameter index in a specific manner:

    ```
    0: ──RZ(0)──RY(1)──RZ(5)─┤
    1: ─────────RY(2)────────┤
    2: ─────────RY(3)────────┤
    3: ─────────RY(4)────────┤
    ```

    This function supports broadcasting along the *last* axis of ``params``.
    """
    qml.RZ(params[0], wires[0])
    [qml.RY(p, w) for p, w in zip(params[1 : len(params) - 1], wires, strict=True)]
    qml.RZ(params[len(params) - 1], wires[0])


def _circuit_block_su(params, wires):
    """Circuit block for universal ansatz for SU(N).

    Args:
        params (np.ndarray): Parameters for the circuit block. Should have length 4.
        wires (Sequence[int]): Wires on which the circuit block acts. Should have length 2.

    Returns:

    Queues:
        A ``qml.CZ`` gate and four single-qubit rotations (two of ``qml.RY`` and ``qml.RZ`` each)

    This function increases the parameter index across qubits (along the circuit width) first,
    and across time (along the circuit depth) second.

    This function supports broadcasting along the *last* axis of ``params``.
    """
    qml.CZ(wires)
    qml.RY(params[0], wires[0])
    qml.RY(params[1], wires[1])
    qml.RZ(params[2], wires[0])
    qml.RZ(params[3], wires[1])


def _final_circuit_block_su(params, wires, num_params):
    """Last, potentially incomplete, circuit block for universal ansatz for SU(N).

    Args:
        params (np.ndarray): Parameters for the circuit block. Should have length ``num_params``.
        wires (Sequence[int]): Wires on which the circuit block acts. Should have length 2.
        num_params (int): Number of parameters passed in ``params``.

    Returns:

    Queues:
        A ``qml.CZ`` gate and ``num_params`` single-qubit rotations

    This function increases the parameter index across qubits (along the circuit width) first,
    and across time (along the circuit depth) second. Accordingly, the last ``4-num_params``
    rotations in that ordering are skipped, compared to ``_circuit_block_su``.

    This function supports broadcasting along the *last* axis of ``params``.
    """
    qml.CZ(wires)
    qml.RY(params[0], wires[0])
    if num_params > 1:
        qml.RY(params[1], wires[1])
        if num_params > 2:
            qml.RZ(params[2], wires[0])
            if num_params > 3:
                qml.RZ(params[3], wires[1])


def _circuit_block_so(params, wires):
    """Circuit block for universal ansatz for SO(N).

    Args:
        params (np.ndarray): Parameters for the circuit block. Should have length 2.
        wires (Sequence[int]): Wires on which the circuit block acts. Should have length 2.

    Returns:

    Queues:
        A ``qml.CZ`` gate and two single-qubit rotations (``qml.RY``).

    This function increases the parameter index across qubits (along the circuit width).

    This function supports broadcasting along the *last* axis of ``params``.
    """
    qml.CZ(wires)
    qml.RY(params[0], wires[0])
    qml.RY(params[1], wires[1])


def _final_circuit_block_so(params, wires, num_params):
    """Last, potentially incomplete, circuit block for universal ansatz for SO(N).

    Args:
        params (np.ndarray): Parameters for the circuit block. Should have length ``num_params``.
        wires (Sequence[int]): Wires on which the circuit block acts. Should have length 2.
        num_params (int): Number of parameters passed in ``params``.

    Returns:

    Queues:
        A ``qml.CZ`` gate and ``num_params`` single-qubit rotations (``qml.RY``).

    This function increases the parameter index across qubits (along the circuit width).

    This function supports broadcasting along the *last* axis of ``params``.
    """
    qml.CZ(wires)
    qml.RY(params[0], wires[0])
    if num_params > 1:
        qml.RY(params[1], wires[1])


def _circuit_block_sp(params, wires):
    """Circuit block for universal ansatz for Sp(N).

    Args:
        params (np.ndarray): Parameters for the circuit block. Should have length 3.
        wires (Sequence[int]): Wires on which the circuit block acts. Should have length 2.

    Returns:

    Queues:
        A ``qml.CY`` gate, a ``qml.S`` gate, and three single-qubit rotations
        (two ``qml.RY`` and one ``qml.RZ``)

    This function increases the parameter index across time on the first qubit first, and to
    the second qubit second.

    Note that this block, unlike those for SU and SU, is structurally not symmetric between
    the wires. The first qubit is treated as the "symplectic" qubit

    This function supports broadcasting along the *last* axis of ``params``.
    """
    qml.CY([wires[1], wires[0]])
    qml.S(wires[1])
    qml.RZ(params[0], wires[0])
    qml.RY(params[1], wires[0])
    qml.RY(params[2], wires[1])


def _final_circuit_block_sp(params, wires, num_params):
    """Last, potentially incomplete, circuit block for universal ansatz for Sp(N).

    Args:
        params (np.ndarray): Parameters for the circuit block. Should have length ``num_params``.
        wires (Sequence[int]): Wires on which the block acts. Should have length 2.
        num_params (int): Number of parameters passed in ``params``.

    Returns:

    Queues:
        A ``qml.CY`` gate, a ``qml.S`` gate, and ``num_params`` single-qubit rotations.

    This function increases the parameter index across time on the first qubit first, and to
    the second qubit second. Accordingly, the last ``3-num_params``
    rotations in that ordering are skipped, compared to ``_circuit_block_sp``.

    This function supports broadcasting along the *last* axis of ``params``.
    """
    qml.CY([wires[1], wires[0]])
    qml.S(wires[1])
    qml.RZ(params[0], wires[0])
    if num_params > 1:
        qml.RY(params[1], wires[0])
        if num_params > 2:
            qml.RY(params[2], wires[1])


def make_ansatz(n, group):
    """Create a universal circuit ansatz for a given group and number of qubits.

    Args:
        n (int): Number of qubits the ansatz acts on.
        group (str): Group for which the ansatz is universal. Must be ``"SO"`` or ``"SU"``.

    Returns:
        callable: Quantum function that implements the universal ansatz for ``n`` qubits and
        the provided ``group``. The ansatz takes a single argument, namely its parameters, and
        does not return anything but queues the operations for the ansatz. The ansatz supports
        parameter broadcasting in its *first* axis.
    """

    if n < 1:
        raise NotImplementedError("The number of qubits n must be a positive integer.")
    if group not in ("SU", "SO", "Sp"):
        raise NotImplementedError("Only the groups {'SU', 'SO', 'Sp'} are supported.")

    dim, params_init, params_per_cz, num_cz, num_final_params = ansatz_specs(n, group)
    if group == "SU":
        init_layer = _init_layer_su
        block = _circuit_block_su
        final_block = _final_circuit_block_su
    elif group == "SO":
        init_layer = _init_layer_so
        block = _circuit_block_so
        final_block = _final_circuit_block_so
    elif group == "Sp":
        init_layer = _init_layer_sp
        block = _circuit_block_sp
        final_block = _final_circuit_block_sp

    if n == 1:

        def ansatz(params):
            init_layer(qml.math.transpose(params), (0,))

        return ansatz

    if group == "Sp":
        # The block wires are simpler for the symplectic group because all layers use the same
        # wires on which the blocks act.
        block_wires = [(0, (i % (n - 1)) + 1) for i in range(num_cz)]
    else:
        # The following creates the first wire index for the blocks to be queued.
        # This logic was found by fiddling around a bit for the even and odd case, it is not obvious.
        # It leads to dense packing of the blocks, with even-indexed block layers starting to act on
        # qubit 0 and odd-indexed block layers starting to act on qubit 1.
        _block_wires = 2 * np.arange(num_cz) % (n - 1)
        if n % 2:
            _block_wires = _block_wires + np.arange(num_cz) // (n // 2) % 2
        block_wires = [(w, w + 1) for w in _block_wires]

    # Create slices that slice into the parameters array to extract parameters for each block
    params_idx = params_init
    param_slices = [
        slice(params_idx, params_idx := params_idx + params_per_cz) for _ in range(num_cz)
    ]

    final_block_ = partial(final_block, num_params=num_final_params)

    def ansatz(params):
        params = qml.math.transpose(params)  # For broadcasting support
        init_layer(params[:params_init], range(n))
        for _slice, _wires in zip(param_slices[:-1], block_wires[:-1]):
            block(params[_slice], _wires)
        final_block_(params[param_slices[-1]], block_wires[-1])

    if group == "Sp":
        extra_gates_str = f", {num_cz} ``qml.S`` gates,"
        entangler_str = "``qml.CY``"
    else:
        extra_gates_str = ""
        entangler_str = "``qml.CZ``"

    ansatz.__doc__ = f"""Universal circuit ansatz for {group}({2**n}).

Args:
    params (np.ndarray): Parameters for the single-qubit rotation gates in the ansatz.
        Should have length {dim}.

Returns:

Queues:
    {num_cz} entangling {entangler_str} gates{extra_gates_str} and {dim} single-qubit rotation gates
    (``qml.RZ`` and ``qml.RY``).

The ansatz support parameter broadcasting in the *first* axis of ``params``.
"""

    return ansatz


def make_tape_from_ansatz(ansatz, params):
    with qml.queuing.AnnotatedQueue() as q:
        ansatz(params)
    return qml.tape.QuantumScript.from_queue(q)
