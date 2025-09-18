import pytest
import numpy as np
import pennylane as qml
from unicirc.universal_ansatze import (
    ansatz_specs,
    _init_layer_su,
    _init_layer_so,
    _init_layer_sp,
    _circuit_block_su,
    _circuit_block_so,
    _circuit_block_sp,
    _final_circuit_block_su,
    _final_circuit_block_so,
    _final_circuit_block_sp,
    make_ansatz,
    )

class TestAnsatzSpecs:

    @pytest.mark.parametrize("n, expected", [(2, (15, 6, 4, 3, 1)), (3, (63, 9, 4, 14, 2)), (4, (255, 12, 4, 61, 3))])
    def test_su(self, n, expected):
        assert ansatz_specs(n, "SU") == expected

    @pytest.mark.parametrize("n, expected", [(2, (6, 2, 2, 2, 2)), (3, (28, 3, 2, 13, 1)), (4, (120, 4, 2, 58, 2))])
    def test_so(self, n, expected):
        assert ansatz_specs(n, "SO") == expected

    @pytest.mark.parametrize("n, expected", [(2, (10, 4, 3, 2, 3)), (3, (36, 5, 3, 11, 1)), (4, (136, 6, 3,44, 1))])
    def test_sp(self, n, expected):
        assert ansatz_specs(n, "Sp") == expected

def assert_all_ops_equal(*ops_lists):
    assert len(ops_lists) > 1
    for second in ops_lists[1:]:
        for op0, op1 in zip(ops_lists[0], second, strict=True):
            qml.assert_equal(op0, op1)

class TestInitLayers:

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_su(self, n):
        params = np.arange(3 * n)
        wires = list(range(n))
        with qml.queuing.AnnotatedQueue() as q:
            _init_layer_su(params, wires)

        ops = q.queue
        assert len(ops) == 3 * n
        assert np.allclose([op.data[0] for op in ops], params)
        assert [isinstance(op, (qml.RY if n<=i<2*n else qml.RZ)) for i, op in enumerate(ops)]
        assert [op.wires[0] for op in ops] == wires * 3


    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_so(self, n):
        params = np.arange(n)
        wires = list(range(n))
        with qml.queuing.AnnotatedQueue() as q:
            _init_layer_so(params, wires)

        ops = q.queue
        assert len(ops) == n
        assert np.allclose([op.data[0] for op in ops], params)
        assert [isinstance(op, qml.RY) for op in ops]
        assert [op.wires[0] for op in ops] == wires

    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_sp(self, n):
        params = np.arange(n + 2)
        wires = list(range(n))
        with qml.queuing.AnnotatedQueue() as q:
            _init_layer_sp(params, wires)

        ops = q.queue
        assert len(ops) == n + 2
        assert np.allclose([op.data[0] for op in ops], params)
        assert [isinstance(op, (qml.RY if 1<=i<n+2 else qml.RZ)) for i, op in enumerate(ops)]
        assert [op.wires[0] for op in ops] == [0] + wires + [0]

class TestCircuitBlocks:

    @pytest.mark.parametrize("wires", [(0, 1), (2, 3), ("a", "b")])
    def test_circuit_block_su(self, wires):
        params = np.arange(4)
        with qml.queuing.AnnotatedQueue() as q:
            _circuit_block_su(params, wires)

        ops = q.queue
        expected_ops = [qml.CZ(wires), qml.RY(0., wires[0]), qml.RY(1., wires[1]),qml.RZ(2., wires[0]), qml.RZ(3., wires[1])]
        assert_all_ops_equal(ops, expected_ops)

    @pytest.mark.parametrize("wires", [(0, 1), (2, 3), ("a", "b")])
    def test_circuit_block_so(self, wires):
        params = np.arange(2)
        with qml.queuing.AnnotatedQueue() as q:
            _circuit_block_so(params, wires)

        ops = q.queue
        expected_ops = [qml.CZ(wires), qml.RY(0., wires[0]), qml.RY(1., wires[1])]
        assert_all_ops_equal(ops, expected_ops)

    @pytest.mark.parametrize("wires", [(0, 1), (2, 3), ("a", "b")])
    def test_circuit_block_sp(self, wires):
        params = np.arange(3)
        with qml.queuing.AnnotatedQueue() as q:
            _circuit_block_sp(params, wires)

        ops = q.queue
        expected_ops = [qml.CY(wires[::-1]), qml.S(wires[1]), qml.RZ(0., wires[0]), qml.RY(1., wires[0]),qml.RY(2., wires[1])]
        assert_all_ops_equal(ops, expected_ops)


    @pytest.mark.parametrize("wires", [(0, 1), (2, 3), ("a", "b")])
    @pytest.mark.parametrize("num_params", [1, 2, 3, 4])
    def test_final_circuit_block_su(self, wires, num_params):
        params = np.arange(num_params)
        with qml.queuing.AnnotatedQueue() as q:
            _final_circuit_block_su(params, wires, num_params)

        ops = q.queue
        expected_ops = [qml.CZ(wires), qml.RY(0., wires[0]), qml.RY(1., wires[1]),qml.RZ(2., wires[0]), qml.RZ(3., wires[1])][:num_params+1]
        for op, exp_op in zip(ops, expected_ops, strict=True):
            qml.assert_equal(op, exp_op)

        if num_params == 4:
            with qml.queuing.AnnotatedQueue() as q2:
                _circuit_block_su(params, wires)
            ops_full = q2.queue
            assert_all_ops_equal(ops, ops_full)


    @pytest.mark.parametrize("wires", [(0, 1), (2, 3), ("a", "b")])
    @pytest.mark.parametrize("num_params", [1, 2])
    def test_circuit_block_so(self, wires, num_params):
        params = np.arange(num_params)
        with qml.queuing.AnnotatedQueue() as q:
            _final_circuit_block_so(params, wires, num_params)

        ops = q.queue
        expected_ops = [qml.CZ(wires), qml.RY(0., wires[0]), qml.RY(1., wires[1])][:num_params+1]
        for op, exp_op in zip(ops, expected_ops, strict=True):
            qml.assert_equal(op, exp_op)

        if num_params == 2:
            with qml.queuing.AnnotatedQueue() as q2:
                _circuit_block_so(params, wires)
            ops_full = q2.queue
            assert_all_ops_equal(ops, ops_full)


    @pytest.mark.parametrize("wires", [(0, 1), (2, 3), ("a", "b")])
    @pytest.mark.parametrize("num_params", [1, 2, 3])
    def test_circuit_block_sp(self, wires, num_params):
        params = np.arange(num_params)
        with qml.queuing.AnnotatedQueue() as q:
            _final_circuit_block_sp(params, wires, num_params)

        ops = q.queue
        expected_ops = [qml.CY(wires[::-1]), qml.S(wires[1]), qml.RZ(0., wires[0]), qml.RY(1., wires[0]), qml.RY(2., wires[1])][:num_params+2]
        for op, exp_op in zip(ops, expected_ops, strict=True):
            qml.assert_equal(op, exp_op)

        if num_params == 3:
            with qml.queuing.AnnotatedQueue() as q2:
                _circuit_block_sp(params, wires)
            ops_full = q2.queue
            assert_all_ops_equal(ops, ops_full)

groups = ["SU", "SO", "Sp"]

class TestMakeAnsatz:

    def test_validation(self):
        with pytest.raises(NotImplementedError, match="Only the groups"):
            make_ansatz(1, "Not a valid group")
        with pytest.raises(NotImplementedError, match="Only the groups"):
            make_ansatz(2, "Not a valid group")
        with pytest.raises(NotImplementedError, match="Only the groups"):
            make_ansatz(3, "Not a valid group")
        with pytest.raises(NotImplementedError, match="The number of qubits"):
            make_ansatz(0, "SU")

    @pytest.mark.parametrize("group, init_layer, dim", zip(groups, [_init_layer_su, _init_layer_so, _init_layer_sp], [3, 1, 3]))
    def test_single_qubit(self, group, init_layer, dim):
        ansatz = make_ansatz(1, group)
        params = np.arange(dim)

        with qml.queuing.AnnotatedQueue() as q:
            ansatz(params)

        ansatz_ops = q.queue

        with qml.queuing.AnnotatedQueue() as q2:
            init_layer(params, (0,))

        layer_ops = q2.queue
        assert_all_ops_equal(layer_ops, ansatz_ops)

    @pytest.mark.parametrize("group, entangler_ops", zip(groups, [1, 1, 2]))
    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_n_qubits(self, group, entangler_ops, n):
        dim, *_, num_cz, _ = ansatz_specs(n, group)
        ansatz = make_ansatz(n, group)
        params = np.arange(dim)

        with qml.queuing.AnnotatedQueue() as q:
            ansatz(params)

        ansatz_ops = q.queue
        assert len(ansatz_ops) == num_cz * entangler_ops + dim

        if group == "Sp":
            gate_set =(qml.CY, qml.RY, qml.RZ, qml.S)
        else:
            gate_set =(qml.CZ, qml.RY, qml.RZ)
        assert all(isinstance(op, gate_set) for op in ansatz_ops)
        ordered_data = [op.data[0] for op in ansatz_ops if len(op.data) > 0]
        assert np.allclose(ordered_data, params)

        assert ansatz.__doc__.startswith(f"Universal circuit ansatz for {group}")

