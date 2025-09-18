import pytest
import pennylane as qml
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
from unicirc.matrix import (
    matrix_v1 ,
    matrix_v2 ,
    matrix_v3,
)
from unicirc.universal_ansatze import make_ansatz, ansatz_specs

def ansatz_x():
    qml.X(0)

interfaces = ["numpy", "jax"]
groups = ["SU", "SO", "Sp"]

def random_array(size, seed, interface="numpy"):
    np.random.seed(seed)
    params = np.random.random(size)
    if interface=="jax":
        params = jnp.array(params)
    return params



class TestMatrixV1:

    def test_validation(self):
        mat_fn = matrix_v1(ansatz_x, interface="jax")
        with pytest.raises(NotImplementedError, match=r"Gate X\(0\) not "):
            mat_fn()
        mat_fn = matrix_v1(ansatz_x, interface="numpy")
        with pytest.raises(NotImplementedError, match=r"Gate X\(0\) not "):
            mat_fn()
        mat_fn = matrix_v1(ansatz_x, interface="torch")
        with pytest.raises(NotImplementedError, match="Interface torch not supported"):
            mat_fn()

    @pytest.mark.parametrize("interface", interfaces)
    @pytest.mark.parametrize("op", [qml.CZ((0, 1)), qml.RY(12.412, 0), qml.RZ(9.2, 0), qml.CY((0, 1)), qml.S(0)])
    def test_single_op(self, op, interface):
        def ansatz():
            qml.apply(op)

        mat = matrix_v1(ansatz, interface=interface)()
        assert np.allclose(mat, op.matrix(wire_order=sorted(op.wires)))

    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.parametrize("group", groups)
    @pytest.mark.parametrize("interface", interfaces)
    def test_compare_to_qml_matrix(self, n, group, interface):
        ansatz = make_ansatz(n, group)

        mat_fn = matrix_v1(ansatz, interface=interface)
        mat_fn_qml = qml.matrix(ansatz, wire_order=list(range(n)))
        dim, *_ = ansatz_specs(n, group)
        params = random_array(dim, seed=2152, interface=interface)
        assert np.allclose(mat_fn(params), mat_fn_qml(params))


class TestMatrixV2:

    def test_validation(self):
        with pytest.raises(NotImplementedError, match="Only 'SU', 'SO' and 'Sp' are"):
            _ = matrix_v2(3, "not a group")

    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.parametrize("group, dtype", zip(groups, [jnp.complex128, jnp.float64, jnp.complex128]))
    def test_basics(self, n, group, dtype):
        mat_fn = matrix_v2(n, group)
        dim, *_ = ansatz_specs(n, group)
        params = random_array(dim, seed=8215, interface="jax")
        matrix = mat_fn(params)
        assert matrix.dtype == dtype
        assert matrix.shape == (2**n, 2**n)
        assert np.allclose(matrix @ matrix.conj().T , np.eye(2**n))
        if group == "Sp":
            # Symplecticity test
            J = qml.Y(0).matrix(wire_order=range(n))
            assert np.allclose(matrix @ J @ matrix.T, J)


    @pytest.mark.parametrize("n", [2, 3, 4])
    @pytest.mark.parametrize("group", groups)
    def test_compare_to_qml_matrix(self, n, group):
        mat_fn = matrix_v2(n, group)

        mat_fn_qml = qml.matrix(make_ansatz(n, group), wire_order=list(range(n)))
        dim, *_ = ansatz_specs(n, group)
        params = random_array(dim, seed=8215, interface="jax")
        assert np.allclose(mat_fn(params), mat_fn_qml(params))


v3_supported =[(3, "SU", jnp.complex128), (4, "SU", jnp.complex128), (3, "SO", jnp.float64)]

class TestMatrixV3:

    def test_validation(self):
        with pytest.raises(NotImplementedError):
            matrix_v3(5, "SU", 1)
        with pytest.raises(NotImplementedError):
            matrix_v3(4, "SO", 1)
        with pytest.raises(NotImplementedError):
            matrix_v3(3, "Sp", 1)

    @pytest.mark.parametrize("n, group, dtype", v3_supported)
    @pytest.mark.parametrize("num_cz", [1, 4, 10])
    def test_basics(self, n, group, dtype, num_cz):
        mat_fn = matrix_v3(n, group, num_cz)
        connections = ([(i, i+1) for i in range(n-1)] * num_cz)[:num_cz]
        _, params_init, params_per_cz, *_ = ansatz_specs(n, group)
        params = random_array(params_init + num_cz * params_per_cz, 29512, interface="jax")
        matrix = mat_fn(params, jnp.array(connections))

        assert matrix.dtype == dtype
        assert matrix.shape == (2**n, 2**n)
        assert np.allclose(matrix @ matrix.conj().T , np.eye(2**n))
        if group == "Sp":
            # Symplecticity test
            J = qml.Y(0).matrix(wire_order=range(n))
            assert np.allclose(matrix @ J @ matrix.T, J)

    @pytest.mark.parametrize("n, group, dtype", v3_supported)
    def test_compare_to_other_matrix_universal(self, n, group, dtype):
        dim, _, params_per_cz, num_cz, final_params = ansatz_specs(n, group)
        connections = [(2*i+k, 2*i+1+k) for k in [0, 1] * num_cz for i in range((n-k)//2)][:num_cz]
        universal_params = random_array(dim, seed=8215, interface="jax")

        mat_fn = matrix_v3(n, group, num_cz)
        mat_fn_qml = qml.matrix(make_ansatz(n, group), wire_order=list(range(n)))
        v3_params = jnp.concatenate([universal_params, jnp.zeros(params_per_cz - final_params)])
        print(universal_params)
        print(v3_params)
        print(connections)
        mat_v3 =mat_fn(v3_params, jnp.array(connections))
        assert np.allclose(mat_v3, mat_fn_qml(universal_params))

        mat_fn_v2 = matrix_v2(n, group)
        mat_v2 =mat_fn_v2(universal_params)
        assert np.allclose(mat_v3, mat_v2)
