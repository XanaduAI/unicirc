"""This file generates the data for Sec. IVA and App. F on the expressibility measure
by Sim et al. Call it with

python gen_data.py NUM

where NUM is the number of qubits `n` that you would like to run the test for.
Afterwards, the figures can be produced with IVA_expressibility_plot.py.
Currently, only the group SU is supported.
"""
import sys
import numpy as np

from jax import numpy as jnp
from unicirc import matrix_v2, ansatz_specs
from tqdm.auto import tqdm

n = int(sys.argv[1])
group = "SU"

num_params, *_ = ansatz_specs(n, group)
mat_fn = matrix_v2(n, group)

def make_fidelities(num_samples):
    params = np.random.random((2 * num_samples, num_params)) * (2 * np.pi)
    mats = jnp.stack([mat_fn(p) for p in params])
    states = mats[..., 0].reshape((2, num_samples, 2**n))
    fidelities = jnp.abs(jnp.einsum("bi,bi->b", states[0].conj(), states[1])) ** 2
    return fidelities

# The following is a convenience choice to not make the computation time too long
if n==3:
    reg_samples = [100, 1000, 10_000, 100_000, 1_000_000]
else:
    reg_samples = [100, 1000, 10_000, 100_000]

_num = (len(reg_samples)-1) * 9 + 1
many_reg_samples = list(map(int, np.round(np.exp(np.linspace(np.log(min(reg_samples)), np.log(max(reg_samples)), _num)))))
assert all(s in many_reg_samples for s in reg_samples)
fidelities_reg_many = {num_sam: make_fidelities(num_sam) for num_sam in tqdm(many_reg_samples)}

DATA = {f"fidelities_{num_sam}": f for num_sam, f in fidelities_reg_many.items()}
jnp.savez(f"./data/expressibility_{n}", **DATA)

