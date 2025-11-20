import pennylane as qml
import jax
import numpy as np

import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
from scipy.sparse.linalg import eigsh
import optax


from unicirc import compile_adapt
filename = 'data/pentacene_16' # 'anthracene_6'
n_epochs = 50_000
n_thetas = 5
optimizer_name = "lbfgs"
lr = None
tol = 1e-10
seed = 1

Us2 = np.load(filename + ".npz", allow_pickle=True)["arr_0"]

Us = []
for U_i in Us2:
    n = int(np.ceil(np.log(len(U_i))/np.log(2)))
    U = np.eye(2**n, dtype=complex)
    U[:U_i.shape[0], :U_i.shape[1]] = U_i
    Us.append(U)

Us = np.array(Us, dtype=complex)

optimizer = optax.lbfgs(learning_rate=lr, memory_size=100)

for jjj, U in enumerate(Us):
    print(f"Compiling unitary {jjj+1} / {len(Us)}\n")
    energies, thetas, num_cz = compile_adapt(U,
        "SU",
        optimizer,
        n_epochs,
        num_czs=range(55, 61+1),
        max_attempts_per_num_cz=n_thetas,
        tol=tol,
        max_const=0,
        progress_bar=False,
        num_records=500,
        seed=seed,)

    results_name = filename + f"_results_n-{n}_epochs-{n_epochs}_thetas-{n_thetas}_{optimizer_name}-{lr}_seed-{seed}_{jjj}"
    results = {"energy": energies, "thetas": thetas, "num_cz": num_cz}
    print(f"saving under {results_name}")
    np.savez(results_name + ".npz", **results)


#import os
#os.system('sudo shutdown now')
