from functools import partial

import jax.numpy as jnp

import pennylane as qml
import jax
import numpy as np

from pennylane import X, Y, Z, I

import scipy

import optax
from datetime import datetime

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import random


for run_i, (n, n_epochs, shuffle_) in enumerate(zip([3, 4, 4, 4], [20_000, 20_000, 20_000, 20_000], [False, False, True, True])):
    print("################################################")
    print(f"n = {n} run {run_i+1}")
    print("################################################")

    sun = list(qml.pauli.pauli_group(n))[::-1]

    dim = len(sun)
    filename = f'PPR_SU2^{n}_shuffle-{shuffle_}'

    # hyper parameters and definitions
    n_targets = 1
    n_thetas = 20
    targets = [scipy.stats.unitary_group(2**n, seed=i).rvs() for i in range(n_targets)]

    interrupt_tol=None
    optimizer_name = "lbfgs"

    if shuffle_:
        random.shuffle(sun)

    sun_m = np.array([qml.matrix(P, wire_order=range(n)) for P in sun])
    id_m = np.eye(2**n)

    def ansatz_func(params):
        U = jnp.eye(2**n, dtype=complex)
        for ll, P in enumerate(sun_m):
            U = (jnp.cos(params[ll]) * id_m - 1j * jnp.sin(params[ll]) * P) @ U
        
        return U

    jac_mat_fn = jax.jacobian(ansatz_func, argnums=0, holomorphic=True)

    lr = None

    if optimizer_name == "lbfgs":
        optimizer = optax.lbfgs(learning_rate=lr, memory_size=2*(4**n-1))
    if optimizer_name == "adam":
        optimizer = optax.adam(learning_rate=lr)

    results_name = filename + f"_results_n-{n}_epochs-{n_epochs}_targets-{n_targets}_thetas-{n_thetas}_{optimizer_name}-{lr}_run-{run_i}"
    print(results_name)

    # cost
    def cost(params, u_target):
        U = ansatz_func(params)
        return 1 - jnp.abs(jnp.trace(u_target.conj().T @ U))/(2**n)
    
    value_and_grad = jax.jit(jax.value_and_grad(cost, argnums=0))

    n_params = dim
    theta0s = jax.random.normal(jax.random.PRNGKey(1), shape=(n_thetas, n_params))

    @jax.jit
    def partial_step(grad_circuit, opt_state, theta, **kwargs):
        updates, opt_state = optimizer.update(grad_circuit, opt_state, theta, **kwargs, value=val, grad=grad_circuit, value_fn=cost)
        theta = optax.apply_updates(theta, updates)

        return opt_state, theta

    results = []
    t0 = datetime.now()
    ## Optimization loop
    try:
        energyss = []
        thetasss = []
        for j, u_target in enumerate(targets):
            print(f"Target {j+1} / {len(targets)}")
            energys, gradientss, thetass = [], [], []
            for i, theta in enumerate(theta0s):
                print(f"Seed {i+1} / {len(theta0s)}")

                energy, thetas = [], []
                opt_state = optimizer.init(theta)

                for ii in range(n_epochs):
                    val, grad_circuit = value_and_grad(theta, u_target=u_target)
                    opt_state, theta = partial_step(grad_circuit, opt_state, theta, u_target=u_target)

                    energy.append(val)
                    thetas.append(theta)
                
                energys.append(energy)
                pick = np.argmin(energy) - 1 # take only the best theta
                thetass.append(thetas[pick])
                print(f"Total time elapsed: {datetime.now() - t0} with min energy {np.min(energy)}")
            
            energyss.append(energys)
            thetasss.append(thetass)
            results = {"energy": energyss, "thetas": thetasss} # dummy initialize
            np.savez(results_name + ".npz", **results)

    except KeyboardInterrupt:
        print(
            "KeyboardInterrupt received. Cancelled the optimization and will return intermediate result."
        )

    t1 = datetime.now()

    print(f"total time: {t1 - t0}")
    
    jax.clear_caches()

# import os
# os.system('sudo shutdown now')