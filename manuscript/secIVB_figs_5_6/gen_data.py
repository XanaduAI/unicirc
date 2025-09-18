import concurrent
from multiprocessing import Pool
import sys
from unicirc import matrix_v2, make_cost_fn, make_optimization_run, compile, ansatz_specs, sample_from_group
import jax
from jax import numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from functools import partial
from tqdm.auto import tqdm

import time
import optax

MODE = sys.argv[1]
NUM_BATCHES = int(sys.argv[2])
BATCH_IDX = int(sys.argv[3])
assert 0 <= BATCH_IDX <= NUM_BATCHES-1

BATCHING = NUM_BATCHES != 1

ns = [3, 4, 5]
group = "SU"
tol = 1e-10
n_epochs = {3: 1_000, 4: 5_000, 5: 10_000}
num_records = {n: max((1_000, epochs//2)) for n, epochs in n_epochs.items()}
max_const = {3: 100, 4: 100, 5: 100}

if MODE == "TEST":
    progress_bar = True
    n_targets = {3: 2, 4: 2, 5: 2}
    max_attempts = 10
    direc = "./test_data/"
elif MODE == "PRODUCTION":
    progress_bar = True
    n_targets = {3: 1_000, 4: 500, 5: 100}
    max_attempts = 20
    direc = "./data/"
else:
    raise ValueError

seed = 2152

data = {"wall_times": {}, "cpu_times": {}, "costs": {}, "theta_opt": {}, "success": {}}
for n in ns:
    n_epochs_ = n_epochs[n]
    n_targets_ = n_targets[n]
    num_records_ = num_records[n]
    if num_records_ is None:
        num_records = n_epochs_

    # Re-process max_const into the actual max_const that takes num_records into account
    epochs_per_record = n_epochs_ // num_records_
    max_const_ = max((2, max_const[n] // epochs_per_record))

    N = 2**n
    d, *_ = ansatz_specs(n, group)

    optimizer = optax.lbfgs(learning_rate=None, memory_size=5 * d)
    targets = sample_from_group(n, n_targets_, group, seed)
    if BATCHING:
        assert (n_targets_ % NUM_BATCHES) == 0
        BATCH_SIZE = n_targets_ // NUM_BATCHES
        targets = targets[BATCH_SIZE * BATCH_IDX: BATCH_SIZE * (BATCH_IDX + 1)]

    matrix_fn = matrix_v2(n, group)
    cost_fn = make_cost_fn(matrix_fn)
    run_optimization = make_optimization_run(cost_fn, optimizer)

    data["wall_times"][n] = []
    data["cpu_times"][n] = []
    data["costs"][n] = []
    data["theta_opt"][n] = []
    data["success"][n] = []

    opt_run = jax.jit(
        partial(
            run_optimization,
            n_epochs=n_epochs_,
            tol=tol,
            max_const=max_const_,
            progress_bar=progress_bar,
            num_records=num_records_,
        )
    )

    for target_idx, target in tqdm(enumerate(targets), total=len(targets)):
        target_dag = target.conj().T
        start_wall = time.time()
        start_cpu = time.process_time()

        energies, thetas, successful = compile(d, partial(opt_run, target_dag=target_dag), key=seed, tol=tol, max_attempts=max_attempts)

        end_wall = time.time()
        end_cpu = time.process_time()
        wall_time = end_wall - start_wall
        cpu_time = end_cpu - start_cpu

        data["wall_times"][n].append(wall_time)
        data["cpu_times"][n].append(cpu_time)
        data["costs"][n].append(energies)
        data["theta_opt"][n].append(thetas)
        data["success"][n].append(successful)

    if BATCHING:
        filename_stub = f"{group}_n-{n}_targets-{n_targets_}_epochs-{n_epochs_}_tol-{tol}_batch_{BATCH_IDX}_{NUM_BATCHES}"
    else:
        filename_stub = f"{group}_n-{n}_targets-{n_targets_}_epochs-{n_epochs_}_tol-{tol}"

    cost_data = {f"target {i}": c for i, c in enumerate(data["costs"][n])}
    jnp.savez(f"{direc}cost_{filename_stub}", **cost_data)
    params_data = {f"target {i}": th for i, th in enumerate(data["theta_opt"][n])}
    jnp.savez(f"{direc}params_{filename_stub}", **params_data)
    jnp.savez(f"{direc}walltimes_{filename_stub}", data["wall_times"][n])
    jnp.savez(f"{direc}cputimes_{filename_stub}", data["cpu_times"][n])
    jnp.savez(f"{direc}success_{filename_stub}", data["success"][n])
