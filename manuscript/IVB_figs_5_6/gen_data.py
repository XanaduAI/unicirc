"""This file generates the data for Sec. IVB (Fig. 5& 6) 
for compilation of random matrices into brickwall circuits.
Call it with

python gen_data.py GROUP MODE NUM_BATCHES BATCH_IDX

where 
- GROUP is one of SU, SO, Sp,
- MODE is TEST or PRODUCTION, leading to a small test data set or the full
  experiment data being produced.
- NUM_BATCHES is the number of batches the runs should be divided into.
- BATCH_IDX is the index of the batch to be run (must be < NUM_BATCHES)

If you prefer not to batch, just use NUM_BATCHES=1 and BATCH_IDX=0.
"""
import json
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

group = sys.argv[1]
MODE = sys.argv[2]
NUM_BATCHES = int(sys.argv[3])
BATCH_IDX = int(sys.argv[4])
BATCHING = NUM_BATCHES != 1
assert 0 <= BATCH_IDX <= NUM_BATCHES-1

with open("config.json", "r") as config_file:
    config = json.load(config_file)[group]

ns = config["ns"]
n_epochs = {int(n): val for n, val in config["n_epochs"].items()}
num_records = {int(n): val for n, val in config["num_records"].items()}
max_const = {int(n): val for n, val in config["max_const"].items()}
tol = config["tol"]
seed = config["seed"]
mode_config = config["mode_config"][MODE]
n_targets = {int(n): val for n, val in mode_config["n_targets"].items()}
max_attempts = mode_config["max_attempts"]
direc = mode_config["data_direc"]
progress_bar = True

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
