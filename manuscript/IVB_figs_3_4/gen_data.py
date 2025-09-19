import sys
import json
import jax
import numpy as np
from unicirc import matrix_v2, make_cost_fn, make_optimization_run, compile, ansatz_specs, sample_from_group
import optax
from functools import partial
from tqdm.auto import tqdm

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
n_targets = mode_config["n_targets"]
num_attempts = mode_config["num_attempts"]
direc = mode_config["data_direc"]
progress_bar = True

data = {n: [] for n in ns}
for n in ns:
    n_epochs_ = n_epochs[n]
    num_records_ = num_records[n]
    if num_records_ is None:
        num_records_ = n_epochs_

    # Re-process max_const into the actual max_const that takes num_records_ into account
    epochs_per_record = n_epochs_ // num_records_
    max_const_ = max((2, max_const[n] // epochs_per_record))

    d, *_ = ansatz_specs(n, group)
    optimizer = optax.lbfgs(learning_rate=None, memory_size=5 * d)
    targets = sample_from_group(n, n_targets, group, seed)

    if BATCHING:
        assert (n_targets % NUM_BATCHES) == 0
        BATCH_SIZE = n_targets // NUM_BATCHES
        targets = targets[BATCH_SIZE * BATCH_IDX: BATCH_SIZE * (BATCH_IDX + 1)]

    matrix_fn = matrix_v2(n, group)
    cost_fn = make_cost_fn(matrix_fn)
    run_optimization = make_optimization_run(cost_fn, optimizer)

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

    if BATCHING:
        filename_stub = f"{group}_n-{n}_targets-{n_targets}_epochs-{n_epochs_}_tol-{tol}_batch_{BATCH_IDX}_{NUM_BATCHES}"
    else:
        filename_stub = f"{group}_n-{n}_targets-{n_targets}_epochs-{n_epochs_}_tol-{tol}"

    for target_idx, target in tqdm(enumerate(targets), total=len(targets)):
        print(f"{target_idx} / {len(targets)}")
        target_dag = target.conj().T
        energies, thetas, successful = compile(d, partial(opt_run, target_dag=target_dag), key=seed, tol=None, max_attempts=num_attempts)
        data[n].append(energies)

    cost_data = {f"target {i}": c for i, c in enumerate(data[n])}
    np.savez(f"{direc}{filename_stub}", **cost_data)
