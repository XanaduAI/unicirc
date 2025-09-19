from functools import partial
import optax
import unicirc
import jax
import matplotlib.pyplot as plt

group = "SU" # Group from which to sample a target and for which to use the ansatz
n = 3 # Qubit count
seed = 6381 # Randomness seed

# 1. Sample a random matrix from the given group on n qubits
target = unicirc.sample_from_group(n, n_samples=1, group=group, seed=seed)[0]

# 2. Set hyperparameters for compilation
# Number of optimization epochs (per attempt/random initialization)
n_epochs = 500

# Tolerance after which to interrupt a given compilation
tol_interrupt = 1e-10

# Number of datapoints to collect during optimization. Needs to be a divisor of n_epochs.
# Due to a bug, this currently needs to be less than n_epochs.
num_records = n_epochs // 2

# Number of epochs after which a constant cost function leads to interruption of optimization
# This number is re-processed below to take num_records into account.
max_const = 50

# Get dimension of the group to use for setting memory of optimizer
d, *_ = unicirc.ansatz_specs(n, group)
memory_size = 5 * d # Generous memory budget
learning_rate = None # L-BFGS uses line search, learning rate is not required

# 3. Create cost function, optimizer, and optimization executable
matrix_fn = unicirc.matrix_v2(n, group)
cost_fn = unicirc.make_cost_fn(matrix_fn)
optimizer = optax.lbfgs(learning_rate=None, memory_size=5 * d)
run_optimization = unicirc.make_optimization_run(cost_fn, optimizer)


# Reprocess max_const to take num_records into account
_epochs_per_record = n_epochs // num_records
max_const = max((2, max_const // _epochs_per_record))

opt_run = jax.jit(
    partial(
        run_optimization,
        n_epochs=n_epochs,
        tol=tol_interrupt,
        max_const=max_const,
        progress_bar=True,
        num_records=num_records,
    )
)

target_dag = target.conj().T
# Setting tol here implies that the compiler stops after an attempt has converged to tol_interrupt
# For tol=None, the compiler just attempts compilation `max_attempts` times.
energies, thetas, successful = unicirc.compile(d, partial(opt_run, target_dag=target_dag), key=seed, tol=tol_interrupt, max_attempts=10)

print(f"Compiler was successful: {successful}")
for idx, energy_attempt in enumerate(energies):
    plt.plot(*energy_attempt.T, label=f"Attempt {idx}")

plt.plot([0, n_epochs], [tol_interrupt]*2, ls=":", color="k")
plt.legend()
plt.yscale("log")
plt.show()
