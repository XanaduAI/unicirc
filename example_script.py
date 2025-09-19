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
tol = 1e-10

# Number of datapoints to collect during optimization. Needs to be a divisor of n_epochs.
# Due to a bug, this currently needs to be less than n_epochs.
num_records = n_epochs // 2

# Number of epochs after which a constant cost function leads to interruption of optimization
# This number is re-processed below to take num_records into account.
max_const = 50

costs, thetas, successful = unicirc.compile(
    target,
    group,
    key=seed,
    tol=tol,
    n_epochs=n_epochs,
    max_const=max_const,
    break_at_success=True,
)

for idx, cost_per_attempt in enumerate(costs):
    plt.plot(*cost_per_attempt.T, label=f"Attempt {idx}")

plt.plot([0, n_epochs], [tol]*2, ls=":", color="k")
plt.legend()
plt.yscale("log")
plt.show()
