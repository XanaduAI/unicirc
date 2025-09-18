import sys
import numpy as np
import matplotlib.pyplot as plt


MODE = sys.argv[1]

NUM_BATCHES = int(sys.argv[2])

BATCHING = NUM_BATCHES != 1

ns = [3, 4, 5]
group = "SU"
tol = 1e-12

n_epochs = {3: 1_000, 4: 5_000, 5: 10_000}

if MODE == "TEST":
    progress_bar = True
    n_targets = 2
    num_attempts = 2
    direc = "../test_data/indep_compilation/"

elif MODE == "PRODUCTION":
    progress_bar = True
    n_targets = 10
    num_attempts = 10
    direc = "../data/indep_compilation/"
else:
    raise ValueError


costs = {}
for n in ns:
    n_epochs_ = n_epochs[n]

    if BATCHING:
        BATCH_SIZE = n_targets // NUM_BATCHES
        costs[n] = []
        for BATCH_IDX in range(NUM_BATCHES):
            filename_stub = f"{group}_n-{n}_targets-{n_targets}_epochs-{n_epochs_}_tol-{tol}_batch_{BATCH_IDX}_{NUM_BATCHES}"
            X = np.load(f"{direc}{filename_stub}.npz")
            costs[n].extend([X[f"target {i}"] for i in range(BATCH_SIZE)])
    else:
        filename_stub = f"{group}_n-{n}_targets-{n_targets}_epochs-{n_epochs_}_tol-{tol}"
        X = np.load(f"{direc}{filename_stub}.npz")
        costs[n] = [X[f"target {i}"] for i in range(n_targets)]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18
plt.rcParams['text.usetex'] = True
cm = plt.get_cmap('turbo')

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
tols = 10.**np.arange(-5, -13, -1)
colors = [cm(i/len(tols)) for i in range(len(tols))]

# print(tols/
for tol, c in zip(tols, colors):
    kappa = []
    for n in ns:
        num_runs = n_targets * num_attempts
        values = np.reshape(costs[n], (num_runs, -1, 2))
        conv = np.sum(np.min(values[...,1], axis=1)< tol, axis=0)
        kappa.append(conv/num_runs)
    ax.plot(ns, kappa, color=c, marker="d", ls=":", label=f"$\\epsilon=10^{{{int(np.round(np.log10(tol)))}}}$")
ax.set_xlabel(r"$n$")
ax.set_xticks(ns)
ax.set_yticks([0.3, 0.65, 1.])
ax.set_yticklabels([r"$30\%$", r"$65\%$", r"$100\%$"])
ax.set_ylabel(r"$\kappa$")
ax.legend(loc="upper left", bbox_to_anchor=(1., 1.05))
plt.savefig(f"../figures/indep_compilation_success_rates_{group}.pdf", dpi=300, bbox_inches='tight')
