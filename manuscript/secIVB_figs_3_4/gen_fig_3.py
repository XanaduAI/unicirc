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
    direc = "./test_data/"

elif MODE == "PRODUCTION":
    progress_bar = True
    n_targets = 10
    num_attempts = 10
    direc = "./data/"
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

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True, gridspec_kw={"wspace": 0.1})

for ax, n in zip(axs, ns):
    num_runs = n_targets * num_attempts
    n_epochs_ = n_epochs[n]
    colors = [cm(i/num_runs) for i in range(num_runs)]
    converged = 0
    sub_converged = {i: False for i in range(n_targets)}
    for i, (cost, c) in enumerate(zip(np.reshape(costs[n], (num_runs, -1, 2)), colors)):
        ax.plot(*cost.T, color=c)
        if np.min(cost[:, 1]) <= 1e-10:
            converged += 1
            sub_converged[i//num_attempts] = True

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Epochs")
    ax.set_title(f"${n=}$")
    ax.text(0.05, 0.3, f"$\\kappa={converged/num_runs*100:.1f}\\%$", transform=ax.transAxes)
    xlim = (ax.get_xlim()[0], n_epochs_)
    ax.plot(xlim, [1e-10]*2, color="k", ls=":", zorder=-10)
    ax.plot(xlim, [1e-12]*2, color="0.8", ls="--", zorder=-10)
    ax.set_xlim(xlim)
    print(f"{n=}. At least one converged for all targets: {all(sub_converged.values())}.")

axs[0].set_ylabel("$\\mathcal{L}(\\mathbf{\\theta})$")
plt.savefig(f"../figures/indep_compilation_cost_{group}.pdf", dpi=300, bbox_inches='tight')
