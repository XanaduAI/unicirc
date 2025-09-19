import json
import sys
import numpy as np
import matplotlib.pyplot as plt

group = sys.argv[1]
MODE = sys.argv[2]
NUM_BATCHES = int(sys.argv[3])
BATCHING = NUM_BATCHES != 1

with open("config.json", "r") as config_file:
    config = json.load(config_file)[group]

ns = sorted(config["ns"])
tol = config["tol"]
n_epochs = {int(n): val for n, val in config["n_epochs"].items()}
mode_config = config["mode_config"][MODE]
n_targets = mode_config["n_targets"]
num_attempts = mode_config["num_attempts"]
data_direc = mode_config["data_direc"]
figure_direc = mode_config["figure_direc"]

costs = {}
for n in ns:
    n_epochs_ = n_epochs[n]

    if BATCHING:
        BATCH_SIZE = n_targets // NUM_BATCHES
        costs[n] = []
        for BATCH_IDX in range(NUM_BATCHES):
            filename_stub = f"{group}_n-{n}_targets-{n_targets}_epochs-{n_epochs_}_tol-{tol}_batch_{BATCH_IDX}_{NUM_BATCHES}"
            X = np.load(f"{data_direc}{filename_stub}.npz")
            costs[n].extend([X[f"target {i}"] for i in range(BATCH_SIZE)])
    else:
        filename_stub = f"{group}_n-{n}_targets-{n_targets}_epochs-{n_epochs_}_tol-{tol}"
        X = np.load(f"{data_direc}{filename_stub}.npz")
        costs[n] = [X[f"target {i}"] for i in range(n_targets)]


plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18
plt.rcParams['text.usetex'] = True
cm = plt.get_cmap('turbo')

fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True, gridspec_kw={"wspace": 0.1})

group_label = r"Sp^\ast" if group == "Sp" else group
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
    ax.set_title(f"${n=}, {group_label}({2**n})$")
    ax.text(0.05, 0.3, f"$\\kappa={converged/num_runs*100:.1f}\\%$", transform=ax.transAxes)
    xlim = (ax.get_xlim()[0], n_epochs_)
    ax.plot(xlim, [1e-10]*2, color="k", ls=":", zorder=-10)
    ax.plot(xlim, [1e-12]*2, color="0.8", ls="--", zorder=-10)
    ax.set_xlim(xlim)
    print(f"{n=}. At least one converged for all targets: {all(sub_converged.values())}.")

axs[0].set_ylabel("$\\mathcal{L}(\\mathbf{\\theta})$")
plt.savefig(f"{figure_direc}fig_3_compilation_stats_cost_{group}.pdf", dpi=300, bbox_inches='tight')
