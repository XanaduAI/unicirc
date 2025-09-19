import json
import sys
import matplotlib.pyplot as plt
import numpy as np

group = sys.argv[1]
MODE = sys.argv[2]
NUM_BATCHES = int(sys.argv[3])
BATCHING = NUM_BATCHES != 1

with open("config.json", "r") as config_file:
    config = json.load(config_file)[group]

ns = sorted(config["ns"])
n_epochs = {int(n): val for n, val in config["n_epochs"].items()}
tol = config["tol"]
seed = config["seed"]
mode_config = config["mode_config"][MODE]
n_targets = {int(n): val for n, val in mode_config["n_targets"].items()}
data_direc = mode_config["data_direc"]
figure_direc = mode_config["figure_direc"]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['text.usetex'] = True

costs = {}

for n in ns:
    n_epochs_ = n_epochs[n]
    n_targets_ = n_targets[n]
    if BATCHING:
        BATCH_SIZE = n_targets_ // NUM_BATCHES
        costs[n] = []
        for BATCH_IDX in range(NUM_BATCHES):
            filename_stub = f"{group}_n-{n}_targets-{n_targets_}_epochs-{n_epochs_}_tol-{tol}_batch_{BATCH_IDX}_{NUM_BATCHES}"
            X = np.load(f"{data_direc}cost_{filename_stub}.npz")
            costs[n].extend([X[f"target {i}"] for i in range(BATCH_SIZE)])
    else:
        filename_stub = f"{group}_n-{n}_targets-{n_targets_}_epochs-{n_epochs_}_tol-{tol}"
        X = np.load(f"{data_direc}cost_{filename_stub}.npz")
        costs[n] = [X[f"target {i}"] for i in range(n_targets_)]


n_panels = max([len(e) for n in ns for e in costs[n]])
ratios = [2] + [1] * (n_panels-1)
fig, axss = plt.subplots(len(ns), n_panels, figsize=(14, 7), gridspec_kw={"wspace": 0}, width_ratios=ratios, sharey=True)
if n_panels == 1:
    axss = axss[:, None]
cm = plt.get_cmap('turbo')

x_ticks_first_panel = {
    3: [10**1, 10**2, 10**3],
    4: [10**1, 10**3],
    5: [10**0, 10**2, 10**4],
}

for axs, n in zip(axss, ns, strict=True):
    n_epochs_ = n_epochs[n]
    n_targets_ = n_targets[n]

    colors = [cm(i/n_targets_) for i in range(n_targets_)]

    converged = np.zeros(len(axs),dtype=int)
    for i, (e, color) in enumerate(zip(costs[n], colors, strict=False)):
        for ax_idx, (ax, sub_e) in enumerate(zip(axs, e, strict=False)):
            ax.plot(*sub_e.T, label=f"Target {i}", color=color)
            converged[ax_idx] += int(sub_e[-1, 1]<tol)

    converged = np.cumsum(converged, dtype=int)

    for j, ax in enumerate(axs):
        if len(ax.get_lines())==0:
            ax.remove()
            continue
        ax.set_yscale("log")
        ax.set_xscale("log")

        #if n == ns[0]:
            #ax.set_title(f"Run {j+1}" if j==0 else f"{j+1}")
        ax.text(n_epochs_**0.95, 1, f"Run {j+1}" if j==0 else f"{j+1}", fontdict={"fontsize": 13}, ha="right", va="top")

        ax.plot([0, n_epochs_], [tol] * 2, color="k", ls=":")
        ax.set_xlim([1, n_epochs_])
        ax.set_ylim([tol/100, 2])
        if n==5:
            ax.text(n_epochs_**(1/20), tol/40, f"{converged[j]/n_targets_*100:.0f}\\%", fontdict={"fontsize": 15})
        else:
            ax.text(n_epochs_**(1/20), tol/40, f"{converged[j]/n_targets_*100:.1f}\\%", fontdict={"fontsize": 15})

        if j > 0:
            ax.set_xticks([], minor=True)
            ax.set_xticks([], minor=False)
            ax.tick_params(axis="y", length=0, width=0)
        else:
            ax.set_xticks(x_ticks_first_panel[n])
            ax.set_ylabel(f"$\\mathcal{{L}}(\\theta)$")
            ax.text(1.5, tol**(1/2), f"${n=}$")
plt.savefig(f"{figure_direc}fig_6_chained_compilation_cost_{group}.pdf", dpi=300, bbox_inches='tight')
