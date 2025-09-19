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
group_label = r"Sp^\ast" if group == "Sp" else group
ax.text(3, 1., f"${group_label}$", ha="left", va="top")
ax.legend(loc="upper left", bbox_to_anchor=(1., 1.05))
if group == "SU":
    fig_no = 3
if group == "SO":
    fig_no = 9
if group == "Sp":
    fig_no = 10
plt.savefig(f"{figure_direc}fig_{fig_no}_compilation_stats_success_rates_{group}.pdf", dpi=300, bbox_inches='tight')
