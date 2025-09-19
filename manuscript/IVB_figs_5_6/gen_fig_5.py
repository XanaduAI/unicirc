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

red = "#D7333B"
green = "#00973E"
xanablue = "#4D53C8"

colors = [red, xanablue, green]
xticks = [0.1, 1, 10, 60, 600, 3600, 36000]
xticklabels = ["$100ms$", "$1s$", "$10s$", "$1min$", "$10min$", "$1h$", "$10h$"]

for name in ["wall", "cpu"]:

    # Read data
    times = {}
    for n in ns:
        n_epochs_ = n_epochs[n]
        n_targets_ = n_targets[n]
        if BATCHING:
            times[n] = []
            for BATCH_IDX in range(NUM_BATCHES):
                filename_stub = f"{group}_n-{n}_targets-{n_targets_}_epochs-{n_epochs_}_tol-{tol}_batch_{BATCH_IDX}_{NUM_BATCHES}"
                X = np.load(f"{data_direc}{name}times_{filename_stub}.npz")
                times[n].extend(X["arr_0"])
        else:
            filename_stub = f"{group}_n-{n}_targets-{n_targets_}_epochs-{n_epochs_}_tol-{tol}"
            X = np.load(f"{data_direc}{name}times_{filename_stub}.npz")
            times[n] = X["arr_0"]

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for n, c in zip(ns, colors):
        _times = np.array(times[n])
        log_times = np.log10(_times)
        mean_time = np.mean(_times)
        out = ax.hist(log_times, label=f"${n=}, \\mu={mean_time:.2f}s$", bins=50, density=True, color=c, alpha=0.8)
    ax.legend()

    if name == "wall":
        ax.set_xlabel(r"Compilation wall clock time, $t [s]$")
    elif name == "cpu":
        ax.set_xlabel(r"Compilation CPU time, $t [s]$")

    ax.set_ylabel("Density")
    ax.set_xticks(np.log10(xticks));
    ax.set_xticklabels(xticklabels);
    plt.savefig(f"{figure_direc}fig_5_chained_compilation_{name}times_{group}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
