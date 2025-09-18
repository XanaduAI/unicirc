import sys
import matplotlib.pyplot as plt
import numpy as np

MODE = sys.argv[1]
NUM_BATCHES = int(sys.argv[2])

BATCHING = NUM_BATCHES != 1

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 16
plt.rcParams['text.usetex'] = True

ns = [3, 4, 5]
group = "SU"
tol = 1e-10
n_epochs = {3: 1_000, 4: 5_000, 5: 10_000}

if MODE == "TEST":
    n_targets = {3: 2, 4: 2, 5: 2}
    direc = "../test_data/chained_compilation/"
elif MODE == "PRODUCTION":
    n_targets = {3: 1_000, 4: 500, 5: 100}
    direc = "../data/chained_compilation/"
else:
    raise ValueError


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
                X = np.load(f"{direc}{name}times_{filename_stub}.npz")
                times[n].extend(X["arr_0"])
        else:
            filename_stub = f"{group}_n-{n}_targets-{n_targets_}_epochs-{n_epochs_}_tol-{tol}"
            X = np.load(f"{direc}{name}times_{filename_stub}.npz")
            times[n] = X["arr_0"]

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    for n, c in zip(ns, colors):
        times = np.array(times[n])
        log_times = np.log10(times)
        mean_time = np.mean(times)
        out = ax.hist(log_times, label=f"${n=}, \\mu={mean_time:.2f}s$", bins=50, density=True, color=c, alpha=0.8)
    ax.legend()

    if name == "wall":
        ax.set_xlabel(r"Compilation wall clock time, $t [s]$")
    elif name == "cpu":
        ax.set_xlabel(r"Compilation CPU time, $t [s]$")

    ax.set_ylabel("Density")
    ax.set_xticks(np.log10(xticks));
    ax.set_xticklabels(xticklabels);
    plt.savefig(f"../figures/chained_compilation_{name}times_{group}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
