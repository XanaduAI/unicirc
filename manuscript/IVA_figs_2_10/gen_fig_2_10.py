"""This file generates the figures for Sec. IVA and App. F on the expressibility measure
by Sim et al. After generating the data with IVA_expressibility_gen_data.py, call it with

python IVA_expressibility_plot.py NUM

where NUM is the number of qubits `n` that you would like to run the test for.
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18
plt.rcParams['text.usetex'] = True

n = int(sys.argv[1])

if n==3:
    reg_samples = [100, 1000, 10_000, 100_000, 1_000_000]
else:
    reg_samples = [100, 1000, 10_000, 100_000]
_num = (len(reg_samples)-1) * 9 + 1
many_reg_samples = list(map(int, np.round(np.exp(np.linspace(np.log(min(reg_samples)), np.log(max(reg_samples)), _num)))))

X = np.load(f"./data/expressibility_{n}.npz")
fidelities_reg_many = {int(key.split("_")[1]): f for key, f in X.items()}


red = "#D7333B"
yellow = "#ECC154"
green = "#52BD7C"
lightblue = "#82C7F0"
xanablue = "#4D53C8"

num_bins = 300
top_ylim = {3: 7.8, 4: 14.5, 5: 30.1}

def make_haar_expr(counts, F, dim, num_bins):
    F = (F[:-1] + F[1:]) / 2
    num_samples = np.sum(counts)
    probs = counts / num_samples
    probs_haar = (dim - 1) * (1 - F)**(dim - 2) / num_bins * num_samples
    expr = np.dot(probs, np.log(probs / probs_haar * num_samples, where=probs>0.))
    idle_expr = (dim - 1) * np.log(num_bins)
    rel_expr = - np.log(expr / idle_expr)
    return F, probs_haar, expr, rel_expr

plot_reg_samples = reg_samples[-3:]
colors = [red, green, yellow]
fig, axs = plt.subplots(2,2,figsize=(8, 5),gridspec_kw={"wspace": 0, "hspace": 0})

for ax in axs[0]:
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
for ax in axs[:, 1]:
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()

panel_labels = ["a)", "b)", "c)"]

for i, (num_samples, color, ax) in enumerate(zip(plot_reg_samples, colors, axs[:, :2].flat)):
    counts, F, _ = ax.hist(fidelities_reg_many[num_samples], bins=num_bins, range=(0, 1), color=color, alpha=0.0, density=True)
    F, probs_haar, expr, rel_expr = make_haar_expr(counts, F, 2**n, num_bins)
    label = f"$K=10^{int(np.round(np.log10(num_samples)))}$\n$\\mathrm{{Expr}}={expr:.5f}$"
    ax.hist(fidelities_reg_many[num_samples], bins=num_bins, range=(0, 1), label=label, color=color, alpha=0.5, density=True)
    ax.plot(F, probs_haar,  ls="-", lw=2, color=color, zorder=10);
    print(f"Regular ansatz, optimized; {num_samples} samples: {expr=},  {rel_expr=}")
    ax.legend()
    ax.set_xlabel(r"Fidelity", labelpad=-10)
    ax.set_xticks([0, 1])
    ax.set_ylabel(r"$P(F)$")
    ax.set_ylim((0, top_ylim[n]))
    ax.text(0.95, 0.1, panel_labels[i], transform=ax.transAxes, ha="right")

_T = axs[0, 1].get_yticks()[1:-1]
axs[0, 1].set_yticks(_T)

ax = axs[1, 1]
exprs = []
j = 0
ylim = axs[1, 0].get_ylim()
for num_samples in many_reg_samples:
    counts, F, _ = axs[1, 0].hist(
        fidelities_reg_many[num_samples],
        bins=num_bins,
        range=(0, 1),
        color=color,
        alpha=0.0,
        density=True,
    )
    *_, expr, _ = make_haar_expr(counts, F, 2**n, num_bins)
    exprs.append(expr)
    if num_samples in plot_reg_samples:
        ax.plot(num_samples, expr, marker="d", ls="", color=colors[j])
        j += 1
axs[1, 0].set_ylim(ylim)

def linear(x, a, b):
    return a * x +b

popt, pcov = curve_fit(linear, np.log10(many_reg_samples), np.log10(exprs), p0=[-1, 1])

ax.plot(many_reg_samples, exprs, marker="d", ls="", color=xanablue, zorder=-1)
ax.plot(
    many_reg_samples,
    10**(linear(np.log10(many_reg_samples), *popt)),
    label=f"${10**popt[1]:.2f} K^{{{popt[0]:.1f}}}$",
    c=lightblue,
    linewidth=1,
    # zorder=-10,
)
ax.legend()
ax.text(0.05, 0.1, "d)", transform=ax.transAxes, ha="left")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$K$", labelpad=-12)
ax.set_ylabel(r"$\mathrm{Expr}$", labelpad=-12)
ax.set_yticks([1e-1, 1e-3])
ax.set_yticks([], minor=True)

if n==3:
    plt.savefig("figures/fig_2_expressibility_histogram_SU_n_3.pdf", dpi=300, bbox_inches="tight")
else:
    plt.savefig(f"figures/fig_10_expressibility_histogram_SU_n_{n}.pdf", dpi=300, bbox_inches="tight")



