import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "notebook"])
plt.rcParams["figure.dpi"] = 300

from zbi.pipeline import kl_matrix_from_run

# función que resume la KL matrix en max/mean de los off-diagonales
# (todos los elementos fuera de la diagonal, ambos triángulos)
def summarize(K):
    off = K[~np.eye(K.shape[0], dtype=bool)]
    off = off[np.isfinite(off)]
    return float(off.max()), float(off.mean())


# función que computa la KL matrix de cada ensamble de la lista (con
# kl_matrix_from_run, igual que en cmb/plot_kl_matrix.py) y retorna el
# max de los off-diagonales en función del número de simulaciones
def compute_kl_vs_nsims(run_dir, ensembles, n_sims, n_samples=10_000):
    if not ensembles:
        raise ValueError("ensembles is empty")
    if len(ensembles) != len(n_sims):
        raise ValueError("ensembles y n_sims deben tener la misma longitud")

    n_sims_arr = np.array(n_sims, dtype=float)
    max_kl = []
    print(f"{'n_sims':>8}  {'max KL':>9}")
    for ns, ckpts in zip(n_sims, ensembles):
        K = kl_matrix_from_run(run_dir, ckpts, n_samples=n_samples)
        mx, _ = summarize(K)
        max_kl.append(mx)
        print(f"{ns:>8d}  {mx:9.4g}")

    return n_sims_arr, np.array(max_kl)


def plot_kl_nsims(
    results,
    labels=None,
    colors=None,
    output_path=None,
    figsize=(4, 4),
    lw=3,
    xlabel=None,
    ylabel=None,
):
    if labels is None:
        labels = [f"Curve {i + 1}" for i in range(len(results))]
    if colors is None:
        colors = [f"C{i}" for i in range(len(results))]
    if xlabel is None:
        xlabel = r"Training set size $n_{\mathrm{train}}$"
    if ylabel is None:
        ylabel = "KL divergence"

    fig, ax = plt.subplots(figsize=figsize)
    for (n_sims_arr, max_kl), label, color in zip(results, labels, colors):
        ax.plot(n_sims_arr, max_kl, "-", color=color, lw=lw, label=label)

    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(prop={"size": 15})

    fig.tight_layout()
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"plot saved: {output_path}")
    return fig


if __name__ == "__main__":
    run_dir = "runs/planck_lite_r2"

    ensembles = [
        ["round_00000_5000_mlpv2_r1.pt", "round_00000_5000_mlpv2_r2.pt", "round_00000_5000_mlpv2_r3.pt"],
        ["round_00000_10000_mlpv2_r1.pt", "round_00000_10000_mlpv2_r2.pt", "round_00000_10000_mlpv2_r3.pt"],
        ["round_00000_20000_mlpv2_r1.pt", "round_00000_20000_mlpv2_r2.pt", "round_00000_20000_mlpv2_r3.pt"],
        ["round_00000_50000_mlpv2_r1.pt", "round_00000_50000_mlpv2_r2.pt", "round_00000_50000_mlpv2_r3.pt"],
        ["round_00000_100000_mlpv2_r1.pt", "round_00000_100000_mlpv2_r2.pt", "round_00000_100000_mlpv2_r3.pt"],
        ["round_00000_125000_mlpv2_r1.pt", "round_00000_125000_mlpv2_r2.pt", "round_00000_125000_mlpv2_r3.pt"],
        ["round_00000_150000_mlpv2_r1.pt", "round_00000_150000_mlpv2_r2.pt", "round_00000_150000_mlpv2_r3.pt"],
        ["round_00000_175000_mlpv2_r1.pt", "round_00000_175000_mlpv2_r2.pt", "round_00000_175000_mlpv2_r3.pt"],
        ["round_00000_200000_mlpv2_r1.pt", "round_00000_200000_mlpv2_r2.pt", "round_00000_200000_mlpv2_r3.pt"],
    ]
    n_sims = [5000, 10000, 20000, 50000, 100000, 125000, 150000, 175000, 200000]

    result = compute_kl_vs_nsims(run_dir, ensembles, n_sims)

    plot_kl_nsims(
        [result],
        labels=["MLPv2"],
        colors=["#006FED"],
        output_path="plots/kl/planck_lite_r2_mlpv2_max_kl.pdf",
    )