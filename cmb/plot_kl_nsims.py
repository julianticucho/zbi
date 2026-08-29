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
# kl_matrix_from_run, igual que en cmb/plot_kl_matrix.py), y plotea el
# max/mean de los off-diagonales en función del número de simulaciones
def main(
    run_dir,
    ensembles,
    n_sims,
    n_samples=10_000,
    output=None,
):
    if not ensembles:
        print("Empty list of ensembles.")
        return
    if len(ensembles) != len(n_sims):
        raise ValueError("ensembles y n_sims deben tener la misma longitud")

    n_sims_arr = np.array(n_sims, dtype=float)
    max_kl, mean_kl = [], []
    print(f"{'n_sims':>8}  {'max KL':>9}  {'mean KL':>9}")
    for ns, ckpts in zip(n_sims, ensembles):
        K = kl_matrix_from_run(run_dir, ckpts, n_samples=n_samples)
        mx, mn = summarize(K)
        max_kl.append(mx)
        mean_kl.append(mn)
        print(f"{ns:>8d}  {mx:9.4g}  {mn:9.4g}")

    max_kl = np.array(max_kl)
    mean_kl = np.array(mean_kl)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(n_sims_arr, max_kl, "-", color="#006FED", lw=3, label="Max KL")
    ax.plot(n_sims_arr, mean_kl, "-", color="#333333", lw=3, label="Mean KL")

    ax.set_xscale("log")
    ax.set_xlabel(r"Training set size $n_{\mathrm{train}}$")
    ax.set_ylabel("KL divergence")
    ax.legend(prop={"size": 15})

    fig.tight_layout()
    if output is None:
        output = f"{run_dir}/plots/kl_vs_nsims.pdf"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"plot saved: {output}")


if __name__ == "__main__":
    run_dir = "runs/planck_lite_r2"

    # ensambles que yo quiero comparar: cada elemento es la lista de
    # checkpoints (los miembros r1..rN) de un tamaño de entrenamiento,
    # igual que en cmb/plot_kl_matrix.py
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
    output = f"plots/kl/planck_lite_r2_mlpv2_5.pdf"

    main(run_dir, ensembles, n_sims, output=output)