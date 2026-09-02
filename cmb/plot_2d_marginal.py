import matplotlib
matplotlib.use("Agg")

import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from matplotlib.lines import Line2D
from getdist import MCSamples, plots
from zbi.pipeline import sample_model

plt.style.use(["science", "bright"])
plt.rcParams["figure.dpi"] = 300
plt.rcParams["axes.labelsize"] = 16

x_param = "omch2"
y_param = "ns"

run_dir = "runs/planck_lite_r2"
n_samples = 10_000
n_rows = 3
n_cols = 3

ALL_PARAM_NAMES = ["ombh2", "omch2", "theta_mc", "tau", "logA", "ns"]
ALL_PARAM_LABELS = [
    r"\Omega_b h^2", r"\Omega_c h^2", r"100\theta_{MC}",
    r"\tau", r"\ln(10^{10} A_s)", r"n_s",
]

PARAM_INFO = {
    "ombh2":    {"label": r"$\Omega_b h^2$",     "center": 0.02237, "width": 0.00015},
    "omch2":    {"label": r"$\Omega_c h^2$",     "center": 0.1200,  "width": 0.0012},
    "theta_mc": {"label": r"$100\theta_{MC}$",   "center": 1.04092, "width": 0.00031},
    "tau":      {"label": r"$\tau$",              "center": 0.0544,  "width": 0.0073},
    "logA":     {"label": r"$\ln(10^{10} A_s)$",  "center": 3.044,   "width": 0.014},
    "ns":       {"label": r"$n_s$",               "center": 0.9649,  "width": 0.0042},
}

ENSEMBLES = [
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
SIM_COUNTS = [5000, 10000, 20000, 50000, 100000, 125000, 150000, 175000, 200000]

if __name__ == "__main__":
    assert x_param != y_param, "x_param and y_param must be different"
    n_plots = n_rows * n_cols

    x_info = PARAM_INFO[x_param]
    y_info = PARAM_INFO[y_param]
    x_lo = x_info["center"] - 5 * x_info["width"]
    x_hi = x_info["center"] + 5 * x_info["width"]
    y_lo = y_info["center"] - 5 * y_info["width"]
    y_hi = y_info["center"] + 5 * y_info["width"]

    output = f"plots/posteriors/planck_lite_r2_2d_marginal_{x_param}_{y_param}.pdf"
    ensembles = ENSEMBLES[:n_plots]
    sim_counts = SIM_COUNTS[:n_plots]
    colors = ["#cccccc", "#666666", "#000000"]

    g = plots.get_subplot_plotter(subplot_size=3.5)
    g.settings.scaling = False
    g.make_figure(n_plots, nx=n_cols, ny=n_rows)

    for idx in range(n_plots):
        row, col = divmod(idx, n_cols)
        g.subplots[row, col] = g.fig.add_subplot(g.gridspec[row, col])

    for idx, (ensemble, count) in enumerate(zip(ensembles, sim_counts)):
        row, col = divmod(idx, n_cols)
        ax = g.subplots[row, col]

        gd_samples = []
        for ckpt in ensemble:
            raw = sample_model(run_dir=run_dir, checkpoint=ckpt, n_samples=n_samples)
            gd = MCSamples(
                samples=np.array(raw, copy=True),
                names=ALL_PARAM_NAMES,
                labels=ALL_PARAM_LABELS,
            )
            gd.fine_bins = gd.fine_bins_2D = 500
            gd_samples.append(gd)

        for i, gd in enumerate(gd_samples):
            g.add_2d_contours(
                gd, x_param, y_param,
                ax=ax,
                filled=(i == 0),
                color=colors[i],
                alpha=0.6 if i == 0 else 0.9,
            )

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)

        if col == 0:
            ax.set_ylabel(y_info["label"], fontsize=16)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        if row == n_rows - 1:
            ax.set_xlabel(x_info["label"], fontsize=16)
        else:
            ax.set_xlabel("")
            ax.set_xticklabels([])

        ax.text(
            0.95, 0.05, f"{count} sims",
            transform=ax.transAxes, fontsize=17, va="bottom", ha="right",
        )

        if row == 0 and col == n_cols - 1:
            proxy_filled = Line2D([0], [0], color=colors[0], lw=8, alpha=0.6)
            proxy_line1 = Line2D([0], [0], color=colors[1], lw=1.5)
            proxy_line2 = Line2D([0], [0], color=colors[2], lw=1.5)
            ax.legend(
                handles=[proxy_filled, proxy_line1, proxy_line2],
                labels=["Model 1", "Model 2", "Model 3"],
                fontsize=12, loc="upper right",
            )

    g.fig.subplots_adjust(wspace=0, hspace=0)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    g.fig.savefig(output, bbox_inches="tight")
    print(f"plot saved: {output}")
    plt.close()
