from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from getdist import loadMCSamples
from zbi.utils.plotting import plot_ppc


def load_chain(chain_dir: str) -> np.ndarray:
    s = loadMCSamples(os.path.join(chain_dir, "chain"))
    s.removeBurn(0)
    s.removeBurnFraction(0.3)
    names = list(s.getParamNames().list())
    by_name = {n: i for i, n in enumerate(names)}

    ombh2 = s.samples[:, by_name["ombh2"]]
    omch2 = s.samples[:, by_name["omch2"]]
    tau = s.samples[:, by_name["tau"]]
    logA = s.samples[:, by_name["logA"]]
    ns = s.samples[:, by_name["ns"]]
    theta_MC_100 = s.samples[:, by_name["theta_MC_100"]]

    return np.column_stack([ombh2, omch2, theta_MC_100, tau, logA, ns])


if __name__ == "__main__":
    chain_dir = os.path.join("chains", "pliklite_r2")
    samples = load_chain(chain_dir)

    g = plot_ppc(
        all_samples=[samples],
        true_parameter=[0.02237, 0.1200, 1.04092, 0.0544, 3.044, 0.9649],
        param_names=["ombh2", "omch2", "theta_MC_100", "tau", "logA", "ns"],
        param_labels=[
            r"\Omega_b h^2", r"\Omega_c h^2", r"100\theta_{MC}",
            r"\tau", r"\ln(10^{10} A_s)", r"n_s",
        ],
        sample_labels=["Plik lite TTTEEE"],
        filled=[True],
        limits=[
            (0.02237 - 5*0.00015, 0.02237 + 5*0.00015), 
            (0.1200 - 5*0.0012, 0.1200 + 5*0.0012), 
            (1.04092 - 5*0.00031, 1.04092 + 5*0.00031),
            (0.0544 - 5*0.0073, 0.0544 + 20*0.0073),
            (3.044 - 5*0.014, 3.044 + 20*0.014),
            (0.9649 - 5*0.0042, 0.9649 + 5*0.0042)
        ],
        gdist_fine_bins=500,
        gdist_scaling_factor=0.1,
        legend_fontsize=20,
        output_path="cmb/plots/pliklite_r2_posterior_prior_limits.pdf",
    )
    del g
    plt.close("all")
