import os
import numpy as np
from zbi.pipeline import sample_model, simulate_obs
from zbi.utils.plotting import plot_ppc
from cmb.simulators import BmodeSimulator, BmodeSimulator128
from cmb.plot_cobaya_posterior import load_chain


if __name__ == "__main__":
    run_dir = "runs/planck_lite_r2"

    samples_2 = sample_model(
        run_dir=run_dir, 
        checkpoint="round_00000_50000.pt", 
        n_samples=10_000,
    )
    chain_samples = load_chain("chains/pliklite_r2")


    os.makedirs("cmb/plots", exist_ok=True)
    g = plot_ppc(
        all_samples=[samples_2, chain_samples],
        param_names=["ombh2", "omch2", "theta_mc", "tau", "logA", "ns"],
        param_labels=[r"\Omega_b h^2", r"\Omega_c h^2", r"100\theta_{MC}", r"\tau", r"\ln(10^{10} A_s)", r"n_s"],
        sample_labels=["round 1 (50k)", "Plik lite"],
        filled=[True, True],
        limits=[
            (0.02237 - 5*0.00015, 0.02237 + 5*0.00015), 
            (0.1200 - 5*0.0012, 0.1200 + 5*0.0012), 
            (1.04092 - 5*0.00031, 1.04092 + 5*0.00031),
            (0.0544 - 5*0.0073, 0.0544 + 20*0.0073),
            (3.044 - 5*0.014, 3.044 + 20*0.014),
            (0.9649 - 5*0.0042, 0.9649 + 5*0.0042)
        ],
        gdist_fine_bins=500,
        gdist_scaling_factor=0.1
    )
    g.export("cmb/plots/planck_lite_r2_round_00000_50000_vs_pliklite_r2_prior_limits.pdf")
