import os
from zbi.pipeline import sample_model, simulate_obs
from zbi.utils.plotting import plot_ppc
from cmb.load_chain import load_chain

if __name__ == "__main__":
    run_dir = "runs/planck_lite_r2"
    chain_samples = load_chain("chains/pliklite_r2")
    runs = ['r1', 'r2', 'r3']
    all_samples = []
    for r in runs:
        s = sample_model(
            run_dir=run_dir, n_samples=10000, 
            checkpoint=f"round_00000_200000_mlpv2_{r}.pt"
        )
        all_samples.append(s)

    g = plot_ppc(
        all_samples=all_samples,
        param_names=["ombh2", "omch2", "theta_mc", "tau", "logA", "ns"],
        param_labels=[r"\Omega_b h^2", r"\Omega_c h^2", r"100\theta_{MC}", r"\tau", r"\ln(10^{10} A_s)", r"n_s"],
        sample_labels=["Model 1", "Model 2", "Model 3"],
        sample_colors=['#cccccc','#666666','#000000'],
        filled=[True, False, False],
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
        legend_fontsize=30,
        output_path="plots/posteriors/planck_lite_r2_round_00000_200000_mlpv2_(r1,r2,r3)_prior_limits.pdf"
    )













