import os
import numpy as np
from zbi.pipeline import sample_model, simulate_obs
from zbi.utils.plotting import plot_ppc
from cmb.plot_cobaya_posterior import load_chain
from cmb.simulators import PlanckLiteSimulatorR1

if __name__ == "__main__":
    run_dir = "runs/planck_lite_r2"
    chain_samples = load_chain("chains/pliklite_r2")

    sim = PlanckLiteSimulatorR1()
    theta_true = [0.02237, 0.1200, 1.04092, 0.0544+5*0.0073, 3.044+5*0.014, 0.9649]

    # samples_0 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_5000_mlpv2_r1.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # samples_1 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_5000_mlpv2_r2.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # samples_2 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_5000_mlpv2_r3.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # plot(
    #     all_samples=[samples_0, samples_1, samples_2],
    #     output="plots/posteriors/planck_lite_r2_round_00000_5000_mlpv2_(r1,r2,r3)_prior_limits_simulated_obs.pdf",
    #     param_names=["ombh2", "omch2", "theta_mc", "tau", "logA", "ns"],
    #     param_labels=[r"\Omega_b h^2", r"\Omega_c h^2", r"100\theta_{MC}", r"\tau", r"\ln(10^{10} A_s)", r"n_s"],
    #     sample_labels=["train 1 (5000 sims)", "train 2 (5000 sims)", "train 3 (5000 sims)"],
    #     sample_colors=['#cccccc','#666666','#000000'],
    #     filled=[True, False, False],
    #     limits=[
    #         (0.02237 - 5*0.00015, 0.02237 + 5*0.00015), 
    #         (0.1200 - 5*0.0012, 0.1200 + 5*0.0012), 
    #         (1.04092 - 5*0.00031, 1.04092 + 5*0.00031),
    #         (0.0544 - 5*0.0073, 0.0544 + 20*0.0073),
    #         (3.044 - 5*0.014, 3.044 + 20*0.014),
    #         (0.9649 - 5*0.0042, 0.9649 + 5*0.0042)
    #     ],
    #     gdist_fine_bins=500,
    #     gdist_scaling_factor=0.1,
    #     legend_fontsize=20,
    #     theta_true=theta_true
    # )

    # samples_0 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_10000_mlpv2_r1.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # samples_1 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_10000_mlpv2_r2.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # samples_2 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_10000_mlpv2_r3.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # plot(
    #     all_samples=[samples_0, samples_1, samples_2],
    #     output="plots/posteriors/planck_lite_r2_round_00000_10000_mlpv2_(r1,r2,r3)_prior_limits_simulated_obs.pdf",
    #     param_names=["ombh2", "omch2", "theta_mc", "tau", "logA", "ns"],
    #     param_labels=[r"\Omega_b h^2", r"\Omega_c h^2", r"100\theta_{MC}", r"\tau", r"\ln(10^{10} A_s)", r"n_s"],
    #     sample_labels=["train 1 (10000 sims)", "train 2 (10000 sims)", "train 3 (10000 sims)"],
    #     sample_colors=['#cccccc','#666666','#000000'],
    #     filled=[True, False, False],
    #     limits=[
    #         (0.02237 - 5*0.00015, 0.02237 + 5*0.00015), 
    #         (0.1200 - 5*0.0012, 0.1200 + 5*0.0012), 
    #         (1.04092 - 5*0.00031, 1.04092 + 5*0.00031),
    #         (0.0544 - 5*0.0073, 0.0544 + 20*0.0073),
    #         (3.044 - 5*0.014, 3.044 + 20*0.014),
    #         (0.9649 - 5*0.0042, 0.9649 + 5*0.0042)
    #     ],
    #     gdist_fine_bins=500,
    #     gdist_scaling_factor=0.1,
    #     legend_fontsize=20,
    #     theta_true=theta_true
    # )

    # samples_0 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_20000_mlpv2_r1.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # samples_1 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_20000_mlpv2_r2.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # samples_2 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_20000_mlpv2_r3.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # plot(
    #     all_samples=[samples_0, samples_1, samples_2],
    #     output="plots/posteriors/planck_lite_r2_round_00000_20000_mlpv2_(r1,r2,r3)_prior_limits_simulated_obs.pdf",
    #     param_names=["ombh2", "omch2", "theta_mc", "tau", "logA", "ns"],
    #     param_labels=[r"\Omega_b h^2", r"\Omega_c h^2", r"100\theta_{MC}", r"\tau", r"\ln(10^{10} A_s)", r"n_s"],
    #     sample_labels=["train 1 (20000 sims)", "train 2 (20000 sims)", "train 3 (20000 sims)"],
    #     sample_colors=['#cccccc','#666666','#000000'],
    #     filled=[True, False, False],
    #     limits=[
    #         (0.02237 - 5*0.00015, 0.02237 + 5*0.00015), 
    #         (0.1200 - 5*0.0012, 0.1200 + 5*0.0012), 
    #         (1.04092 - 5*0.00031, 1.04092 + 5*0.00031),
    #         (0.0544 - 5*0.0073, 0.0544 + 20*0.0073),
    #         (3.044 - 5*0.014, 3.044 + 20*0.014),
    #         (0.9649 - 5*0.0042, 0.9649 + 5*0.0042)
    #     ],
    #     gdist_fine_bins=500,
    #     gdist_scaling_factor=0.1,
    #     legend_fontsize=20,
    #     theta_true=theta_true
    # )

    # samples_0 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_50000_mlpv2_r1.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # samples_1 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_50000_mlpv2_r2.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # samples_2 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_50000_mlpv2_r3.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # plot(
    #     all_samples=[samples_0, samples_1, samples_2],
    #     output="plots/posteriors/planck_lite_r2_round_00000_50000_mlpv2_(r1,r2,r3)_prior_limits_simulated_obs.pdf",
    #     param_names=["ombh2", "omch2", "theta_mc", "tau", "logA", "ns"],
    #     param_labels=[r"\Omega_b h^2", r"\Omega_c h^2", r"100\theta_{MC}", r"\tau", r"\ln(10^{10} A_s)", r"n_s"],
    #     sample_labels=["train 1 (50000 sims)", "train 2 (50000 sims)", "train 3 (50000 sims)"],
    #     sample_colors=['#cccccc','#666666','#000000'],
    #     filled=[True, False, False],
    #     limits=[
    #         (0.02237 - 5*0.00015, 0.02237 + 5*0.00015), 
    #         (0.1200 - 5*0.0012, 0.1200 + 5*0.0012), 
    #         (1.04092 - 5*0.00031, 1.04092 + 5*0.00031),
    #         (0.0544 - 5*0.0073, 0.0544 + 20*0.0073),
    #         (3.044 - 5*0.014, 3.044 + 20*0.014),
    #         (0.9649 - 5*0.0042, 0.9649 + 5*0.0042)
    #     ],
    #     gdist_fine_bins=500,
    #     gdist_scaling_factor=0.1,
    #     legend_fontsize=20,
    #     theta_true=theta_true
    # )

    # samples_0 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_100000_mlpv2_r1.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # samples_1 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_100000_mlpv2_r2.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # samples_2 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_100000_mlpv2_r3.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # plot(
    #     all_samples=[samples_0, samples_1, samples_2],
    #     output="plots/posteriors/planck_lite_r2_round_00000_100000_mlpv2_(r1,r2,r3)_prior_limits_simulated_obs.pdf",
    #     param_names=["ombh2", "omch2", "theta_mc", "tau", "logA", "ns"],
    #     param_labels=[r"\Omega_b h^2", r"\Omega_c h^2", r"100\theta_{MC}", r"\tau", r"\ln(10^{10} A_s)", r"n_s"],
    #     sample_labels=["train 1 (100000 sims)", "train 2 (100000 sims)", "train 3 (100000 sims)"],
    #     sample_colors=['#cccccc','#666666','#000000'],
    #     filled=[True, False, False],
    #     limits=[
    #         (0.02237 - 5*0.00015, 0.02237 + 5*0.00015), 
    #         (0.1200 - 5*0.0012, 0.1200 + 5*0.0012), 
    #         (1.04092 - 5*0.00031, 1.04092 + 5*0.00031),
    #         (0.0544 - 5*0.0073, 0.0544 + 20*0.0073),
    #         (3.044 - 5*0.014, 3.044 + 20*0.014),
    #         (0.9649 - 5*0.0042, 0.9649 + 5*0.0042)
    #     ],
    #     gdist_fine_bins=500,
    #     gdist_scaling_factor=0.1,
    #     legend_fontsize=20,
    #     theta_true=theta_true
    # )

    # samples_0 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_125000_mlpv2_r1.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # samples_1 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_125000_mlpv2_r2.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # samples_2 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_125000_mlpv2_r3.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # plot(
    #     all_samples=[samples_0, samples_1, samples_2],
    #     output="plots/posteriors/planck_lite_r2_round_00000_125000_mlpv2_(r1,r2,r3)_prior_limits_simulated_obs.pdf",
    #     param_names=["ombh2", "omch2", "theta_mc", "tau", "logA", "ns"],
    #     param_labels=[r"\Omega_b h^2", r"\Omega_c h^2", r"100\theta_{MC}", r"\tau", r"\ln(10^{10} A_s)", r"n_s"],
    #     sample_labels=["train 1 (125000 sims)", "train 2 (125000 sims)", "train 3 (125000 sims)"],
    #     sample_colors=['#cccccc','#666666','#000000'],
    #     filled=[True, False, False],
    #     limits=[
    #         (0.02237 - 5*0.00015, 0.02237 + 5*0.00015), 
    #         (0.1200 - 5*0.0012, 0.1200 + 5*0.0012), 
    #         (1.04092 - 5*0.00031, 1.04092 + 5*0.00031),
    #         (0.0544 - 5*0.0073, 0.0544 + 20*0.0073),
    #         (3.044 - 5*0.014, 3.044 + 20*0.014),
    #         (0.9649 - 5*0.0042, 0.9649 + 5*0.0042)
    #     ],
    #     gdist_fine_bins=500,
    #     gdist_scaling_factor=0.1,
    #     legend_fontsize=20,
    #     theta_true=theta_true
    # )

    # samples_0 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_150000_mlpv2_r1.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # samples_1 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_150000_mlpv2_r2.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # samples_2 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_150000_mlpv2_r3.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # plot(
    #     all_samples=[samples_0, samples_1, samples_2],
    #     output="plots/posteriors/planck_lite_r2_round_00000_150000_mlpv2_(r1,r2,r3)_prior_limits_simulated_obs.pdf",
    #     param_names=["ombh2", "omch2", "theta_mc", "tau", "logA", "ns"],
    #     param_labels=[r"\Omega_b h^2", r"\Omega_c h^2", r"100\theta_{MC}", r"\tau", r"\ln(10^{10} A_s)", r"n_s"],
    #     sample_labels=["train 1 (150000 sims)", "train 2 (150000 sims)", "train 3 (150000 sims)"],
    #     sample_colors=['#cccccc','#666666','#000000'],
    #     filled=[True, False, False],
    #     limits=[
    #         (0.02237 - 5*0.00015, 0.02237 + 5*0.00015), 
    #         (0.1200 - 5*0.0012, 0.1200 + 5*0.0012), 
    #         (1.04092 - 5*0.00031, 1.04092 + 5*0.00031),
    #         (0.0544 - 5*0.0073, 0.0544 + 20*0.0073),
    #         (3.044 - 5*0.014, 3.044 + 20*0.014),
    #         (0.9649 - 5*0.0042, 0.9649 + 5*0.0042)
    #     ],
    #     gdist_fine_bins=500,
    #     gdist_scaling_factor=0.1,
    #     legend_fontsize=20,
    #     theta_true=theta_true
    # )

    # samples_0 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_175000_mlpv2_r1.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # samples_1 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_175000_mlpv2_r2.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # samples_2 = sample_model(
    #     run_dir=run_dir, 
    #     checkpoint="round_00000_175000_mlpv2_r3.pt", 
    #     n_samples=10_000,
    #     x_o=x_o
    # )

    # plot(
    #     all_samples=[samples_0, samples_1, samples_2],
    #     output="plots/posteriors/planck_lite_r2_round_00000_175000_mlpv2_(r1,r2,r3)_prior_limits_simulated_obs.pdf",
    #     param_names=["ombh2", "omch2", "theta_mc", "tau", "logA", "ns"],
    #     param_labels=[r"\Omega_b h^2", r"\Omega_c h^2", r"100\theta_{MC}", r"\tau", r"\ln(10^{10} A_s)", r"n_s"],
    #     sample_labels=["train 1 (175000 sims)", "train 2 (175000 sims)", "train 3 (175000 sims)"],
    #     sample_colors=['#cccccc','#666666','#000000'],
    #     filled=[True, False, False],
    #     limits=[
    #         (0.02237 - 5*0.00015, 0.02237 + 5*0.00015), 
    #         (0.1200 - 5*0.0012, 0.1200 + 5*0.0012), 
    #         (1.04092 - 5*0.00031, 1.04092 + 5*0.00031),
    #         (0.0544 - 5*0.0073, 0.0544 + 20*0.0073),
    #         (3.044 - 5*0.014, 3.044 + 20*0.014),
    #         (0.9649 - 5*0.0042, 0.9649 + 5*0.0042)
    #     ],
    #     gdist_fine_bins=500,
    #     gdist_scaling_factor=0.1,
    #     legend_fontsize=20,
    #     theta_true=theta_true
    # )

    samples_0 = sample_model(
        run_dir=run_dir, 
        checkpoint="round_00000_200000_mlpv2_r1.pt", 
        n_samples=1000,
    )

    samples_1 = sample_model(
        run_dir=run_dir, 
        checkpoint="round_00000_200000_mlpv2_r2.pt", 
        n_samples=1000,
    )

    samples_2 = sample_model(
        run_dir=run_dir, 
        checkpoint="round_00000_200000_mlpv2_r3.pt", 
        n_samples=1000,
    )

    samples_3 = sample_model(
        run_dir=run_dir, 
        checkpoint="round_00000_200000_mlpv2_r4.pt", 
        n_samples=1000,
    )

    samples_4 = sample_model(
        run_dir=run_dir, 
        checkpoint="round_00000_200000_mlpv2_r5.pt", 
        n_samples=1000,
    )

    samples_5 = sample_model(
        run_dir=run_dir, 
        checkpoint="round_00000_200000_mlpv2_r6.pt", 
        n_samples=1000,
    )

    samples_6 = sample_model(
        run_dir=run_dir, 
        checkpoint="round_00000_200000_mlpv2_r7.pt", 
        n_samples=1000,
    )

    samples_7 = sample_model(
        run_dir=run_dir, 
        checkpoint="round_00000_200000_mlpv2_r8.pt", 
        n_samples=1000,
    )

    samples_8 = sample_model(
        run_dir=run_dir, 
        checkpoint="round_00000_200000_mlpv2_r9.pt", 
        n_samples=1000,
    )

    samples_9 = sample_model(
        run_dir=run_dir, 
        checkpoint="round_00000_200000_mlpv2_r10.pt", 
        n_samples=1000,
    )

    samples_ensemble = np.concatenate([samples_0, samples_1, samples_2, samples_3, samples_4, samples_5, samples_6, samples_7, samples_8, samples_9], axis=0)

    g = plot_ppc(
        all_samples=[chain_samples, samples_ensemble],
        param_names=["ombh2", "omch2", "theta_mc", "tau", "logA", "ns"],
        param_labels=[r"\Omega_b h^2", r"\Omega_c h^2", r"100\theta_{MC}", r"\tau", r"\ln(10^{10} A_s)", r"n_s"],
        sample_labels=["MCMC Plik Lite", "Ensemble SBI"],
        sample_colors=['#666666','#000000'],
        filled=[True, False],
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
    )
    output = "plots/posteriors/planck_lite_r2_round_00000_200000_mlpv2_ensemble_(r1,r2,r3)_vs_plik_lite_prior_limits.pdf"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    g.export(output)













