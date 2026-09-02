import os
from zbi.pipeline import kl_matrix_from_run
from zbi.utils.plotting import plot_kl_matrix


if __name__ == "__main__":
    run_dir = "runs/planck_lite_r2"
    checkpoints = [
        "round_00000_125000_mlpv2_r1.pt",
        "round_00000_125000_mlpv2_r2.pt",
        "round_00000_125000_mlpv2_r3.pt",
    ]

    k = kl_matrix_from_run(
        run_dir=run_dir,
        checkpoints=checkpoints,
        n_samples=10_000,
        device="cpu",
        x_o=None,
        norm_posterior=True,
        sample_batch_size=2_000,
        log_prob_batch_size=2_000,
    )

    fig, ax = plot_kl_matrix(
        k,
        output_path="plots/kl_matrix/round_00000_125000_mlpv2_(r1,r2,r3).pdf",
        sample_labels=["r1", "r2", "r3"],
        log_scale=False,
        annotate=True,
        annotate_fmt=".2g",
        cmap="viridis",
        vmin=None,
        vmax=None,
        linthresh=None,
        figsize=None,
        dim_theta=6,
    )

