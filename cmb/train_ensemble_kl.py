import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "notebook"])
plt.rcParams["figure.dpi"] = 300

from zbi.pipeline import train_ensemble_kl


# grafica épocas vs max KL divergence del ensamble,
# recibe history directamente como dict con keys "epochs" y "max_kl"
def plot_kl_vs_epochs(run_dir, history, output=None):
    epochs = np.array(history["epochs"])
    max_kl = np.array(history["max_kl"])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(epochs, max_kl, "-", color="#006FED", lw=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Max $\mathrm{KL}(q_i \| q_j)$")
    ax.set_yscale("log")

    fig.tight_layout()
    if output is None:
        output = f"{run_dir}/plots/kl_vs_epochs.pdf"
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"plot saved: {output}")


if __name__ == "__main__":
    run_dir = "runs/planck_lite_r2"

    models, history = train_ensemble_kl(
        run_dir=run_dir,
        round=0,
        n_sims=200000,
        offset=0,
        n_members=3,
        kl_every=5,
        n_samples_kl=5000,
        device="cuda",
        batch_size=64,
        lr=5e-4,
        max_epochs=200,
        stop_after_epochs=20,
        hidden_features=50,
        num_transforms=5,
        num_blocks=2,
        tag="ens_kl",
        interest_dims=None,
        load_to_ram=True,
    )

    plot_kl_vs_epochs(
        run_dir,
        history,
        output=f"plots/kl_vs_epochs/planck_lite_r2_round_00000_200000_ens_kl.pdf",
    )
