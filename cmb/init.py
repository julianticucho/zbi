import os
import zbi
from cmb.simulators import PlanckLiteSimulatorR1
from cmb.embeddings import PlanckLiteMLPEmbeddingV2

if __name__ == "__main__":
    run_dir = "runs/planck_lite_r2"
    os.makedirs(run_dir, exist_ok=True)

    sim = PlanckLiteSimulatorR1(use_cl=["tt", "te", "ee"])
    x_o = sim.get_observation()

    zbi.init(
        run_dir=run_dir,
        x_o=x_o,
        simulator_class=PlanckLiteSimulatorR1,
        simulator_kwargs=dict(use_cl=["tt", "te", "ee"]),
        embedding_class=PlanckLiteMLPEmbeddingV2,
        embedding_kwargs=dict(dim_out=10),
        prior_low=(
            0.022383 - 5*0.00015,   # ombh2
            0.12011 - 5*0.0012,     # omch2
            1.040909 - 5*0.00031,   # theta_MC_100
            max(0.0543 - 5*0.0073, 0.005),  # tau
            3.0448 - 5*0.014,       # logA
            0.96605 - 5*0.0042,     # ns
        ),
        prior_high=(
            0.022383 + 5*0.00015,   # ombh2
            0.12011 + 5*0.0012,     # omch2
            1.040909 + 5*0.00031,   # theta_MC_100
            0.0543 + 20*0.0073,     # tau
            3.0448 + 20*0.014,      # logA
            0.96605 + 5*0.0042,     # ns
        ),
        dim_theta=6,
        dim_x=613,
        zarr_N=200000,
        zarr_chunk_size=128,
        maf_kwargs=dict(
            hidden_features=50,
            num_transforms=5,
            num_blocks=2,
            dropout_probability=0.0,
            use_batch_norm=False,
        ),
    )
