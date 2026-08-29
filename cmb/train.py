import os, torch
from cmb.simulators import PlanckLiteSimulatorR1
from cmb.embeddings import PlanckLiteMLPEmbeddingV2
from zbi.pipeline import (
    init, simulate, train, update_proposal, 
    update_embedding, update_store
)

if __name__ == "__main__":
    device = "cuda"

    # sim = PlanckLiteSimulatorR1(use_cl=["tt", "te", "ee"])
    # emb = PlanckLiteMLPEmbeddingV2

    run_dir = "runs/planck_lite_r2"
    os.makedirs(run_dir, exist_ok=True)

    # x_o = sim.get_observation().unsqueeze(0).to(device)

    # init(
    #     run_dir=run_dir,
    #     x_o=x_o,
    #     simulator_class=PlanckLiteSimulatorR1,
    #     embedding_class=emb,
    #     embedding_kwargs=dict(dim_out=6),
    #     simulator_kwargs=dict(use_cl=["tt", "te", "ee"]),
    #     prior_low=(
    #         0.02237 - 5*0.00015,   # ombh2
    #         0.1200 - 5*0.0012,     # omch2
    #         1.04092 - 5*0.00031,   # theta_MC_200
    #         0.0544 - 5*0.0073,  # tau
    #         3.044 - 5*0.014,       # logA
    #         0.9649 - 5*0.0042,     # ns
    #     ),
    #     prior_high=(
    #         0.02237 + 5*0.00015,   # ombh2
    #         0.1200 + 5*0.0012,     # omch2
    #         1.04092 + 5*0.00031,   # theta_MC_100
    #         0.0544 + 20*0.0073,     # tau
    #         3.044 + 20*0.014,       # logA
    #         0.9649 + 5*0.0042,     # ns
    #     ),
    #     dim_theta=6,
    #     dim_x=613,
    #     zarr_N=100_000,
    #     zarr_chunk_size=256,
    # )

    # update_embedding(run_dir, emb, dim_out=10)
    # update_store(run_dir, new_N=250_000)


    # simulate(run_dir, round=0, n_sims=2500, n_jobs=11)
    # simulate(run_dir, round=0, n_sims=2500, n_jobs=11)
    # simulate(run_dir, round=0, n_sims=2500, n_jobs=11)
    # simulate(run_dir, round=0, n_sims=2500, n_jobs=11)
    # simulate(run_dir, round=0, n_sims=2500, n_jobs=11)
    # simulate(run_dir, round=0, n_sims=2500, n_jobs=11)
    # simulate(run_dir, round=0, n_sims=2500, n_jobs=11)
    # simulate(run_dir, round=0, n_sims=2500, n_jobs=11)
    # simulate(run_dir, round=0, n_sims=2500, n_jobs=11)
    # simulate(run_dir, round=0, n_sims=2500, n_jobs=11)

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=100,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r1"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=100,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r2"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=100,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r3"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=200,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r1"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=200,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r2"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=200,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r3"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=500,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r1"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=500,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r2"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=500,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r3"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=1000,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r1"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=1000,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r2"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=1000,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r3"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=2000,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r1"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=2000,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r2"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=2000,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r3"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=5000,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r1"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=5000,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r2"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=5000,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r3"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=10000,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r1"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=10000,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r2"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=10000,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r3"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=20000,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r1"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=20000,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r2"
    # )

    # train(
    #     run_dir,
    #     round=0,
    #     n_sims=20000,
    #     device=device,
    #     batch_size=64,
    #     lr=5e-4,
    #     max_epochs=2147483647,
    #     stop_after_epochs=20,
    #     hidden_features=50,
    #     num_transforms=5,
    #     num_blocks=2,
    #     load_to_ram=True,
    #     offset=0,
    #     tag="mlpv2_r3"
    # )

    train(
        run_dir,
        round=0,
        n_sims=50000,
        device=device,
        batch_size=64,
        lr=5e-4,
        max_epochs=2147483647,
        stop_after_epochs=20,
        hidden_features=50,
        num_transforms=5,
        num_blocks=2,
        load_to_ram=True,
        offset=0,
        tag="mlpv2_r1"
    )

    train(
        run_dir,
        round=0,
        n_sims=50000,
        device=device,
        batch_size=64,
        lr=5e-4,
        max_epochs=2147483647,
        stop_after_epochs=20,
        hidden_features=50,
        num_transforms=5,
        num_blocks=2,
        load_to_ram=True,
        offset=0,
        tag="mlpv2_r2"
    )

    train(
        run_dir,
        round=0,
        n_sims=50000,
        device=device,
        batch_size=64,
        lr=5e-4,
        max_epochs=2147483647,
        stop_after_epochs=20,
        hidden_features=50,
        num_transforms=5,
        num_blocks=2,
        load_to_ram=True,
        offset=0,
        tag="mlpv2_r3"
    )

    train(
        run_dir,
        round=0,
        n_sims=100000,
        device=device,
        batch_size=64,
        lr=5e-4,
        max_epochs=2147483647,
        stop_after_epochs=20,
        hidden_features=50,
        num_transforms=5,
        num_blocks=2,
        load_to_ram=True,
        offset=0,
        tag="mlpv2_r1"
    )

    train(
        run_dir,
        round=0,
        n_sims=100000,
        device=device,
        batch_size=64,
        lr=5e-4,
        max_epochs=2147483647,
        stop_after_epochs=20,
        hidden_features=50,
        num_transforms=5,
        num_blocks=2,
        load_to_ram=True,
        offset=0,
        tag="mlpv2_r2"
    )

    train(
        run_dir,
        round=0,
        n_sims=100000,
        device=device,
        batch_size=64,
        lr=5e-4,
        max_epochs=2147483647,
        stop_after_epochs=20,
        hidden_features=50,
        num_transforms=5,
        num_blocks=2,
        load_to_ram=True,
        offset=0,
        tag="mlpv2_r3"
    )

