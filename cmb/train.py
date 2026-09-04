import os
import zbi

if __name__ == "__main__":
    device = "cuda"
    run_dir = "runs/planck_lite_r2"
    os.makedirs(run_dir, exist_ok=True)
    runs = ['r1', 'r2', 'r3']
    
    for r in runs:
        zbi.train(
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
            tag=f"mlpv2_{r}"
        )

