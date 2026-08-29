import os, torch
from torch import nn

from zbi.simulators.base import Simulator


class SimuladorLineal(Simulator):
    def __init__(self):
        super().__init__()
        self.x_grid = torch.linspace(-5, 5, 100)

    def simulate(self, theta: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            torch.manual_seed(seed)
        a, b = theta[0].item(), theta[1].item()
        y = a * self.x_grid + b + 0.1 * torch.randn(100)
        return y


class EmbeddingNet(nn.Module):
    def __init__(self, dim_in: int = 100, dim_out: int = 5):
        super().__init__()
        self.net = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_example():
    import shutil
    from zbi.pipeline import init, simulate, train, simulate_obs, sample_model, update_proposal

    device = "cpu"
    run_dir = "runs/simple_example"

    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)

    x_o = simulate_obs(SimuladorLineal(), theta_true=(1.0, 0.5), seed=42).to(device)

    init(
        run_dir=run_dir,
        x_o=x_o,
        simulator_class=SimuladorLineal,
        embedding_class=EmbeddingNet,
        embedding_kwargs=dict(dim_in=100, dim_out=5),
        prior_low=(-3.0, -3.0),
        prior_high=(3.0, 3.0),
        dim_theta=2,
        dim_x=100,
        zarr_N=7_000,
        zarr_chunk_size=128,
    )

    simulate(run_dir, round=0, n_sims=500)
    train(run_dir, round=0, n_sims=500)
    # update_proposal(run_dir, checkpoint="round_00000_500.pt", threshold=1e-3)

    # x_samples = sample_model(run_dir, checkpoint="round_00000_500.pt", n_samples=25_000)

    # plot(
    #     all_samples=[x_samples],
    #     output="round_00000_500.pdf",
    #     run_dir=run_dir,
    #     theta_true=[1.0, 0.5],
    #     param_names=["theta_0", "theta_1"],
    #     param_labels=[r'\theta_0', r'\theta_1'],
    #     sample_labels=["round 1 (2k sims)"],
    #     sample_colors=["C0"],
    #     filled=[True],
    # )

    # print(f"Finalizado. Resultados en: {run_dir}/")


if __name__ == "__main__":
    run_example()
