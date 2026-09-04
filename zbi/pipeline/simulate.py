import json, os, shutil
import torch
from torch.distributions import Uniform
from zbi.data.zarr_store import ZarrStore
from zbi.utils.truncation import TruncatedBoxPrior
from zbi.pipeline._resolve import resolve_class, resolve_prior


def _simulate_round(
    n_sims: int,
    simulator,
    proposal,
    zarr_store,
    n_jobs: int = 1,
) -> int:
    theta = proposal.sample((n_sims,))
    x = simulator(theta, n_jobs=n_jobs)
    offset = zarr_store.num_filled
    zarr_store.append(theta, x)
    return offset


def simulate(
    run_dir: str,
    round: int,
    n_sims: int,
    seed: int | None = None,
    n_jobs: int = 1,
):
    with open(f"{run_dir}/config.json") as f:
        config = json.load(f)

    sim_kwargs = {k: v for k, v in config["simulator"].items() if k != "class"}
    simulator = resolve_class(config["simulator"]["class"])(**sim_kwargs)
    zarr_store = ZarrStore(f"{run_dir}/data")

    if round == 0:
        pm = config["prior_meta"]
        proposal = Uniform(torch.tensor(pm["low"]), torch.tensor(pm["high"]))
    else:
        proposal_path = f"{run_dir}/proposal.json"
        with open(proposal_path) as f:
            p = json.load(f)
        original = resolve_prior(p["prior_meta"])
        bounds = torch.tensor([p["bounds"]["low"], p["bounds"]["high"]])
        proposal = TruncatedBoxPrior(original, bounds)

        proposals_dir = f"{run_dir}/proposals"
        os.makedirs(proposals_dir, exist_ok=True)
        shutil.copy(proposal_path, f"{proposals_dir}/round_{round}.json")

    if seed is not None:
        torch.manual_seed(seed)
    offset = _simulate_round(n_sims, simulator, proposal, zarr_store, n_jobs=n_jobs)

    meta_path = f"{run_dir}/sim_round_{round}.json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            old = json.load(f)
        if offset != old["offset"] + old["n_sims"]:
            raise RuntimeError(
                f"Cannot accumulate: new data (offset={offset}) "
                f"is not contiguous with previous data "
                f"(offset={old['offset']}, n_sims={old['n_sims']})."
            )
        n_sims_total = old["n_sims"] + n_sims
        offset_final = old["offset"]
    else:
        n_sims_total = n_sims
        offset_final = offset

    meta = {"round_id": round, "n_sims": n_sims_total, "offset": offset_final}
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print(f"Round {round}: {n_sims} new sims, {n_sims_total} accumulated (offset={offset_final})")


def delete_last_round(run_dir: str) -> None:
    import glob

    sim_files = sorted(glob.glob(f"{run_dir}/sim_round_*.json"))
    if not sim_files:
        raise FileNotFoundError(f"No sim_round_*.json files found in {run_dir}")

    last_file = sim_files[-1]
    with open(last_file) as f:
        meta = json.load(f)

    round_id = meta["round_id"]
    offset = meta["offset"]
    n_sims = meta["n_sims"]

    zarr_store = ZarrStore(f"{run_dir}/data")
    zarr_store.clear(offset, n_sims)
    os.remove(last_file)

    proposal_file = f"{run_dir}/proposals/round_{round_id}.json"
    if os.path.exists(proposal_file):
        os.remove(proposal_file)

    print(f"Deleted round {round_id}: cleared {n_sims} sims at offset={offset}")


def simulate_obs(simulator, theta_true: tuple, seed: int = 42) -> torch.Tensor:
    theta_o = torch.tensor(theta_true)
    return simulator.simulate(theta_o, seed=seed).unsqueeze(0)
