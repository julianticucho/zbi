import json, os
import torch
from torch.distributions import Uniform

from zbi.utils.checkpoint import load_checkpoint
from zbi.utils.truncation import compute_bounding_box, TruncatedBoxPrior
from zbi.pipeline._resolve import resolve_prior
from zbi.pipeline.checkpoints import _load_maf_from_run
from zbi.inference.posterior import Posterior


def _sample_from_model(run_dir, checkpoint, original_prior, interest_dims,
                       x_o, n_samples, batch_size, device):
    maf, ckpt = _load_maf_from_run(run_dir, checkpoint, device)

    if interest_dims:
        marginal_prior = Uniform(
            original_prior.low[interest_dims],
            original_prior.high[interest_dims],
        )
        posterior = Posterior(maf, marginal_prior, device=device)
    else:
        posterior = Posterior(maf, original_prior, device=device)

    samples = posterior.sample((n_samples,), x_o)
    log_probs = []
    for i in range(0, len(samples), batch_size):
        lp = posterior.log_prob(samples[i : i + batch_size], x_o)
        log_probs.append(lp)
    log_probs = torch.cat(log_probs)

    return samples, log_probs


def update_proposal(
    run_dir: str,
    checkpoint: str | list[str],
    threshold: float,
    n_samples: int = 10_000,
    batch_size: int = 256,
    device: str = "cpu",
):
    if isinstance(checkpoint, str):
        checkpoints = [checkpoint]
    else:
        checkpoints = list(checkpoint)

    first_ckpt = load_checkpoint(f"{run_dir}/models/{checkpoints[0]}")
    interest_dims = first_ckpt.get("config", {}).get("interest_dims")
    original_prior = resolve_prior(first_ckpt["prior_meta"])

    x_o = torch.load(f"{run_dir}/x_o.pt", weights_only=True).to(device)

    all_samples, all_log_probs = [], []
    for ckpt_name in checkpoints:
        samples_i, log_probs_i = _sample_from_model(
            run_dir, ckpt_name, original_prior, interest_dims,
            x_o, n_samples, batch_size, device,
        )
        all_samples.append(samples_i)
        all_log_probs.append(log_probs_i)

    samples = torch.cat(all_samples)
    log_probs = torch.cat(all_log_probs)

    proposal_path = f"{run_dir}/proposal.json"
    if os.path.exists(proposal_path):
        with open(proposal_path) as f:
            cur = json.load(f)
        current_bounds = torch.tensor(
            [cur["bounds"]["low"], cur["bounds"]["high"]]
        )
    else:
        current_bounds = torch.stack([
            original_prior.low, original_prior.high
        ])

    if interest_dims:
        orig_int = torch.stack([
            original_prior.low[interest_dims],
            original_prior.high[interest_dims],
        ])
        bounds_int = compute_bounding_box(samples, log_probs,
                                          threshold, orig_int)
        new_bounds = current_bounds.clone()
        for i, dim in enumerate(interest_dims):
            new_bounds[0, dim] = bounds_int[0, i]
            new_bounds[1, dim] = bounds_int[1, i]
    else:
        orig_all = torch.stack([
            original_prior.low, original_prior.high
        ])
        new_bounds = compute_bounding_box(samples, log_probs,
                                          threshold, orig_all)

    json.dump({
        "prior_meta": first_ckpt["prior_meta"],
        "bounds": {
            "low": new_bounds[0].tolist(),
            "high": new_bounds[1].tolist(),
        },
        "threshold": threshold,
        "checkpoints": checkpoints,
    }, open(proposal_path, "w"), indent=2)
