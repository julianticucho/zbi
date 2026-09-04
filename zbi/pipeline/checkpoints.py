import torch
import numpy as np

from zbi.utils.checkpoint import load_checkpoint, load_maf_from_checkpoint, load_prior_from_checkpoint
from zbi.inference.posterior import Posterior
from zbi.pipeline._resolve import resolve_class


def _load_maf_from_run(
    run_dir: str,
    checkpoint: str,
    device: str = "cpu",
) -> tuple[torch.nn.Module, dict]:
    ckpt = load_checkpoint(f"{run_dir}/models/{checkpoint}")
    emb_config = ckpt.get("embedding_net_config", {})
    emb_class = resolve_class(emb_config["class"])
    emb_kwargs = {k: v for k, v in emb_config.items() if k != "class"}
    maf = load_maf_from_checkpoint(ckpt, embedding_net=emb_class(**emb_kwargs)).to(device)
    return maf, ckpt


def sample_model(
    run_dir: str,
    checkpoint: str,
    n_samples: int = 25_000,
    device: str = "cpu",
    x_o: torch.Tensor | None = None,
) -> np.ndarray:
    maf, _ = _load_maf_from_run(run_dir, checkpoint, device)

    if x_o is None:
        x_o = torch.load(f"{run_dir}/x_o.pt", weights_only=True).to(device)
    else:
        x_o = x_o.to(device)

    maf.eval()
    with torch.no_grad():
        samples = maf.sample(n_samples, x_o)
    return samples.cpu().numpy()


def kl_matrix_from_run(
    run_dir: str,
    checkpoints: list[str],
    n_samples: int = 10_000,
    device: str = "cpu",
    x_o: torch.Tensor | None = None,
    norm_posterior: bool = True,
    **kwargs,
) -> np.ndarray:
    if x_o is None:
        x_o = torch.load(f"{run_dir}/x_o.pt", weights_only=True).to(device)
    else:
        x_o = x_o.to(device)

    estimators = []
    for ckpt_name in checkpoints:
        maf, ckpt = _load_maf_from_run(run_dir, ckpt_name, device)
        prior = load_prior_from_checkpoint(ckpt)
        estimators.append(Posterior(maf, prior, device=device))

    from zbi.inference import kl_matrix
    return kl_matrix(
        estimators, x_o,
        n_samples=n_samples,
        norm_posterior=norm_posterior,
        **kwargs,
    )
