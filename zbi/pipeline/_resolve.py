import importlib
import torch


def resolve_class(path: str):
    mod_path, cls_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod_path), cls_name)


def resolve_prior(prior_meta: dict):
    import torch.distributions as dist
    cls = getattr(dist, prior_meta["type"])
    kwargs = {k: torch.tensor(v) for k, v in prior_meta.items() if k != "type"}
    return cls(**kwargs)
