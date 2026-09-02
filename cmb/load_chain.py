from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from getdist import loadMCSamples
from zbi.utils.plotting import plot_ppc


def load_chain(chain_dir: str) -> np.ndarray:
    s = loadMCSamples(os.path.join(chain_dir, "chain"))
    s.removeBurn(0)
    s.removeBurnFraction(0.3)
    names = list(s.getParamNames().list())
    by_name = {n: i for i, n in enumerate(names)}

    ombh2 = s.samples[:, by_name["ombh2"]]
    omch2 = s.samples[:, by_name["omch2"]]
    tau = s.samples[:, by_name["tau"]]
    logA = s.samples[:, by_name["logA"]]
    ns = s.samples[:, by_name["ns"]]
    theta_MC_100 = s.samples[:, by_name["theta_MC_100"]]

    return np.column_stack([ombh2, omch2, theta_MC_100, tau, logA, ns])

