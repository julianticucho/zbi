import sys
from pathlib import Path
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cmb.simulators.planck_lite_r1 import PlanckLiteSimulatorR1

N = 50
prior_low  = np.array([0.021, 0.110, 1.038, 0.03, 3.00, 0.95])
prior_high = np.array([0.024, 0.125, 1.043, 0.16, 3.30, 0.99])
theta = torch.tensor(
    np.random.default_rng(42).uniform(prior_low, prior_high, size=(N, 6)),
    dtype=torch.float32,
)

sim = PlanckLiteSimulatorR1()

for njobs in [1, 2, 4, 8, 12, 16]:
    t0 = time.perf_counter()
    x = sim(theta, show_progress=False, n_jobs=njobs)
    dt = time.perf_counter() - t0
    print(f"n_jobs={njobs:2d}  →  {dt:.1f}s  shape={tuple(x.shape)}")
