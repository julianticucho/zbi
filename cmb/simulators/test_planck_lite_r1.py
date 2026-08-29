import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from cmb.simulators.planck_lite_r1 import PlanckLiteSimulatorR1

N = 200
RNG = np.random.default_rng(42)

# Prior truncado basado en Cobaya chain (min/max con margen)
prior_low  = np.array([0.021, 0.110, 1.038, 0.03, 3.00, 0.95])
prior_high = np.array([0.024, 0.125, 1.043, 0.16, 3.30, 0.99])

theta = RNG.uniform(prior_low, prior_high, size=(N, 6))

sim = PlanckLiteSimulatorR1()

for i, t in enumerate(theta):
    x = sim.simulate(torch.tensor(t), seed=i)
    assert x.shape == (613,), f"i={i}: shape {x.shape}"
    assert torch.isfinite(x).all(), f"i={i}: non-finite"
    assert not torch.isnan(x).any(), f"i={i}: NaN present"

print(f"✓ {N} simulations passed (theta_MC_100 + logA).")

# Best-fit chi2
bf = np.array([0.02238, 0.12011, 1.04089, 0.0543, 3.044, 0.96605])
chi2 = sim.chi_squared(torch.tensor(bf))
assert np.isfinite(chi2)
print(f"✓ chi2 at best-fit = {chi2:.1f}")

print("✓ All tests passed.")
