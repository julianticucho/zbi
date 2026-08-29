import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from cmb.simulators.planck_lite_r1 import PlanckLiteSimulatorR1


def __getattr__(name):
    if name == "BmodeSimulator":
        from cmb.simulators.bmode_simulator import BmodeSimulator
        return BmodeSimulator
    if name == "BmodeSimulator128":
        from cmb.simulators.bmode_simulator import BmodeSimulator128
        return BmodeSimulator128
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
