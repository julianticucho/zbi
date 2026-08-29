import getpass, os, sys, time
import numpy as np

RUN_DIR = "chains/pliklite_r2"
os.makedirs(RUN_DIR, exist_ok=True)

info: dict = {
    "likelihood": {
        "planck_2018_highl_plik.TTTEEE_lite": None,
    },
    "theory": {
        "camb": {
            "output_params": ["H0"],
        },
    },
    "params": {
        "ombh2": {
            "prior": {"dist": "uniform", "min": 0.022383 - 5*0.00015, "max": 0.022383 + 5*0.00015},
            "ref": 0.022383,
            "latex": r"\Omega_b h^2",
        },
        "omch2": {
            "prior": {"dist": "uniform", "min": 0.12011 - 5*0.0012, "max": 0.12011 + 5*0.0012},
            "ref": 0.12011,
            "latex": r"\Omega_c h^2",
        },
        "theta_MC_100": {
            "prior": {"dist": "uniform", "min": 1.040909 - 5*0.00031, "max": 1.040909 + 5*0.00031},
            "ref": 1.040909,
            "latex": r"100\theta_\mathrm{MC}",
            "drop": True,
            "renames": "theta",
        },
        "cosmomc_theta": {
            "value": "lambda theta_MC_100: 1e-2 * theta_MC_100",
        },
        "tau": {
            "prior": {"dist": "uniform", "min": max(0.0543 - 5*0.0073, 0.005), "max": 0.0543 + 20*0.0073},
            "ref": 0.0543,
            "latex": r"\tau",
        },
        "logA": {
            "prior": {"dist": "uniform", "min": 3.0448 - 5*0.014, "max": 3.0448 + 20*0.014},
            "ref": 3.0448,
            "latex": r"\ln(10^{10} A_s)",
            "drop": True,
        },
        "As": {
            "value": "lambda logA: np.exp(logA) * 1e-10",
            "latex": r"A_s",
        },
        "ns": {
            "prior": {"dist": "uniform", "min": 0.96605 - 5*0.0042, "max": 0.96605 + 5*0.0042},
            "ref": 0.96605,
            "latex": r"n_s",
        },
        "A_planck": {
            "value": 1.0,
        },
    },
    "sampler": {
        "mcmc": {
            "max_samples": 50000,
            "Rminus1_stop": 0.02,
            "Rminus1_cl_stop": 0.2,
        },
    },
    "output": os.path.abspath(os.path.join(RUN_DIR, "chain")),
    "force": True,
    "speed": None,
}


def main() -> None:
    from cobaya.run import run

    start = time.time()
    updated_info, products = run(info)
    elapsed = time.time() - start

    chain_dir = os.path.join(RUN_DIR, "chain")
    print(f"Finished in {elapsed / 60:.1f} min")
    print(f"Samples in: {chain_dir}/")
    print(f"Prior to analyse: cobaya-run getdist {chain_dir}")


if __name__ == "__main__":
    main()
