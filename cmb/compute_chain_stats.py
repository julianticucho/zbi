"""Calcular medias y desviaciones estandar de una cadena de Cobaya/GetDist.

Replica el metodo usado en notebooks/02_sbi_plik_lite.ipynb:
1. Se descarta una fraccion de burn-in de la cadena (por peso acumulado).
2. Se "deweightean" las muestras (cada linea se repite `weight` veces).
3. Se calculan media y desviacion estandar poblacional sobre las muestras deweighted.
"""
import argparse

import numpy as np


def load_chain(chain: str):
    with open(chain) as f:
        header = f.readline().lstrip("#").split()
    data = np.loadtxt(chain, skiprows=1)
    cols = {name: i for i, name in enumerate(header)}
    return data, cols


def remove_burn_in(data: np.ndarray, weight_col: int, frac: float) -> np.ndarray:
    if frac <= 0:
        return data
    weight = data[:, weight_col]
    total = weight.sum()
    cum = np.cumsum(weight)
    keep = cum >= frac * total
    return data[keep]


def deweight(data: np.ndarray, weight_col: int, rng: np.random.Generator | None = None):
    weight = data[:, weight_col]
    n = int(weight.sum())
    if rng is not None:
        idx = rng.choice(data.shape[0], size=n, p=weight / weight.sum())
        return data[idx]
    return np.repeat(data, weight.astype(int), axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chain", help="Ruta a la cadena (chain.1.txt)")
    parser.add_argument(
        "--params",
        nargs="+",
        default=["ombh2", "omch2", "theta_MC_100", "tau", "logA", "ns"],
        help="Columnas de la cadena a reportar",
    )
    parser.add_argument("--burn-fraction", type=float, default=0.3, help="Fraccion de burn-in a descartar")
    parser.add_argument("--sample", action="store_true", help="Deweightear por muestreo estocastico (default: repeticion)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data, cols = load_chain(args.chain)
    weight_col = cols["weight"]
    data = remove_burn_in(data, weight_col, args.burn_fraction)
    rng = np.random.default_rng(args.seed) if args.sample else None
    samples = deweight(data, weight_col, rng)

    print(f"chain: {args.chain}")
    print(f"lines: {data.shape[0]} | burn-in: {args.burn_fraction:.0%} | deweighted samples: {samples.shape[0]}")
    for name in args.params:
        col = samples[:, cols[name]]
        print(f"{name:<15} {col.mean():>14.9f} +/- {col.std():>6.6f}")


if __name__ == "__main__":
    main()