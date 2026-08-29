from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from scipy.io import FortranFile

from zbi.simulators.base import Simulator

CL_NAMES = ["tt", "te", "ee"]

# directorio de los datos de planck de la likelihoood plik lite en mi pc
# sobreescribible con la variable de entorno PLANCK_DATA
DEFAULT_DATA_DIR = os.environ.get(
    "PLANCK_DATA",
    str(Path.home() / "Documents/cmb-sbi-inference/data/cobaya/data/planck_2018_pliklite_native"),
)


class PlanckLiteSimulatorR1(Simulator):
    def __init__(
        self,
        data_dir: Optional[str] = None,
        use_cl: Sequence[str] = ("tt", "te", "ee"),
        lmax_camb: int = 2508,
    ):
        super().__init__()

        if data_dir is None:
            data_dir = DEFAULT_DATA_DIR
        self.data_dir = Path(data_dir)
        self.use_cl = [c.lower() for c in use_cl]
        for c in self.use_cl:
            if c not in CL_NAMES:
                raise ValueError(f"Unknown spectrum '{c}', valid: {CL_NAMES}")

        # se carga blmin.dat y blmax.dat (límites de los bins), se 
        # definen offsets para navegar los archivos planos que 
        # contienen los datos concatenados de TT, TE y EE, la estructura
        # está documentada en plik_lite_v22.dataset

        # blmin_raw_ y blmax_raw_ son arrays de 645 enteros, donde cada
        # entrada es el multipolo min/max de un bin, son 215 bins por
        # espectro
        blmin_raw_: np.ndarray = np.loadtxt(self.data_dir / "blmin.dat").astype(int)
        blmax_raw_: np.ndarray = np.loadtxt(self.data_dir / "blmax.dat").astype(int)
        
        # plik usa lmin = 30, pues desde ahí la aproximación de
        # likelihood gaussiana es aceptable
        self._bin_lmin_offset = 30

        # los offsets son las posiciones en bweight.dat donde empiezan
        # los pesos de cada espectro (se ven en el valor del primer 
        # blmin de cada bloque) offsets: TT=0, TE=2479, EE=4958
        _offsets = {"tt": 0, "te": 2479, "ee": 4958}

        # son 215 bins crudos por espectro, sin embargo
        # hay 215, 199 y 199 datos por espectro, parece que
        # se descartan los bins ruidosos de EE y TE en los datos publicados
        _n_per_spec = {"tt": 215, "te": 215, "ee": 215}
        _n_data_spec = {"tt": 215, "te": 199, "ee": 199}

        self.lmin_per_bin: dict[str, np.ndarray] = {}
        self.lmax_per_bin: dict[str, np.ndarray] = {}
        self.bin_weights: dict[str, list[np.ndarray]] = {}

        # se definen offsets para navegar los archivos planos que 
        # contienen los datos concatenados de TT, TE y EE
        self._data_offsets = {"tt": 0, "te": 215, "ee": 414}
        self._bl_offsets = {"tt": 0, "te": 215, "ee": 430}
        self._weight_offsets = _offsets

        bweight_raw = np.loadtxt(self.data_dir / "bweight.dat")

        # para cada espectro, se extraen sus 215 l min/max, luego para
        # cada bin: 1) se usa raw_low como índice en bweight_raw para
        # obtener el vector w de pesos crudos, 2) se convierten los
        # l físicos a l real 3) se transforman los pesos de Cl a Dl
        # se guardan los pesos convertidos y los l efectivos
        for cl in CL_NAMES:
            off = self._bl_offsets[cl]
            woff = _offsets[cl]
            nbin = _n_per_spec[cl]
            lmin_phys = blmin_raw_[off : off + nbin]
            lmax_phys = blmax_raw_[off : off + nbin]

            weights_list = []
            for j in range(nbin):
                raw_low = lmin_phys[j]
                raw_high = lmax_phys[j]

                fi_start = raw_low
                fi_end = raw_high + 1
                w = bweight_raw[fi_start:fi_end].copy()

                l_phys_start = (raw_low - woff) + self._bin_lmin_offset
                ls_w = np.arange(l_phys_start, l_phys_start + len(w))
                w_dl = w * (2 * np.pi) / ls_w / (ls_w + 1)
                weights_list.append(w_dl)

            self.bin_weights[cl] = weights_list
            self.lmin_per_bin[cl] = (lmin_phys - woff) + self._bin_lmin_offset
            self.lmax_per_bin[cl] = (lmax_phys - woff) + self._bin_lmin_offset

        # se cargan los 613 Dl observados y la covarianza 613x613, se
        # reconstruye simétrica y se seleccionan solo los espectros usados,
        # extrañendo X_data, cov, inv_cov, y L_cov (cholesky)
        data: np.ndarray = np.loadtxt(self.data_dir / "cl_cmb_plik_v22.dat")
        f = FortranFile(self.data_dir / "c_matrix_plik_v22.dat", "r")
        cov_full = f.read_reals(dtype=float).reshape((613, 613))
        cov_full = np.tril(cov_full) + np.tril(cov_full, -1).T

        self.bin_indices: dict[str, list[int]] = {}
        used_global_rows: list[int] = []
        self.n_bins_total = 0

        for cl in CL_NAMES:
            data_off = self._data_offsets[cl]
            ndata = _n_data_spec[cl]
            nbin = _n_per_spec[cl]
            if cl in self.use_cl:
                self.bin_indices[cl] = list(range(ndata))
                used_global_rows.extend(range(data_off, data_off + ndata))
                self.n_bins_total += ndata
            else:
                self.bin_indices[cl] = []

        self.X_data = data[used_global_rows, 1].copy()
        self.cov = cov_full[np.ix_(used_global_rows, used_global_rows)].copy()
        self.invcov = np.linalg.inv(self.cov)
        self.L_cov = np.linalg.cholesky(self.cov)

        # se construye la matriz B para cada espectro (nbins x lmax+1)
        # donde cada fila tiene los pesos del bin en las columnas l
        # correspondientes, B @ Dl proyecta el espectro CAMB al espacio
        # de bins
        self.spec_col = {"tt": 0, "ee": 1, "te": 3}
        self.B: dict[str, np.ndarray] = {}
        self.Lmax_spec: dict[str, int] = {}

        for cl in CL_NAMES:
            if cl not in self.use_cl:
                continue
            nbin = len(self.bin_indices[cl])
            lmax = int(self.lmax_per_bin[cl][:nbin].max())
            self.Lmax_spec[cl] = lmax
            B = np.zeros((nbin, lmax + 1), dtype=np.float64)
            for j in range(nbin):
                lmin = int(self.lmin_per_bin[cl][j])
                lmax_j = int(self.lmax_per_bin[cl][j])
                w = self.bin_weights[cl][j]
                B[j, lmin : lmax_j + 1] = w
            self.B[cl] = B

        self.dim_x = self.n_bins_total

    # método público para obtener la observación de planck
    # en el formato plik lite
    def get_observation(self) -> torch.Tensor:
        return torch.tensor(self.X_data, dtype=torch.float32)

    # método que ejecuta CAMB con theta, obtiene Dl, y para cada
    # espectro aplica B @ Dl[:lmax+1] para proyectar al espacio de bins
    # concatena todo en un vector
    def model(self, theta: torch.Tensor) -> np.ndarray:
        import camb
        pars = self._build_camb_params(theta.numpy())
        results = camb.get_results(pars)
        lmax_needed = max(self.Lmax_spec.get(c, 0) for c in CL_NAMES)
        cls = results.get_total_cls(lmax=lmax_needed, CMB_unit="muK")
        cls = np.ascontiguousarray(cls)
        model_parts = []
        for cl in CL_NAMES:
            if cl not in self.use_cl:
                continue
            Dl = cls[:, self.spec_col[cl]]
            B_j = self.B[cl]
            lmax = self.Lmax_spec[cl]
            model_parts.append(B_j @ Dl[:lmax + 1])
        return np.concatenate(model_parts) if model_parts else np.array([])

    # método que genera una simulación estocástica del tipo
    # model(theta) + L_cov @ randn, simulando ruido gaussiano
    # correlacionado
    def simulate(self, theta: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        model = self.model(theta)
        noise = self.L_cov @ np.random.randn(self.n_bins_total)
        return torch.tensor(model + noise, dtype=torch.float32)

    # método que calcula el chi cuadrado entre el modelo y la observación
    # se incluye el nuisance A_planck
    def chi_squared(self, theta: torch.Tensor, A_planck: float = 1.0) -> float:
        with torch.no_grad():
            x = self.simulate(theta, seed=42)
        diff = self.X_data - x.numpy() / A_planck**2
        return float(diff @ self.invcov @ diff)

    # método que construye parámetros de CAMB desde el vector theta
    # de 6 dimensiones, theta[0]=ombh2, theta[1]=omch2, theta[2]=100*theta_mc,
    # theta[3]=tau, theta[4]=ln(10**10*As), theta[5]=ns, neutrinos fijos en
    # 0.06 eV y universo plano, usa el lmax necesario entre los espectros
    def _build_camb_params(self, theta: np.ndarray) -> "camb.CAMBparams":
        import camb

        pars = camb.CAMBparams()
        pars.set_cosmology(
            ombh2=theta[0],
            omch2=theta[1],
            cosmomc_theta=theta[2] / 100,
            mnu=0.06,
            omk=0.0,
            tau=theta[3],
        )
        pars.InitPower.set_params(As=np.exp(theta[4]) * 1e-10, ns=theta[5])
        lmax_needed = max(self.Lmax_spec.get(c, 0) for c in CL_NAMES)
        pars.set_for_lmax(lmax_needed, lens_potential_accuracy=1)
        return pars
