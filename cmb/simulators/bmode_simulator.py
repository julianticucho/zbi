import torch
import numpy as np
import camb
from camb import model, initialpower

from zbi.simulators.base import Simulator


_lmax = 1000
_pars = camb.set_params(
    H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06,
    omk=0, tau=0.06, As=2e-9, ns=0.965,
    halofit_version='mead', lmax=_lmax,
)
_pars.set_for_lmax(_lmax, lens_potential_accuracy=2)
_pars.WantTensors = True

# espectro de lente (r=0)
_pars_temp = _pars.copy()
_results = camb.get_results(_pars_temp)
_inflation_params = initialpower.InitialPowerLaw()
_inflation_params.set_params(ns=0.965, r=0.0)
_results.power_spectra_from_transfer(_inflation_params)
_cl_lensed = _results.get_total_cls(_lmax, CMB_unit='muK')
_cl_unlensed = _results.get_unlensed_scalar_cls(_lmax, CMB_unit='muK')
_bb_lensing = _cl_lensed[:, 2] - _cl_unlensed[:, 2]

# espectro tensor (r_ref=0.1)
_r_ref = 0.1
_pars_temp2 = _pars.copy()
_results2 = camb.get_results(_pars_temp2)
_inflation_params2 = initialpower.InitialPowerLaw()
_inflation_params2.set_params(ns=0.965, r=_r_ref)
_results2.power_spectra_from_transfer(_inflation_params2)
_cl_tensor = _results2.get_tensor_cls(_lmax, CMB_unit='muK')
_bb_tensor_ref = _cl_tensor[:, 2]

_ell = np.arange(_lmax + 1)


def _make_flat_map(bb_Cl, N=256, pix_size=1.0):
    """Genera un mapa flat-sky B-mode a partir de C_ell^BB."""
    Cl = bb_Cl.copy()
    Cl[0] = 0.0
    if len(Cl) > 1:
        Cl[1] = 0.0
    inds = np.linspace(-0.5, 0.5, N)
    X, Y = np.meshgrid(inds, inds)
    R = np.sqrt(X**2 + Y**2)
    pix_to_rad = (pix_size / 60.0) * np.pi / 180.0
    ell_scale = 2.0 * np.pi / pix_to_rad
    ell2d = R * ell_scale
    Cl_expanded = np.zeros(int(ell2d.max()) + 1)
    n_copy = min(Cl.size, len(Cl_expanded))
    Cl_expanded[:n_copy] = Cl[:n_copy]
    Cl2d = Cl_expanded[ell2d.astype(int)]
    Cl2d = np.maximum(Cl2d, 0)
    random = np.random.normal(0, 1, (N, N))
    FT_random = np.fft.fft2(random)
    FT_2d = np.sqrt(Cl2d) * FT_random
    B_map = np.fft.ifft2(np.fft.fftshift(FT_2d))
    B_map = np.real(B_map) / pix_to_rad
    return B_map


class BmodeSimulator(Simulator):
    """Simulador de mapas de modos B del CMB.

    theta = (r, A_lens)
    x     = mapa B-mode de NxN (aplanado a N*N).
    """

    def __init__(self, N=256, pix_size=1.0):
        super().__init__()
        self.N = N
        self.pix_size = pix_size
        self.dim_x = N * N

    def simulate(self, theta: torch.Tensor, seed: int | None = None) -> torch.Tensor:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        r, Alens = theta[0].item(), theta[1].item()
        bb = Alens * _bb_lensing + (r / _r_ref) * _bb_tensor_ref
        bb = np.maximum(bb, 0)

        mapa = _make_flat_map(bb, self.N, self.pix_size)
        return torch.tensor(mapa.ravel(), dtype=torch.float32)


class BmodeSimulator128(BmodeSimulator):
    """Simulador de mapas B-mode de 128x128.

    Genera directamente a 128x128 con pix_size=4.0 arcmin,
    manteniendo un FOV de 512 arcmin o 8.5 grados.
    """

    def __init__(self):
        super().__init__(N=128, pix_size=4.0)
