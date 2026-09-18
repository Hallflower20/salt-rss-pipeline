"""Telluric absorption correction of a flux-calibrated spectrum.

Default method (``"pypeit"``): fit pypeit's PCA telluric model
(`pypeit.core.telluric.Telluric`, grid ``TellPCA_3000_26000_R10000.fits``,
downloaded once into the pypeit cache) times a low-order polynomial object
model, using only the pixels in and around the telluric bands.  The fitted
transmission is then divided out over the full spectrum.  This works for any
site because the PCA spans the atmospheric parameter space.

Alternative (``"skycalc"``): ESO SkyCalc transmission at the observed airmass
(requires network and `skycalc_ipy`), smoothed to the instrument resolution,
with the band strength tuned by a single power on the O2 bands.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .spectrum import Spectrum1D

DEFAULT_TELGRID = "TellPCA_3000_26000_R10000.fits"

# Telluric bands (air wavelengths, Angstrom) in the optical/near-IR.
TELLURIC_BANDS = [(6270.0, 6330.0),   # O2 gamma
                  (6860.0, 6960.0),   # O2 B
                  (7150.0, 7350.0),   # H2O
                  (7580.0, 7720.0),   # O2 A
                  (8100.0, 8400.0),   # H2O
                  (8900.0, 9800.0)]   # H2O


@dataclass
class TelluricResult:
    spectrum: Spectrum1D
    transmission: np.ndarray
    method: str
    meta: dict[str, Any] = field(default_factory=dict)


def in_bands(wave: np.ndarray, bands=TELLURIC_BANDS, pad: float = 0.0) -> np.ndarray:
    m = np.zeros(wave.shape, dtype=bool)
    for lo, hi in bands:
        m |= (wave >= lo - pad) & (wave <= hi + pad)
    return m


def covered_bands(wave: np.ndarray, gpm: np.ndarray, bands=TELLURIC_BANDS, min_frac: float = 0.8):
    """Bands fully (>= min_frac) inside the good wavelength range."""
    good_w = wave[gpm]
    if good_w.size == 0:
        return []
    lo_w, hi_w = good_w.min(), good_w.max()
    out = []
    for lo, hi in bands:
        frac = (min(hi, hi_w) - max(lo, lo_w)) / (hi - lo)
        if frac >= min_frac:
            out.append((lo, hi))
    return out


def apply_transmission(spec: Spectrum1D, trans: np.ndarray, method: str, **meta) -> Spectrum1D:
    trans = np.asarray(trans, dtype=float)
    ok = np.isfinite(trans) & (trans > 0.05)
    factor = np.where(ok, 1.0 / np.where(ok, trans, 1.0), np.nan)
    out = spec.scaled(factor, telluric=method, **meta)
    out.gpm &= ok
    return out


# ------------------------------------------------------------------ pypeit
def correct_pypeit(spec: Spectrum1D, airmass: float | None = None, resolution: float | None = None,
                   telgridfile: str = DEFAULT_TELGRID, polyorder: int = 3, continuum_pad: float = 150.0,
                   bands=TELLURIC_BANDS, tell_npca: int = 4, maxiter: int = 2, popsize: int = 20,
                   tol: float = 1e-3, seed: int = 777, debug: bool = False) -> TelluricResult:
    """Fit and remove telluric absorption with pypeit's PCA model."""
    from pypeit.core import telluric as _tel
    from pypeit.core.wave import airtovac
    from astropy import units as u

    airmass = float(spec.meta.get("airmass", 1.2) if airmass is None else airmass)
    exptime = float(spec.meta.get("exptime", 1.0))
    bands_here = covered_bands(spec.wave, spec.gpm, bands)
    if not bands_here:
        return TelluricResult(spec, np.ones_like(spec.wave), "none",
                              {"reason": "no telluric band inside the wavelength coverage"})
    fit_gpm = spec.gpm & in_bands(spec.wave, bands_here, pad=continuum_pad) & np.isfinite(spec.flux)
    if fit_gpm.sum() < 50:
        return TelluricResult(spec, np.ones_like(spec.wave), "none", {"reason": "too few pixels in bands"})

    # pypeit's model grid is in vacuum; the SALT solution is in air.
    wave_vac = airtovac(spec.wave * u.AA).value
    # Work in pypeit's 1e-17 units so polynomial coefficients are O(1).
    scale = 1e17 if spec.unit.startswith("erg") else 1.0
    finite = np.isfinite(spec.flux) & np.isfinite(spec.ivar)
    flux = np.where(finite, spec.flux, 0.0) * scale
    ivar = np.where(finite, spec.ivar, 0.0) / scale**2
    fit_gpm &= finite
    if resolution is None:
        resolution = float(spec.meta.get("resolution", 1000.0))

    obj_params = dict(z_obj=0.0, mask_lyman_a=False, airmass=airmass,
                      delta_coeff_bounds=(-20.0, 20.0), minmax_coeff_bounds=(-5.0, 5.0),
                      polyorder_vec=np.array([polyorder]), exptime=exptime, func="legendre",
                      model="exp", sigrej=3.0,
                      output_meta_keys=("airmass", "polyorder_vec", "exptime", "func"), debug=debug)
    # pypeit convolves the model over the full input wavelength range, so only hand it the
    # stretch of spectrum that is actually fitted (bands + continuum pads).
    idx = np.where(fit_gpm)[0]
    i0, i1 = int(idx.min()), int(idx.max()) + 1
    sl = slice(i0, i1)
    tel = _tel.Telluric(wave_vac[sl], flux[sl], ivar[sl], fit_gpm[sl], telgridfile, obj_params,
                        _tel.init_poly_model, _tel.eval_poly_model, airmass_guess=airmass,
                        resln_guess=resolution, resln_frac_bounds=(0.5, 1.5),
                        pix_shift_bounds=(-5.0, 5.0), sn_clip=30.0, maxiter=maxiter, tol=tol,
                        popsize=popsize, teltype="pca", tell_npca=tell_npca, seed=seed, debug=debug)
    tel.run()
    trans = np.ones_like(spec.wave)
    trans_fit = np.asarray(tel.model["TELLURIC"][0], dtype=float)
    # Outside the fitted range the model is 0 -> no correction.
    trans[sl] = np.where((trans_fit > 0) & np.isfinite(trans_fit), trans_fit, 1.0)
    meta = {"telluric_bands": bands_here, "tell_chi2": float(tel.model["CHI2"][0]),
            "tell_success": bool(tel.model["SUCCESS"][0]), "tell_resln": float(tel.model["TELL_RESLN"][0]),
            "tell_shift_pix": float(tel.model["TELL_SHIFT"][0]), "telgridfile": telgridfile}
    out = apply_transmission(spec, trans, "pypeit", **meta)
    return TelluricResult(out, trans, "pypeit", meta)


# ------------------------------------------------------------------ skycalc
def skycalc_transmission(wave_air: np.ndarray, airmass: float, pwv: float = 2.5,
                         observatory: str = "lasilla") -> np.ndarray:
    """ESO SkyCalc atmospheric transmission on ``wave_air`` (network access required)."""
    import skycalc_ipy
    from pypeit.core.wave import vactoair
    from astropy import units as u

    allowed_pwv = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.5, 3.5, 5.0, 7.5, 10.0, 20.0, 30.0])
    sc = skycalc_ipy.SkyCalc()
    sc["observatory"] = observatory
    sc["airmass"] = float(np.clip(airmass, 1.0, 3.0))
    sc["pwv_mode"] = "pwv"
    sc["pwv"] = float(allowed_pwv[np.argmin(np.abs(allowed_pwv - pwv))])
    sc["wmin"] = float(wave_air.min() / 10 - 5)
    sc["wmax"] = float(wave_air.max() / 10 + 5)
    sc["wgrid_mode"] = "fixed_wavelength_step"
    sc["wdelta"] = 0.02
    sc["lsf_type"] = "none"
    tab = sc.get_sky_spectrum(return_type="table")
    lam_nm = np.asarray(tab["lam"].value if hasattr(tab["lam"], "value") else tab["lam"], dtype=float)
    w_air = vactoair(lam_nm * 10 * u.AA).value
    trans = np.asarray(tab["trans"], dtype=float)
    return np.interp(wave_air, w_air, trans)


def correct_skycalc(spec: Spectrum1D, airmass: float | None = None, resolution: float | None = None,
                    pwv: float = 2.5, bands=TELLURIC_BANDS, fit_scale: bool = True) -> TelluricResult:
    """Divide by a resolution-matched SkyCalc transmission, scaled as T**alpha
    where alpha is fit on the covered O2/H2O bands (Beer-Lambert scaling)."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import minimize_scalar

    airmass = float(spec.meta.get("airmass", 1.2) if airmass is None else airmass)
    bands_here = covered_bands(spec.wave, spec.gpm, bands)
    if not bands_here:
        return TelluricResult(spec, np.ones_like(spec.wave), "none",
                              {"reason": "no telluric band inside the wavelength coverage"})
    if resolution is None:
        resolution = float(spec.meta.get("resolution", 1000.0))
    disp = float(np.median(np.diff(spec.wave)))
    trans = skycalc_transmission(spec.wave, airmass, pwv=pwv)
    sigma_pix = (np.median(spec.wave) / resolution / 2.3548) / disp
    trans_s = gaussian_filter1d(trans, max(sigma_pix, 0.3))
    alpha = 1.0
    if fit_scale:
        m = spec.gpm & in_bands(spec.wave, bands_here, pad=100.0) & (trans_s > 0.05)

        def cost(a):
            corr = spec.flux[m] / trans_s[m] ** a
            # roughness of the corrected spectrum inside the bands
            d = np.diff(corr) / (np.abs(corr[:-1]) + 1e-30)
            return float(np.nanmedian(np.abs(d)))
        res = minimize_scalar(cost, bounds=(0.3, 3.0), method="bounded")
        alpha = float(res.x)
    tr = trans_s ** alpha
    out = apply_transmission(spec, tr, "skycalc", telluric_bands=bands_here, skycalc_alpha=alpha, pwv=pwv)
    return TelluricResult(out, tr, "skycalc", {"alpha": alpha, "pwv": pwv, "telluric_bands": bands_here})


def correct(spec: Spectrum1D, method: str = "pypeit", **kwargs) -> TelluricResult:
    """Dispatch: ``method`` in {"pypeit", "skycalc", "none"}."""
    if method in (None, "none", False):
        return TelluricResult(spec, np.ones_like(spec.wave), "none", {})
    if method == "pypeit":
        return correct_pypeit(spec, **kwargs)
    if method == "skycalc":
        return correct_skycalc(spec, **kwargs)
    raise ValueError(f"unknown telluric method {method!r}")
