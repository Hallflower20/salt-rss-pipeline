"""Flux calibration against a spectrophotometric standard (pypeit under the hood).

* the reference spectrum comes from pypeit's standard-star archives
  (CALSPEC, ESO, X-shooter, NOAO, ING) matched by coordinates;
* the zeropoint (sensitivity function) is fit with
  `pypeit.core.flux_calib.fit_zeropoint` (b-spline of 2.5 log10(F_true/N_lam),
  Balmer lines and strong telluric bands masked);
* atmospheric extinction uses the Sutherland curve from PySALT
  (``saltrss/data/suth_extinct.dat``) via `pypeit.core.atmextinction`.

Caveat: SALT's moving pupil means absolute spectrophotometry is uncertain at
the tens-of-percent level; the *shape* of the calibration is what this gives you.
"""
from __future__ import annotations

import importlib.resources as _res
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.table import Table

from .products import FrameInfo, RSSConfig, frame_is_readable, scan_frames
from .spectrum import Spectrum1D

FLUX_SCALE = 1e-17  # pypeit's internal F_lambda unit, erg/s/cm^2/A
UNBINNED_PIXSCALE = 0.1267  # arcsec / unbinned pixel (RSS)


# ------------------------------------------------------------------ extinction
def sutherland_extinction():
    """`pypeit.core.atmextinction.AtmosphericExtinction` for Sutherland (PySALT table)."""
    from pypeit.core.atmextinction import AtmosphericExtinction

    with _res.files("saltrss.data").joinpath("suth_extinct.dat").open() as fh:
        tab = np.loadtxt(fh, comments="#")
    return AtmosphericExtinction(tab[:, 0], tab[:, 1], file="suth_extinct.dat")


def extinction_curve(name: str | None = None):
    """Extinction curve by name: ``None``/``"sutherland"`` (default) or any pypeit
    file such as ``"ctioextinct.dat"``."""
    if name in (None, "sutherland", "suth", "suth_extinct.dat"):
        return sutherland_extinction()
    from pypeit.core.atmextinction import AtmosphericExtinction

    return AtmosphericExtinction.from_file(name)


# ------------------------------------------------------------------ resolution
def spectral_resolution(info: FrameInfo, dispersion: float, fwhm_pix: float | None = None,
                        wave_ref: float | None = None) -> float:
    """Rough R = lambda / FWHM for a frame from the slit width (or seeing if smaller)."""
    slit = info.slit_width_arcsec or 1.5
    slit_pix = slit / (UNBINNED_PIXSCALE * info.config.xbin)
    fwhm = slit_pix if fwhm_pix is None else min(slit_pix, fwhm_pix)
    fwhm = max(fwhm, 2.0)
    if wave_ref is None:
        wave_ref = 5500.0
    return float(wave_ref / (fwhm * dispersion))


# --------------------------------------------------------------- standards
def find_standard_frames(standards_dir: Path | str, config: RSSConfig, angle_tol: float = 0.05,
                         require_ybin: bool = False) -> list[FrameInfo]:
    """All ``mbxgpP*.fits`` basic products under ``standards_dir`` whose
    configuration matches ``config`` (recursive search)."""
    standards_dir = Path(standards_dir)
    files = [p for p in standards_dir.rglob("mbxgpP*.fits") if "Flat" not in p.name and "_" not in p.stem[5:]]
    out = []
    for fi in scan_frames(files):
        if fi.obstype.upper() != "OBJECT" or fi.is_arc:
            continue
        if fi.config.matches(config, angle_tol=angle_tol, require_ybin=require_ybin):
            out.append(fi)
    return out


def choose_standard(cands: list[FrameInfo], jd: float, ra_deg: float | None = None,
                    dec_deg: float | None = None) -> FrameInfo:
    """Pick the standard closest in time that has an archive spectrum in pypeit."""
    from pypeit.core import standard as _std

    cands = sorted(cands, key=lambda f: abs(f.jd - jd))
    for fi in cands:
        if not frame_is_readable(fi.path):
            continue  # truncated/corrupt file
        try:
            _std.get_standard_spectrum(ra=fi.ra_deg, dec=fi.dec_deg)
            return fi
        except Exception:
            continue
    raise LookupError("None of the candidate standards is readable and has a reference spectrum in pypeit's archives")


def reference_spectrum(info: FrameInfo):
    """pypeit `Spectrum` of the standard (flux in 1e-17 erg/s/cm^2/A)."""
    from pypeit.core import standard as _std

    return _std.get_standard_spectrum(ra=info.ra_deg, dec=info.dec_deg)


# ------------------------------------------------------------------ sensfunc
@dataclass
class SensFunc:
    """Zeropoint ZP(lambda) such that F_lambda[1e-17 cgs] = N_lambda * 10^(-0.4 (ZP - ZP0))."""

    wave: np.ndarray
    zeropoint: np.ndarray
    gpm: np.ndarray
    zeropoint_data: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def wave_range(self) -> tuple[float, float]:
        w = self.wave[self.gpm]
        return float(w.min()), float(w.max())

    def factor(self, wave: np.ndarray, exptime: float, airmass: float, atmext=None,
               extrapolate: bool = True) -> np.ndarray:
        """Multiplicative factor turning counts/pixel into erg/s/cm^2/A."""
        from pypeit.core.flux_calib import get_sensfunc_factor

        if atmext is None:
            atmext = extinction_curve(self.meta.get("extinction"))
        f = get_sensfunc_factor(wave, self.wave[self.gpm], self.zeropoint[self.gpm], exptime,
                                atmext=atmext, airmass=airmass, extrap_sens=extrapolate)
        return np.asarray(f, dtype=float) * FLUX_SCALE

    def apply(self, spec: Spectrum1D, atmext=None) -> Spectrum1D:
        exptime = float(spec.meta["exptime"])
        airmass = float(spec.meta["airmass"])
        f = self.factor(spec.wave, exptime, airmass, atmext=atmext)
        inrange = (spec.wave >= self.wave_range[0]) & (spec.wave <= self.wave_range[1])
        out = spec.scaled(f, unit="erg/s/cm2/A", sensfunc=self.meta.get("standard"),
                          fluxcal_extrapolated=bool((~inrange & spec.gpm).any()))
        out.meta["sensfunc_range"] = self.wave_range
        return out

    # persistence ------------------------------------------------------
    def write(self, path: Path | str) -> None:
        tab = Table({"wave": self.wave, "zeropoint": self.zeropoint, "gpm": self.gpm.astype(np.int8),
                     "zeropoint_data": self.zeropoint_data if self.zeropoint_data is not None
                     else np.full_like(self.wave, np.nan)})
        hdr = fits.Header()
        for k, v in self.meta.items():
            if isinstance(v, (str, int, float, bool)) and len(str(v)) < 68:
                hdr[f"HIERARCH SALTRSS {k.upper()}"] = v
        hdu = fits.BinTableHDU(tab, name="SENSFUNC", header=hdr)
        fits.HDUList([fits.PrimaryHDU(header=hdr), hdu]).writeto(path, overwrite=True)

    @classmethod
    def read(cls, path: Path | str) -> "SensFunc":
        with fits.open(path) as hdul:
            tab = Table(hdul["SENSFUNC"].data)
            meta = {k.split()[-1].lower(): v for k, v in hdul[0].header.items() if k.startswith("SALTRSS")}
        zd = np.asarray(tab["zeropoint_data"], dtype=float)
        return cls(np.asarray(tab["wave"], float), np.asarray(tab["zeropoint"], float),
                   np.asarray(tab["gpm"]).astype(bool), None if np.all(np.isnan(zd)) else zd, meta)


def build_sensfunc(std_spec: Spectrum1D, std_info: FrameInfo, resolution: float | None = None,
                   extinction: str | None = None, hydrogen_mask_wid: float = 25.0,
                   polyorder: int = 5, nresln: float = 20.0, trans_thresh: float = 0.9,
                   reference=None, min_counts_snr: float = 3.0) -> SensFunc:
    """Fit the zeropoint from an extracted standard-star counts spectrum.

    Parameters
    ----------
    std_spec : Spectrum1D
        Extracted standard in counts (``meta`` must contain exptime, airmass).
    std_info : FrameInfo
        Header info of the standard (coordinates for the archive lookup).
    resolution : float, optional
        R of the standard observation (defaults from slit/seeing).
    extinction : str, optional
        Extinction curve name (see :func:`extinction_curve`).
    hydrogen_mask_wid : float
        Half-width (A) of the Balmer/Paschen line masks; wide slits need ~25 A.
    reference : pypeit Spectrum, optional
        Reference spectrum (else looked up by coordinates).
    """
    from pypeit.core.flux_calib import counts2Nlam, fit_zeropoint

    atmext = extinction_curve(extinction)
    if reference is None:
        reference = reference_spectrum(std_info)
    if resolution is None:
        disp = float(np.median(np.diff(std_spec.wave)))
        resolution = spectral_resolution(std_info, disp, std_spec.meta.get("fwhm_pix"),
                                         wave_ref=float(np.median(std_spec.wave)))
    exptime, airmass = float(std_spec.meta["exptime"]), float(std_spec.meta["airmass"])
    gpm = std_spec.gpm & (std_spec.flux * np.sqrt(std_spec.ivar) > min_counts_snr)
    gpm &= (std_spec.wave > reference.wave.min()) & (std_spec.wave < reference.wave.max())
    Nlam, Nlam_ivar, gpm2 = counts2Nlam(std_spec.wave, std_spec.flux, std_spec.ivar, gpm,
                                        exptime, airmass, atmext)
    zp_data, zp_data_gpm, zp_fit, zp_fit_gpm = fit_zeropoint(
        std_spec.wave, Nlam, Nlam_ivar, gpm2, reference, mask_hydrogen_lines=True,
        mask_helium_lines=False, polyorder=polyorder, hydrogen_mask_wid=hydrogen_mask_wid,
        nresln=nresln, resolution=resolution, trans_thresh=trans_thresh, polycorrect=True,
        polyfunc=False)
    good = zp_fit_gpm & np.isfinite(zp_fit) & gpm2
    meta = {"standard": std_info.object, "std_file": str(std_info.path), "std_date": std_info.date_obs,
            "std_airmass": airmass, "std_exptime": exptime, "config": std_info.config.key,
            "reference": str(reference.meta.get("File", "")), "reference_name": str(reference.meta.get("Name", "")),
            "resolution": float(resolution), "extinction": extinction or "sutherland",
            "slit_arcsec": std_info.slit_width_arcsec or float("nan"),
            "zp_rms": float(np.nanstd((zp_data - zp_fit)[good & zp_data_gpm])) if good.any() else float("nan")}
    return SensFunc(std_spec.wave.copy(), zp_fit, good, zp_data, meta)
