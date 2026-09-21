"""FITS output with the full set of intermediate spectra."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

from . import __version__


def write_fits_result(result, path: Path | str) -> None:
    """Multi-extension FITS: SPECTRUM (final), COUNTS, FLUXED, TELLURIC, SKY, SENSFUNC."""
    info = result.info
    hdr = fits.Header()
    hdr["OBJECT"] = info.object
    hdr["DATE-OBS"] = info.date_obs
    if info.jd:
        hdr["MJD-OBS"] = (info.jd - 2400000.5, "MJD at the start of the exposure (UTC)")
    hdr["EXPTIME"] = info.exptime
    hdr["AIRMASS"] = info.airmass
    hdr["RA_DEG"] = info.ra_deg
    hdr["DEC_DEG"] = info.dec_deg
    hdr["RSSCONF"] = info.config.key
    hdr["SRCFILE"] = info.path.name
    hdr["SALTRSS"] = __version__
    hdr["WAVEAIR"] = (True, "wavelengths are air, Angstrom")
    hdr["FLUXUNIT"] = result.spectrum.unit
    if result.sensfunc is not None:
        hdr["STDNAME"] = str(result.sensfunc.meta.get("standard", ""))
        hdr["STDDATE"] = str(result.sensfunc.meta.get("std_date", ""))
    if result.telluric is not None:
        hdr["TELLURIC"] = result.telluric.method
    if result.ebv is not None:
        hdr["EBV"] = result.ebv
    hdr["TRACEROW"] = float(result.extraction.trace.center)
    hdr["FWHMPIX"] = float(result.extraction.trace.fwhm)

    def tab(spec, extra=None):
        t = spec.to_table(nan_bad=False)
        if extra:
            for k, v in extra.items():
                t[k] = v
        return t

    hdus = [fits.PrimaryHDU(header=hdr),
            fits.BinTableHDU(tab(result.spectrum), name="SPECTRUM"),
            fits.BinTableHDU(tab(result.counts), name="COUNTS"),
            fits.BinTableHDU(tab(result.fluxed), name="FLUXED")]
    if result.telluric is not None:
        hdus.append(fits.BinTableHDU(Table({"wavelength": result.spectrum.wave,
                                            "transmission": result.telluric.transmission}), name="TELLURIC"))
    if result.extraction.sky is not None:
        hdus.append(fits.BinTableHDU(tab(result.extraction.sky), name="SKY"))
    if result.sensfunc is not None:
        s = result.sensfunc
        hdus.append(fits.BinTableHDU(Table({"wave": s.wave, "zeropoint": s.zeropoint, "gpm": s.gpm.astype(np.int8)}),
                                     name="SENSFUNC"))
    fits.HDUList(hdus).writeto(path, overwrite=True)


def read_spectrum_product(path: Path | str):
    """Read back a saltrss product written by :func:`write_fits_result` (or a CSV).

    Returns ``(Spectrum1D, header)``; ``header`` is None for a CSV.  Used to
    re-upload an already reduced spectrum without redoing the reduction.
    """
    from .spectrum import Spectrum1D

    path = Path(path)
    if path.suffix.lower() in (".csv", ".txt", ".dat"):
        tab = Table.read(path, format="ascii.csv")
        cols = {c.lower(): c for c in tab.colnames}
        wave = np.asarray(tab[cols["wavelength"]], dtype=float)
        flux = np.asarray(tab[cols["flux"]], dtype=float)
        err = np.asarray(tab[cols["fluxerr"]], dtype=float) if "fluxerr" in cols else np.full_like(flux, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            ivar = np.where(np.isfinite(err) & (err > 0), 1.0 / err ** 2, 0.0)
        gpm = np.isfinite(flux)
        if "mask" in cols:
            gpm &= np.asarray(tab[cols["mask"]], dtype=int) == 0
        return Spectrum1D(wave, np.nan_to_num(flux), ivar, gpm, unit="erg/s/cm2/A"), None
    with fits.open(path) as hdul:
        hdr = hdul[0].header
        tab = Table(hdul["SPECTRUM"].data)
        wave = np.asarray(tab["wavelength"], dtype=float)
        flux = np.asarray(tab["flux"], dtype=float)
        err = np.asarray(tab["fluxerr"], dtype=float)
        gpm = np.asarray(tab["mask"], dtype=int) == 0 if "mask" in tab.colnames else np.isfinite(flux)
        with np.errstate(divide="ignore", invalid="ignore"):
            ivar = np.where(np.isfinite(err) & (err > 0), 1.0 / err ** 2, 0.0)
        return Spectrum1D(wave, np.nan_to_num(flux), ivar, gpm & np.isfinite(flux),
                          unit=str(hdr.get("FLUXUNIT", "erg/s/cm2/A")).strip()), hdr
