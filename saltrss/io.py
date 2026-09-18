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
