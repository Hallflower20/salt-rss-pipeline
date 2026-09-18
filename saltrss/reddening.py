"""Milky Way reddening: E(B-V) from `dustmaps`, extinction law from `extinction`."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .spectrum import Spectrum1D

SF11_RECALIBRATION = 0.86  # Schlafly & Finkbeiner 2011 rescaling of SFD E(B-V)


def _configure_dustmaps(data_dir: str | Path | None = None) -> None:
    from dustmaps.config import config

    data_dir = data_dir or os.environ.get("SALTRSS_DUSTMAPS_DIR") or os.environ.get("DUSTMAPS_DATA_DIR")
    if data_dir:
        config["data_dir"] = str(Path(data_dir).expanduser())


def ebv_from_dustmaps(ra_deg: float, dec_deg: float, dustmap: str = "sfd", recalibrate: bool = True,
                      data_dir: str | Path | None = None, fetch: bool = True) -> float:
    """Line-of-sight E(B-V) (mag).

    Parameters
    ----------
    dustmap : {"sfd", "planck", "csfd"}
        SFD98 (default; multiplied by 0.86 when ``recalibrate``), Planck 2013
        tau-based E(B-V), or Chiang 2023 corrected SFD.
    fetch : bool
        Download the map into the dustmaps data directory if missing.
    """
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    _configure_dustmaps(data_dir)
    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    dustmap = dustmap.lower()
    if dustmap == "sfd":
        import dustmaps.sfd as mod
        Query = mod.SFDQuery
        factor = SF11_RECALIBRATION if recalibrate else 1.0
    elif dustmap == "planck":
        import dustmaps.planck as mod
        Query = mod.PlanckQuery
        factor = 1.0
    elif dustmap == "csfd":
        import dustmaps.csfd as mod
        Query = mod.CSFDQuery
        factor = 1.0
    else:
        raise ValueError(f"unknown dust map {dustmap!r}")
    try:
        q = Query()
    except Exception:
        if not fetch:
            raise
        mod.fetch()
        q = Query()
    return float(q(coord)) * factor


def extinction_mag(wave: np.ndarray, ebv: float, r_v: float = 3.1, law: str = "fitzpatrick99") -> np.ndarray:
    """A(lambda) in magnitudes for the given law (``extinction`` package)."""
    import extinction as _ext

    fn = getattr(_ext, law)
    a_v = r_v * ebv
    wave = np.ascontiguousarray(wave, dtype=np.float64)
    if law in ("fitzpatrick99", "ccm89", "odonnell94", "fm07"):
        return fn(wave, a_v, r_v) if law != "fm07" else fn(wave, a_v)
    return fn(wave, a_v, r_v)


def deredden(spec: Spectrum1D, ebv: float, r_v: float = 3.1, law: str = "fitzpatrick99") -> Spectrum1D:
    """Remove Milky Way extinction from a spectrum."""
    if ebv == 0:
        return spec.scaled(1.0, ebv=0.0)
    a_lam = extinction_mag(spec.wave, ebv, r_v=r_v, law=law)
    return spec.scaled(10 ** (0.4 * a_lam), ebv=float(ebv), r_v=r_v, extinction_law=law)
