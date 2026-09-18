"""Minimal 1D spectrum container shared by all pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np


@dataclass
class Spectrum1D:
    """A 1D spectrum with inverse variance and a good-pixel mask.

    Attributes
    ----------
    wave : ndarray
        Wavelength in Angstrom (air, as delivered by the SALT pipeline).
    flux : ndarray
        Flux; units are tracked in ``unit`` (``"count"`` or ``"erg/s/cm2/A"``).
    ivar : ndarray
        Inverse variance of ``flux`` (0 where undefined).
    gpm : ndarray of bool
        Good-pixel mask (True = usable).
    unit : str
    meta : dict
        Free-form provenance (exptime, airmass, object, ...).
    """

    wave: np.ndarray
    flux: np.ndarray
    ivar: np.ndarray
    gpm: np.ndarray | None = None
    unit: str = "count"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.wave = np.asarray(self.wave, dtype=float)
        self.flux = np.asarray(self.flux, dtype=float)
        self.ivar = np.asarray(self.ivar, dtype=float)
        if self.gpm is None:
            self.gpm = np.isfinite(self.flux) & np.isfinite(self.ivar) & (self.ivar > 0)
        self.gpm = np.asarray(self.gpm, dtype=bool)
        n = self.wave.size
        if not (self.flux.size == self.ivar.size == self.gpm.size == n):
            raise ValueError("wave, flux, ivar and gpm must have the same length")

    @property
    def err(self) -> np.ndarray:
        """1-sigma uncertainty (inf where ivar == 0)."""
        with np.errstate(divide="ignore"):
            return np.where(self.ivar > 0, 1.0 / np.sqrt(self.ivar), np.inf)

    def scaled(self, factor: np.ndarray | float, unit: str | None = None, **meta) -> "Spectrum1D":
        """Return a copy multiplied by ``factor`` (ivar divided by factor**2)."""
        factor = np.asarray(factor, dtype=float)
        good = np.isfinite(factor) & (factor != 0)
        flux = np.where(good, self.flux * np.where(good, factor, 1.0), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ivar = np.where(good, self.ivar / np.where(good, factor, 1.0) ** 2, 0.0)
        return replace(self, flux=flux, ivar=ivar, gpm=self.gpm & good,
                       unit=unit or self.unit, meta={**self.meta, **meta})

    def to_table(self, nan_bad: bool = True):
        """Return an `astropy.table.Table` with wavelength, flux, fluxerr (and mask) columns.

        With ``nan_bad`` the flux and error of bad pixels are written as NaN.
        """
        from astropy.table import Table

        flux = self.flux.astype(float).copy()
        err = self.err.astype(float).copy()
        err[~np.isfinite(err)] = np.nan
        if nan_bad:
            flux[~self.gpm] = np.nan
            err[~self.gpm] = np.nan
        return Table({"wavelength": self.wave, "flux": flux, "fluxerr": err,
                      "mask": (~self.gpm).astype(np.int8)})

    def write_csv(self, path, include_mask: bool = False, nan_bad: bool = True) -> None:
        """Write ``wavelength,flux,fluxerr`` (plus ``mask`` if requested) as CSV."""
        tab = self.to_table(nan_bad=nan_bad)
        if not include_mask:
            tab.remove_column("mask")
        tab.write(path, format="ascii.csv", overwrite=True)
