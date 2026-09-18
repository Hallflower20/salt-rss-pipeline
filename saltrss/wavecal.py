"""Wavelength solutions from the SALT pipeline ``wavelength_solutions_nm_spec_ls_img.json``.

Each entry of ``solutions`` holds the fit for one configuration + block visit::

    {"reference": "mbxgpP202510260084", "lamp": "Ar",
     "fit": {"function": "legendre", "order": 5, "niter": 5},
     "pm": [...pixel centroids of identified lines...],
     "wm": [...their wavelengths (Angstrom)...],
     "w_coef": [6 Legendre coefficients], "w_rms": 0.25,
     "w_min": 3640.57, "w_max": 6732.15,
     "z_coef": [3 power-series coefficients], "z_rms": 0.12, "n": 3172}

Conventions (verified against the pipeline's own ``*_wr.fits`` rectified arcs
to < 0.01 pixel):

* ``w_coef`` is a Legendre series in ``x' = 2 (x - min(pm)) / (max(pm) - min(pm)) - 1``
  where ``x`` is the 0-based column of the mosaicked frame; it describes the
  *central* row.  ``w_min``/``w_max`` are the wavelengths at columns 0 and n-1.
* The rectified frame is sampled on ``linspace(w_min, w_max, n)``.
* ``z_coef`` is an ordinary power series in the 0-based row index giving the
  wavelength zero point of each row; row ``r`` has
  ``wave_r(x) = wave_c(x) + z(r) - z(r_ref)`` with ``r_ref = ny // 2``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.polynomial import legendre as _L
from numpy.polynomial import polynomial as _P
from scipy.optimize import fsolve

from .products import RSSConfig


@dataclass
class WavelengthSolution:
    """Pipeline wavelength solution for one configuration."""

    w_coef: np.ndarray
    z_coef: np.ndarray
    x_domain: tuple[float, float]
    w_min: float
    w_max: float
    n: int
    key: str = ""
    reference: str = ""
    lamp: str = ""
    w_rms: float = float("nan")
    z_rms: float = float("nan")

    # ---------------------------------------------------------- construction
    @classmethod
    def from_json_entry(cls, key: str, entry: Mapping, tol: float = 0.05) -> "WavelengthSolution":
        w_coef = np.asarray(entry["w_coef"], dtype=float)
        z_coef = np.asarray(entry.get("z_coef", [0.0]), dtype=float)
        pm = np.asarray(entry.get("pm", []), dtype=float)
        n = int(entry["n"])
        w_min, w_max = float(entry["w_min"]), float(entry["w_max"])
        dom = (float(pm.min()), float(pm.max())) if pm.size >= 2 else None
        sol = cls(w_coef=w_coef, z_coef=z_coef, x_domain=dom or (0.0, n - 1.0), w_min=w_min,
                  w_max=w_max, n=n, key=key, reference=entry.get("reference", ""),
                  lamp=entry.get("lamp", ""), w_rms=float(entry.get("w_rms", np.nan)),
                  z_rms=float(entry.get("z_rms", np.nan)))
        if not sol.domain_is_consistent(tol):
            sol.x_domain = sol._solve_domain()
            if not sol.domain_is_consistent(tol):
                raise ValueError(f"Could not reproduce w_min/w_max for solution {key}")
        return sol

    def domain_is_consistent(self, tol: float = 0.05) -> bool:
        return (abs(self.wave_at(0.0) - self.w_min) < tol
                and abs(self.wave_at(self.n - 1.0) - self.w_max) < tol)

    def _solve_domain(self) -> tuple[float, float]:
        """Fallback: solve for the Legendre domain from the two end-point constraints."""
        def resid(p):
            x0, x1 = p
            xn = lambda x: 2 * (x - x0) / (x1 - x0) - 1  # noqa: E731
            return [_L.legval(xn(0.0), self.w_coef) - self.w_min,
                    _L.legval(xn(self.n - 1.0), self.w_coef) - self.w_max]
        x0, x1 = fsolve(resid, [0.05 * self.n, 0.98 * self.n])
        return float(x0), float(x1)

    # ------------------------------------------------------------- evaluate
    @property
    def config(self) -> RSSConfig:
        return RSSConfig.from_key(self.key)

    def _xnorm(self, x):
        x0, x1 = self.x_domain
        return 2.0 * (np.asarray(x, dtype=float) - x0) / (x1 - x0) - 1.0

    def wave_at(self, x, row=None, ref_row=None):
        """Wavelength (Angstrom) at column(s) ``x``; optionally for a given row."""
        w = _L.legval(self._xnorm(x), self.w_coef)
        if row is not None:
            if ref_row is None:
                raise ValueError("ref_row is required when row is given")
            w = w + self.row_shift(row, ref_row)
        return w

    def row_shift(self, row, ref_row) -> np.ndarray:
        """Wavelength offset of ``row`` relative to ``ref_row`` (Angstrom)."""
        return _P.polyval(np.asarray(row, dtype=float), self.z_coef) - _P.polyval(float(ref_row), self.z_coef)

    def linear_grid(self) -> np.ndarray:
        """The wavelength grid of a rectified frame."""
        return np.linspace(self.w_min, self.w_max, self.n)

    @property
    def dispersion(self) -> float:
        return (self.w_max - self.w_min) / (self.n - 1)

    # -------------------------------------------------------------- rectify
    def rectify(self, image: np.ndarray, ref_row: int | None = None, var: np.ndarray | None = None,
                mask: np.ndarray | None = None, row_scale: float = 1.0):
        """Resample every row of ``image`` onto :meth:`linear_grid`.

        Mirrors the pipeline's ``_wr`` step (linear interpolation per row).
        ``row_scale`` converts the image's row index to the row units of the
        solution (e.g. 0.5 for a 2x2-binned frame and a solution from a 2x4
        binned frame), so the slow row-shift polynomial can be reused across
        spatial binnings.

        Returns
        -------
        rect : ndarray            rectified image (ny, n)
        rect_var : ndarray|None   rectified variance (same interpolation)
        rect_mask : ndarray|None  boolean bad-pixel mask, True where any
                                  contributing input pixel was bad or outside the frame
        """
        image = np.asarray(image, dtype=float)
        ny, nx = image.shape
        if ref_row is None:
            ref_row = ny // 2
        grid = self.linear_grid()
        x = np.arange(nx, dtype=float)
        wave_c = self.wave_at(x)
        rect = np.empty((ny, grid.size), dtype=float)
        rect_var = np.empty_like(rect) if var is not None else None
        badf = None
        if mask is not None:
            badf = np.asarray(mask, dtype=float)
        rect_mask = np.zeros_like(rect, dtype=bool) if mask is not None else None
        shifts = self.row_shift(np.arange(ny) * row_scale, ref_row * row_scale)
        for r in range(ny):
            w = wave_c + shifts[r]
            rect[r] = np.interp(grid, w, image[r], left=np.nan, right=np.nan)
            if rect_var is not None:
                rect_var[r] = np.interp(grid, w, var[r], left=np.nan, right=np.nan)
            if rect_mask is not None:
                rect_mask[r] = np.interp(grid, w, badf[r], left=1.0, right=1.0) > 0
        outside = ~np.isfinite(rect)
        if rect_mask is not None:
            rect_mask |= outside
        rect[outside] = 0.0
        if rect_var is not None:
            rect_var[outside] = np.inf
        return rect, rect_var, rect_mask


def load_solutions(json_dict: Mapping) -> dict[str, WavelengthSolution]:
    """Parse the ``solutions`` of a wavelength dictionary into objects (bad entries skipped)."""
    out = {}
    for key, entry in json_dict.get("solutions", {}).items():
        try:
            out[key] = WavelengthSolution.from_json_entry(key, entry)
        except (KeyError, ValueError):
            continue
    return out


def find_solution(solutions: Mapping[str, WavelengthSolution], config: RSSConfig,
                  angle_tol: float = 0.05, bvisit: str | None = None,
                  require_ybin: bool = True) -> WavelengthSolution:
    """Pick the solution matching ``config`` (same block visit, same spatial binning,
    then lowest rms).  With ``require_ybin=False`` a solution from another spatial
    binning is accepted (use ``row_scale`` when rectifying)."""
    cands = []
    for key, sol in solutions.items():
        try:
            cfg = RSSConfig.from_key(key)
        except ValueError:
            continue
        if cfg.matches(config, angle_tol=angle_tol, require_ybin=require_ybin):
            cands.append((0 if (bvisit and key.endswith(f"BV{bvisit}")) else 1,
                          0 if cfg.ybin == config.ybin else 1,
                          np.nan_to_num(sol.w_rms, nan=1e9), key, sol))
    if not cands:
        raise LookupError(f"No wavelength solution for configuration {config.key} "
                          f"(available: {sorted(solutions)})")
    cands.sort(key=lambda t: t[:4])
    return cands[0][4]


def wcs_grid(header) -> np.ndarray:
    """Linear wavelength grid from a rectified frame's SCI header (CRVAL1/CDELT1/CRPIX1)."""
    n = int(header["NAXIS1"])
    crval = float(header["CRVAL1"])
    cdelt = float(header.get("CDELT1", header.get("CD1_1")))
    crpix = float(header.get("CRPIX1", 1.0))
    return crval + (np.arange(n) + 1 - crpix) * cdelt


_BV_RE = re.compile(r"BV(\d+)$")


def bvisit_of(key: str) -> str | None:
    m = _BV_RE.search(key)
    return m.group(1) if m else None
