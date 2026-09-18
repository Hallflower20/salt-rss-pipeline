"""2D frame container plus preparation of raw standard-star frames.

Science frames arrive fully processed and rectified.  Standard-star
observations from SALT's ``CAL_SPST`` programme, however, are only delivered
as basic products (bias/gain/crosstalk/mosaic).  To put a standard on the same
footing as the science frame we replay the pipeline's remaining steps using the
calibrations shipped in the science product directory:

``bp`` bad-pixel mask -> ``ag`` per-amplifier gain factors -> ``ff`` combined
flat (same configuration) -> ``cr`` L.A.Cosmic (astroscrappy) -> ``cg`` CCD-gap
mask -> ``wr`` rectification with the configuration's wavelength solution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.nddata import NDData, VarianceUncertainty

from .products import FrameInfo, ProductDir
from .wavecal import WavelengthSolution, find_solution, load_solutions, wcs_grid

DEFAULT_READNOISE = 3.0  # e- per binned pixel, RSS FAINT/SLOW is 2.5-3.5


@dataclass
class Frame2D:
    """A 2D longslit frame on a linear wavelength grid (dispersion along axis 1)."""

    data: np.ndarray
    var: np.ndarray
    mask: np.ndarray  # True = bad
    wave: np.ndarray
    info: FrameInfo
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self):
        return self.data.shape

    def cutout(self, row0: int, row1: int) -> "Frame2D":
        row0, row1 = max(int(row0), 0), min(int(row1), self.data.shape[0])
        return Frame2D(self.data[row0:row1], self.var[row0:row1], self.mask[row0:row1],
                       self.wave, self.info, {**self.meta, "row_offset": row0 + self.meta.get("row_offset", 0)})

    def to_nddata(self) -> NDData:
        var = np.where(self.mask | ~np.isfinite(self.var), np.inf, self.var)
        data = np.where(np.isfinite(self.data), self.data, 0.0)
        return NDData(data, unit=u.count, uncertainty=VarianceUncertainty(var), mask=self.mask.copy())


def estimate_variance(data: np.ndarray, readnoise: float = DEFAULT_READNOISE,
                      scale: np.ndarray | float = 1.0) -> np.ndarray:
    """Poisson + read-noise variance for gain-corrected data in electrons.

    ``scale`` is the multiplicative factor already applied to the data (e.g.
    ``1/flat``), so that ``var = (max(data/scale, 0) + rn^2) * scale^2``.
    """
    raw = np.clip(data / scale, 0, None)
    return (raw + readnoise**2) * np.asarray(scale, dtype=float) ** 2


def load_science_frame(path: Path | str, readnoise: float = DEFAULT_READNOISE) -> Frame2D:
    """Load a pipeline ``*_wr.fits`` product (SCI + optional BPM/CRMASK extensions)."""
    path = Path(path)
    with fits.open(path) as hdul:
        info = FrameInfo.from_header(hdul[0].header, path)
        sci = hdul["SCI"] if "SCI" in hdul else hdul[1]
        data = np.asarray(sci.data, dtype=float)
        wave = wcs_grid(sci.header)
        mask = ~np.isfinite(data)
        for ext in ("BPM", "CRMASK"):
            if ext in hdul and hdul[ext].data is not None:
                mask |= np.asarray(hdul[ext].data).astype(bool)
    info.shape = data.shape
    var = estimate_variance(data, readnoise)
    return Frame2D(np.nan_to_num(data), var, mask, wave, info,
                   {"source": str(path), "readnoise": readnoise})


def prepare_standard_frame(std_path: Path | str, product: ProductDir,
                           solution: WavelengthSolution | None = None,
                           apply_flat: bool = True, apply_gain: bool = False,
                           clean_cosmics: bool = False, readnoise: float = DEFAULT_READNOISE,
                           flat_floor: float = 0.1, ref_row: int | None = None) -> Frame2D:
    """Replay the pipeline steps on a basic-product standard frame and rectify it.

    Parameters
    ----------
    std_path : path
        ``mbxgpP*.fits`` standard-star frame (SCI in extension 1).
    product : ProductDir
        Science product directory supplying flat, BPM, gaps, gain factors and
        the wavelength solution for the *same* configuration.
    solution : WavelengthSolution, optional
        Explicit solution; otherwise matched to the standard's configuration
        from the product directory's wavelength dictionary.
    """
    std_path = Path(std_path)
    with fits.open(std_path) as hdul:
        info = FrameInfo.from_header(hdul[0].header, std_path)
        data = np.asarray(hdul[1].data, dtype=float)
    ny, nx = data.shape
    info.shape = data.shape
    cfg = info.config
    steps = []
    mask = ~np.isfinite(data)
    data = np.nan_to_num(data)

    # bp: bad-pixel mask from the product directory (same binning)
    bpm_file = product.bpm_file
    if bpm_file is not None:
        with fits.open(bpm_file) as h:
            bpm = np.asarray(h[1].data if len(h) > 1 else h[0].data)
        if bpm.shape == data.shape:
            mask |= bpm.astype(bool)
            steps.append("bp")

    # ag: per-amplifier gain factors.  Replaying raw/flat on a science frame reproduces
    # the pipeline's ``_wr`` product to 1e-5 *without* this step, i.e. the shipped
    # combined flats already carry the auto-gain factors, so this is off by default.
    if apply_gain:
        for entry in product.gain_corrections_for(cfg):
            x0, x1 = int(entry["x0"]), int(entry["x1"])
            data[:, x0:x1] *= float(entry["correction"])
        if product.gain_corrections_for(cfg):
            steps.append("ag")

    # ff: divide by the combined flat of the same configuration (pipeline convention)
    scale = np.ones_like(data)
    if apply_flat:
        flat_file = product.flat_for(cfg)
        if flat_file is None:
            raise FileNotFoundError(f"No combined flat for {cfg.key} in {product.path}; "
                                    "pass apply_flat=False to skip (sensitivity function will then "
                                    "be inconsistent with flat-fielded science frames)")
        with fits.open(flat_file) as h:
            flat = np.asarray(h[1].data if len(h) > 1 else h[0].data, dtype=float)
        if flat.shape != data.shape:
            raise ValueError(f"Flat {flat_file.name} shape {flat.shape} != frame shape {data.shape}")
        good_flat = np.isfinite(flat) & (flat > flat_floor)
        mask |= ~good_flat
        scale = np.where(good_flat, 1.0 / np.where(good_flat, flat, 1.0), 1.0)
        data = data * scale
        steps.append("ff")

    var = estimate_variance(data, readnoise, scale=scale)

    # cr: L.A.Cosmic.  Off by default: on sharp stellar cores it flags a few per cent of
    # the pixels as cosmic rays; the optimal extraction rejects outliers instead.
    if clean_cosmics:
        import astroscrappy

        crmask, cleaned = astroscrappy.detect_cosmics(
            data, inmask=mask, sigclip=4.5, sigfrac=0.3, objlim=5.0, gain=1.0,
            readnoise=readnoise, satlevel=np.inf, niter=4, cleantype="medmask", verbose=False)
        data = np.asarray(cleaned, dtype=float)
        mask |= crmask
        steps.append("cr")

    # cg: mask CCD gaps
    gaps = product.gaps_for(cfg)
    for x0, x1 in gaps:
        mask[:, x0:x1 + 1] = True
    if gaps:
        steps.append("cg")

    # wr: rectify with the matching wavelength solution (row shifts rescaled if the
    # solution comes from a frame with a different spatial binning)
    if solution is None:
        solution = find_solution(load_solutions(product.json("wavelength")), cfg, require_ybin=False)
    try:
        row_scale = cfg.ybin / solution.config.ybin
    except ValueError:
        row_scale = 1.0
    rect, rect_var, rect_mask = solution.rectify(data, ref_row=ref_row, var=var, mask=mask,
                                                 row_scale=row_scale)
    steps.append("wr")
    info.steps = tuple(steps)
    return Frame2D(rect, rect_var, rect_mask, solution.linear_grid(), info,
                   {"source": str(std_path), "solution": solution.key, "readnoise": readnoise,
                    "flat": str(product.flat_for(cfg)) if apply_flat else None})


def mask_ccd_gaps(frame: Frame2D, gaps: list[tuple[int, int]], solution: WavelengthSolution,
                  pad: float = 1.0, ref_row: int | None = None) -> int:
    """Flag the CCD-gap columns of a *rectified* frame in place.

    The SAAO pipeline fills the two RSS chip gaps by interpolation (``_cg``) and
    does not flag them.  Each gap ``(x0, x1)`` in mosaic columns maps, row by
    row, to the wavelength interval ``[wave_row(x0 - pad), wave_row(x1 + pad)]``;
    those rectified columns are masked.  Returns the number of pixels masked.
    """
    if not gaps:
        return 0
    ny, nx = frame.shape
    ref_row = ny // 2 if ref_row is None else ref_row
    try:
        row_scale = frame.info.config.ybin / solution.config.ybin
    except ValueError:
        row_scale = 1.0
    shifts = solution.row_shift(np.arange(ny) * row_scale, ref_row * row_scale)
    n = 0
    for x0, x1 in gaps:
        w0, w1 = solution.wave_at(x0 - pad), solution.wave_at(x1 + pad)
        lo = np.searchsorted(frame.wave, w0 + shifts, side="left")
        hi = np.searchsorted(frame.wave, w1 + shifts, side="right")
        for r in range(ny):
            if hi[r] > lo[r]:
                n += int((~frame.mask[r, lo[r]:hi[r]]).sum())
                frame.mask[r, lo[r]:hi[r]] = True
    frame.meta["gaps_masked"] = [(float(solution.wave_at(x0)), float(solution.wave_at(x1))) for x0, x1 in gaps]
    return n
