"""Object tracing, sky subtraction and 1D extraction (thin wrapper around specreduce)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from astropy import units as u
from astropy.modeling import fitting, models
from astropy.nddata import NDData, VarianceUncertainty
from scipy.ndimage import median_filter
from specreduce.background import Background
from specreduce.extract import BoxcarExtract
from specreduce.tracing import ArrayTrace, FitTrace

from .frames import Frame2D
from .spectrum import Spectrum1D


@dataclass
class TraceResult:
    """Fitted object trace and spatial profile width."""

    positions: np.ndarray  # row of the trace at each column
    fwhm: float            # spatial FWHM in pixels
    guess: float           # starting row
    profile: np.ndarray    # collapsed spatial profile used to find the object
    peaks: list[float] = field(default_factory=list)  # candidate object rows, brightest first

    @property
    def center(self) -> float:
        return float(np.nanmedian(self.positions))


@dataclass
class ExtractionResult:
    spectrum: Spectrum1D
    trace: TraceResult
    sky: Spectrum1D | None
    sky_windows: tuple[tuple[float, float], ...] | None
    aperture: tuple[float, float] | None
    meta: dict[str, Any] = field(default_factory=dict)


def spatial_profile(frame: Frame2D, disp_frac=(0.2, 0.8)) -> np.ndarray:
    """Median spatial profile over the central part of the dispersion axis."""
    ny, nx = frame.shape
    c0, c1 = int(disp_frac[0] * nx), int(disp_frac[1] * nx)
    data = np.where(frame.mask, np.nan, frame.data)[:, c0:c1]
    with np.errstate(all="ignore"):
        prof = np.nanmedian(data, axis=1)
    prof = np.where(np.isfinite(prof), prof, np.nanmedian(prof))
    return prof - median_filter(prof, size=max(51, ny // 20), mode="nearest")


def find_peaks(profile: np.ndarray, edge: int = 10, min_snr: float = 5.0, min_sep: int = 8) -> list[float]:
    """Candidate object rows in a background-subtracted profile, brightest first."""
    from scipy.signal import find_peaks as _fp

    noise = 1.4826 * np.nanmedian(np.abs(profile - np.nanmedian(profile))) + 1e-12
    idx, props = _fp(profile, height=min_snr * noise, distance=min_sep)
    idx = [i for i in idx if edge <= i < profile.size - edge]
    idx.sort(key=lambda i: -profile[i])
    return [float(i) for i in idx]


def _fit_gaussian_fwhm(profile: np.ndarray, center: float, halfwidth: int = 25) -> float:
    lo, hi = int(max(center - halfwidth, 0)), int(min(center + halfwidth + 1, profile.size))
    y = profile[lo:hi]
    x = np.arange(lo, hi, dtype=float)
    amp = max(float(np.nanmax(y)), 1e-3)
    g = models.Gaussian1D(amplitude=amp, mean=center, stddev=2.0) + models.Const1D(0.0)
    g.stddev_0.bounds = (0.5, halfwidth)
    fit = fitting.TRFLSQFitter()
    try:
        res = fit(g, x, np.nan_to_num(y))
        fwhm = 2.3548 * abs(float(res.stddev_0.value))
    except Exception:
        fwhm = 2.3548 * 2.0
    return float(np.clip(fwhm, 1.5, 2 * halfwidth))


def find_trace(frame: Frame2D, guess: float | None = None, window: int = 25, bins: int = 40,
               degree: int = 2, disp_frac=(0.15, 0.85), peak_method: str = "gaussian",
               expected_row: float | None = None, search_window: float = 60.0,
               target_selection: str = "expected") -> TraceResult:
    """Locate and trace the target along the dispersion axis.

    Parameters
    ----------
    guess : float, optional
        Approximate row of the target; overrides the automatic selection.
    expected_row : float, optional
        Where the target is expected (SALT acquires onto a fixed slit position,
        so the standard star's row or the slit centre is a good prior).
        Defaults to the middle row of the frame.
    search_window : float
        With ``target_selection="expected"`` the brightest detected peak within
        ``search_window`` rows of ``expected_row`` is used; if none is detected
        the expected row itself is used as the guess (a warning is logged).
    target_selection : {"expected", "brightest"}
        ``"brightest"`` takes the brightest peak anywhere on the slit.
    window : int
        Half-width (rows) of the trace-fitting window around the guess.
    bins, degree, peak_method
        Passed to `specreduce.tracing.FitTrace`.
    """
    import logging

    ny, nx = frame.shape
    prof = spatial_profile(frame, disp_frac)
    peaks = find_peaks(prof)
    if expected_row is None:
        expected_row = ny // 2
    if guess is None:
        if target_selection == "brightest":
            if not peaks:
                raise RuntimeError("No object detected in the spatial profile; pass trace_guess")
            guess = peaks[0]
        else:
            near = [p for p in peaks if abs(p - expected_row) <= search_window]
            if near:
                guess = near[0]
            else:
                logging.getLogger("saltrss").warning(
                    "No object detected within %.0f rows of the expected row %.0f (peaks at %s); "
                    "tracing at the expected position", search_window, expected_row,
                    [round(p) for p in peaks[:4]])
                guess = float(expected_row)
    fwhm = _fit_gaussian_fwhm(prof, guess)
    guess = float(np.clip(guess, 2, ny - 3))
    window = int(max(3, min(window, guess - 1, ny - 2 - guess)))  # keep the search window on the frame
    nd = frame.to_nddata()
    c0, c1 = int(disp_frac[0] * nx), int(disp_frac[1] * nx)
    ft = FitTrace(nd, bins=bins, guess=float(guess), window=int(window),
                  trace_model=models.Polynomial1D(degree=degree), peak_method=peak_method,
                  disp_bounds=(c0, c1))
    positions = np.asarray(np.ma.filled(ft.trace, np.nan), dtype=float)
    if not np.all(np.isfinite(positions)):
        positions = np.where(np.isfinite(positions), positions, float(guess))
    if np.any(np.abs(positions - guess) > window):
        logging.getLogger("saltrss").warning(
            "Trace fit wandered %.0f rows from the guess (faint or undetected object); "
            "using a straight trace at row %.1f", np.nanmax(np.abs(positions - guess)), guess)
        positions = np.full(nx, float(guess))
    return TraceResult(positions=positions, fwhm=fwhm, guess=float(guess), profile=prof, peaks=peaks)


def _to_spectrum(flux, variance, wave, gpm, meta, unit="count") -> Spectrum1D:
    flux = np.asarray(flux, dtype=float)
    var = np.asarray(variance, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ivar = np.where(np.isfinite(var) & (var > 0), 1.0 / var, 0.0)
    good = gpm & np.isfinite(flux) & (ivar > 0)
    return Spectrum1D(wave, np.where(np.isfinite(flux), flux, 0.0), ivar, good, unit=unit, meta=meta)


def horne_extract(data: np.ndarray, var: np.ndarray, bad: np.ndarray, positions: np.ndarray,
                  halfwidth: float, poly_order: int = 4, sigma_clip: float = 5.0, niter: int = 2):
    """Optimal extraction following Horne (1986, PASP 98, 609).

    The spatial profile is estimated from the data itself: each column is
    normalised by its boxcar sum, then for every row the normalised values are
    fit with a low-order polynomial along the dispersion axis (with sigma
    clipping), negative values are zeroed and the profile is renormalised per
    column.  Pixels deviating by more than ``sigma_clip`` from ``flux * P`` are
    rejected iteratively (cosmic rays).

    Parameters
    ----------
    data, var, bad : ndarray (nrows, ncols)
        Sky-subtracted image, its variance and bad-pixel mask (True = bad).
    positions : ndarray (ncols,)
        Trace row per column.
    halfwidth : float
        Rows either side of the trace included in the extraction.

    Returns
    -------
    flux, variance, ngood : ndarrays (ncols,)
        Optimal flux, its variance and the number of pixels used per column.
    """
    ny, nx = data.shape
    rows = np.arange(ny, dtype=float)[:, None]
    ap = np.abs(rows - positions[None, :]) <= halfwidth
    good = ap & ~bad & np.isfinite(data) & np.isfinite(var) & (var > 0)
    safe_var = np.where(var > 0, var, np.inf)
    # boxcar estimate for the normalisation
    box = np.sum(np.where(good, data, 0.0), axis=0)
    box = np.where(box > 0, box, np.nan)
    pnorm = np.where(good, data / box[None, :], np.nan)
    cols = np.arange(nx, dtype=float)
    xn = 2 * cols / max(nx - 1, 1) - 1
    prof = np.zeros_like(data)
    for r in range(ny):
        use = np.isfinite(pnorm[r]) & good[r]
        if use.sum() < poly_order + 3:
            continue
        y = pnorm[r]
        w = np.sqrt(1.0 / safe_var[r]) * box
        w = np.where(np.isfinite(w) & use, w, 0.0)
        keep = use.copy()
        for _ in range(3):
            coef = np.polynomial.polynomial.polyfit(xn[keep], y[keep], poly_order, w=w[keep])
            model = np.polynomial.polynomial.polyval(xn, coef)
            resid = (y - model) * w
            sig = 1.4826 * np.nanmedian(np.abs(resid[keep])) + 1e-12
            new_keep = use & (np.abs(resid) < 4 * sig)
            if new_keep.sum() == keep.sum() or new_keep.sum() < poly_order + 3:
                break
            keep = new_keep
        prof[r] = np.where(ap[r], model, 0.0)
    prof = np.clip(prof, 0, None)
    norm = prof.sum(axis=0)
    prof = np.where(norm > 0, prof / np.where(norm > 0, norm, 1.0), 0.0)
    use = good & (prof > 0)
    for _ in range(niter + 1):
        num = np.sum(np.where(use, prof * data / safe_var, 0.0), axis=0)
        den = np.sum(np.where(use, prof**2 / safe_var, 0.0), axis=0)
        flux = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
        dev = (data - flux[None, :] * prof) ** 2 / safe_var
        new_use = use & (dev < sigma_clip**2)
        if new_use.sum() == use.sum():
            break
        use = new_use
    fvar = np.where(den > 0, 1.0 / np.where(den > 0, den, 1.0), np.inf)
    return flux, fvar, use.sum(axis=0)


def extract(frame: Frame2D, trace: TraceResult | None = None, method: str = "horne",
            aperture_fwhm: float = 2.5, sky_sep_fwhm: float = 3.0, sky_width: float = 30.0,
            profile_order: int = 4, trace_guess: float | None = None,
            subtract_sky: bool = True, min_aperture: float = 4.0, **trace_kwargs) -> ExtractionResult:
    """Sky-subtract and extract a 1D spectrum from a rectified frame.

    Parameters
    ----------
    method : {"horne", "boxcar"}
        Optimal extraction (:func:`horne_extract`) or `specreduce.extract.BoxcarExtract`.
    aperture_fwhm : float
        Aperture full width in units of the spatial FWHM.
    sky_sep_fwhm, sky_width : float
        Sky windows start ``sky_sep_fwhm * FWHM`` rows from the trace on either
        side and are ``sky_width`` rows wide (`specreduce.background.Background`,
        column-wise sigma-clipped median).
    profile_order : int
        Polynomial order (along dispersion) used to smooth the empirical spatial profile.
    """
    if trace is None:
        trace = find_trace(frame, guess=trace_guess, **trace_kwargs)
    fwhm = max(trace.fwhm, 1.5)
    ap = max(aperture_fwhm * fwhm, min_aperture)
    sep = max(sky_sep_fwhm * fwhm, ap / 2 + 3)
    half = int(np.ceil(sep + sky_width + 5))
    center = trace.center
    r0 = int(max(np.floor(center - half), 0))
    r1 = int(min(np.ceil(center + half) + 1, frame.shape[0]))
    cut = frame.cutout(r0, r1)
    nd = cut.to_nddata()
    tr = ArrayTrace(nd, trace.positions - r0)

    sky_spec = None
    windows = None
    if subtract_sky:
        # specreduce centres each window on trace +/- separation with total width `width`,
        # so the inner edge sits `sep` rows from the trace when separation = sep + width/2.
        pos = trace.positions - r0
        ncut = cut.shape[0]
        lo_room = float(np.nanmin(pos) - sep)               # rows available below the inner edge
        hi_room = float(ncut - 1 - (np.nanmax(pos) + sep))  # rows available above
        widths = [w for w in (min(sky_width, lo_room), min(sky_width, hi_room)) if w >= 5]
        traces = []
        windows = []
        if widths:
            width = float(min(widths))
            if lo_room >= 5:
                traces.append(tr - (sep + width / 2))
                windows.append((center - sep - width, center - sep))
            if hi_room >= 5:
                traces.append(tr + (sep + width / 2))
                windows.append((center + sep, center + sep + width))
        if not traces:
            import logging

            logging.getLogger("saltrss").warning("No room for sky windows around row %.0f; no sky subtraction", center)
            windows = None
        else:
            bg = Background(nd, traces=traces, width=width, statistic="median", sigma=3.0)
            sub = bg.sub_image()
            sky1d = bg.bkg_spectrum()
            sky_var = np.asarray(sky1d.uncertainty.represent_as(VarianceUncertainty).array, dtype=float) \
                if sky1d.uncertainty is not None else np.zeros(frame.wave.size)
            sky_spec = _to_spectrum(np.asarray(sky1d.flux.value), sky_var, frame.wave,
                                    np.ones(frame.wave.size, bool), {"kind": "sky"}, unit="count")
            var2d = np.asarray(sub.uncertainty.represent_as(VarianceUncertainty).array, dtype=float)
            data2d = np.asarray(sub.flux.value, dtype=float)
            mask2d = np.asarray(sub.mask, dtype=bool) if sub.mask is not None else cut.mask
            nd = NDData(np.where(np.isfinite(data2d), data2d, 0.0), unit=u.count,
                        uncertainty=VarianceUncertainty(np.where(np.isfinite(var2d) & (var2d > 0), var2d, np.inf)),
                        mask=mask2d | ~np.isfinite(data2d))
            windows = tuple(windows)

    ap_used = (center - ap / 2, center + ap / 2)
    if method == "horne":
        data2d = np.asarray(nd.data, dtype=float)
        var2d = np.asarray(nd.uncertainty.array, dtype=float)
        bad2d = np.asarray(nd.mask, dtype=bool)
        flux, var, nused = horne_extract(data2d, var2d, bad2d, trace.positions - r0, ap / 2,
                                         poly_order=profile_order)
    elif method == "boxcar":
        spec = BoxcarExtract(nd, tr, width=ap)()
        flux = np.asarray(spec.flux.value, dtype=float)
        var = np.asarray(spec.uncertainty.represent_as(VarianceUncertainty).array, dtype=float) \
            if spec.uncertainty is not None else np.full_like(flux, np.nan)
    else:
        raise ValueError(f"unknown extraction method {method!r}")
    # columns fully masked in the object aperture are unusable
    lo = np.clip(np.floor(trace.positions - r0 - ap / 2).astype(int), 0, cut.shape[0] - 1)
    hi = np.clip(np.ceil(trace.positions - r0 + ap / 2).astype(int), 1, cut.shape[0])
    colgood = np.array([not np.all(cut.mask[lo[i]:hi[i], i]) for i in range(cut.shape[1])])
    meta = {**frame.info.__dict__.get("meta", {}), "exptime": frame.info.exptime,
            "airmass": frame.info.airmass, "object": frame.info.object, "ra_deg": frame.info.ra_deg,
            "dec_deg": frame.info.dec_deg, "date_obs": frame.info.date_obs, "config": frame.info.config.key,
            "source": frame.meta.get("source"), "extraction": method, "trace_center": center,
            "fwhm_pix": fwhm}
    spectrum = _to_spectrum(flux, var, frame.wave, colgood, meta)
    return ExtractionResult(spectrum=spectrum, trace=trace, sky=sky_spec, sky_windows=windows,
                            aperture=ap_used,
                            meta={"cutout_rows": (r0, r1), "sky_sep": sep, "sky_width": sky_width})
