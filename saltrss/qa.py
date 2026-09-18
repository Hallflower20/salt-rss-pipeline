"""Quality-assurance products: diagnostic figure and optional TypeSafe review."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("saltrss")


def plot_reduction(result, path: Path | str) -> None:
    """Six-panel diagnostic PNG for one reduction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ex, spec = result.extraction, result.spectrum
    frame = result.frame
    tr = ex.trace
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"{result.info.object}  {result.info.date_obs}  {result.info.config.key}  "
                 f"{result.info.path.name}", fontsize=12)

    # (0,0) spatial profile
    ax = axes[0, 0]
    ax.plot(tr.profile, color="0.2", lw=0.8)
    ax.axvline(tr.center, color="C3", lw=1, label=f"trace {tr.center:.1f}, FWHM {tr.fwhm:.1f} px")
    if ex.aperture:
        ax.axvspan(*ex.aperture, color="C3", alpha=0.15, label="aperture")
    if ex.sky_windows:
        for w in ex.sky_windows:
            ax.axvspan(*w, color="C0", alpha=0.15)
        ax.plot([], [], color="C0", alpha=0.4, lw=8, label="sky windows")
    ax.set_xlim(max(tr.center - 150, 0), min(tr.center + 150, tr.profile.size))
    ax.set_xlabel("row"); ax.set_ylabel("median counts"); ax.legend(fontsize=8); ax.set_title("spatial profile")

    # (0,1) 2D cutout
    ax = axes[0, 1]
    r0, r1 = ex.meta.get("cutout_rows", (0, frame.shape[0]))
    img = np.where(frame.mask, np.nan, frame.data)[r0:r1]
    vmin, vmax = np.nanpercentile(img, [5, 99])
    ax.imshow(img, aspect="auto", origin="lower", cmap="gray", vmin=vmin, vmax=vmax,
              extent=[frame.wave[0], frame.wave[-1], r0, r1])
    ax.plot(frame.wave, tr.positions, color="C3", lw=0.6)
    ax.set_xlabel("wavelength [Å]"); ax.set_ylabel("row"); ax.set_title("rectified frame (cutout) with trace")

    # (1,0) counts + sky
    ax = axes[1, 0]
    c = result.counts
    ax.plot(c.wave, np.where(c.gpm, c.flux, np.nan), color="0.2", lw=0.7, label="object (counts)")
    if ex.sky is not None:
        ax.plot(ex.sky.wave, ex.sky.flux, color="C0", lw=0.6, alpha=0.7, label="sky per pixel")
    ax.set_xlabel("wavelength [Å]"); ax.set_ylabel("counts"); ax.legend(fontsize=8); ax.set_title("extracted counts")

    # (1,1) sensitivity function
    ax = axes[1, 1]
    if result.sensfunc is not None:
        s = result.sensfunc
        if s.zeropoint_data is not None:
            ax.plot(s.wave, np.where(np.isfinite(s.zeropoint_data) & (np.abs(s.zeropoint_data) < 40),
                                     s.zeropoint_data, np.nan), ".", ms=2, color="0.6", label="data")
        ax.plot(s.wave[s.gpm], s.zeropoint[s.gpm], color="C1", lw=1.2, label="b-spline fit")
        zp = s.zeropoint[s.gpm]
        if zp.size:
            ax.set_ylim(np.nanmin(zp) - 1, np.nanmax(zp) + 1)
        ax.set_title(f"zeropoint: {s.meta.get('standard')} {s.meta.get('std_date')} (rms {s.meta.get('zp_rms', np.nan):.3f})")
        ax.legend(fontsize=8)
    ax.set_xlabel("wavelength [Å]"); ax.set_ylabel("zeropoint [mag]")

    # (2,0) fluxed vs final, telluric
    ax = axes[2, 0]
    f = result.fluxed
    ax.plot(f.wave, np.where(f.gpm, f.flux, np.nan), color="0.6", lw=0.6, label="flux calibrated")
    ax.plot(spec.wave, np.where(spec.gpm, spec.flux, np.nan), color="0.1", lw=0.7, label="final")
    good = spec.gpm & np.isfinite(spec.flux)
    if good.any():
        lo, hi = np.nanpercentile(spec.flux[good], [1, 99.5])
        ax.set_ylim(min(lo, 0) - 0.1 * abs(hi), 1.3 * hi)
    if result.telluric is not None and result.telluric.method != "none":
        ax2 = ax.twinx()
        ax2.plot(spec.wave, result.telluric.transmission, color="C2", lw=0.6, alpha=0.7)
        ax2.set_ylim(0, 1.05); ax2.set_ylabel("telluric transmission", color="C2")
    ax.set_xlabel("wavelength [Å]"); ax.set_ylabel(f"F$_\\lambda$ [{spec.unit}]"); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("calibrated spectrum" + (f", E(B-V)={result.ebv:.3f}" if result.ebv is not None else ""))

    # (2,1) S/N
    ax = axes[2, 1]
    snr = np.where(spec.gpm, spec.flux * np.sqrt(spec.ivar), np.nan)
    ax.plot(spec.wave, snr, color="0.2", lw=0.6)
    ax.set_xlabel("wavelength [Å]"); ax.set_ylabel("S/N per pixel"); ax.set_title("signal-to-noise")
    ax.set_ylim(bottom=0)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ TypeSafe
def diagnostics(result) -> dict[str, Any]:
    """Compact, JSON-serialisable summary of a reduction for review."""
    spec, ex, tr = result.spectrum, result.extraction, result.extraction.trace
    g = spec.gpm & np.isfinite(spec.flux)
    snr = spec.flux[g] * np.sqrt(spec.ivar[g]) if g.any() else np.array([np.nan])
    flux = spec.flux[g] if g.any() else np.array([np.nan])
    prof = tr.profile
    ny = prof.size
    d = {
        "object": result.info.object,
        "exptime_s": result.info.exptime,
        "airmass": result.info.airmass,
        "slit_arcsec": result.info.slit_width_arcsec,
        "trace_row": round(tr.center, 1),
        "slit_center_row": ny // 2,
        "trace_offset_from_slit_center_rows": round(tr.center - ny // 2, 1),
        "spatial_fwhm_pixels": round(tr.fwhm, 2),
        "spatial_fwhm_arcsec": round(tr.fwhm * 0.1267 * result.info.config.ybin, 2),
        "n_candidate_objects_on_slit": len(tr.peaks),
        "candidate_rows_brightest_first": [round(p, 1) for p in tr.peaks[:6]],
        "trace_curvature_rows": round(float(np.nanmax(tr.positions) - np.nanmin(tr.positions)), 2),
        "good_pixel_fraction": round(float(g.mean()), 3),
        "median_snr_per_pixel": round(float(np.nanmedian(snr)), 1),
        "fraction_negative_flux": round(float(np.mean(flux < 0)), 3),
        "median_flux_cgs": float(np.nanmedian(flux)),
        "wavelength_range_A": [round(float(spec.wave[g].min()), 0), round(float(spec.wave[g].max()), 0)] if g.any() else None,
        "flux_units": spec.unit,
    }
    std_ex = result.qa.get("standard_extraction")
    if std_ex is not None:
        d["standard_star_trace_row_same_setup"] = round(std_ex.trace.center, 1)
        d["standard_star_fwhm_pixels"] = round(std_ex.trace.fwhm, 2)
    if result.sensfunc is not None:
        m = result.sensfunc.meta
        d["standard"] = {"name": m.get("standard"), "date": m.get("std_date"), "config": m.get("config"),
                         "zeropoint_fit_rms_mag": m.get("zp_rms"), "slit_arcsec": m.get("slit_arcsec"),
                         "airmass": m.get("std_airmass")}
        d["science_date"] = result.info.date_obs
    if result.telluric is not None:
        d["telluric"] = {"method": result.telluric.method, **{k: (v if not isinstance(v, float) else round(v, 3))
                                                              for k, v in result.telluric.meta.items()}}
    if result.ebv is not None:
        d["ebv"] = round(result.ebv, 4)
    return d


def typesafe_review(result, api_key: str | None = None) -> dict[str, Any] | None:
    """Ask TypeSafe (System One) for calibrated yes/no judgments about the reduction.

    Uses the ``TYPESAFE_API_KEY`` or ``JEV_API_KEY`` environment variable.
    Returns None (and logs) if the SDK or key is unavailable; never raises.
    """
    key = api_key or os.environ.get("TYPESAFE_API_KEY") or os.environ.get("JEV_API_KEY")
    if not key:
        log.info("TypeSafe review skipped: no API key")
        return None
    try:
        from typesafe_sdk import Choice, Noul, TypeSafeClient
    except ImportError:
        log.info("TypeSafe review skipped: typesafe-sdk not installed")
        return None
    state = {"reduction_diagnostics": diagnostics(result),
             "context": ("Long-slit optical spectrum from SALT/RSS of a (usually faint, transient) point "
                         "source reduced automatically: trace found as the brightest peak along the slit, "
                         "sky subtracted from windows either side, Horne-extracted, flux calibrated with a "
                         "spectrophotometric standard observed on another night through a wider slit "
                         "(absolute scale uncertain, shape should be good), telluric bands fitted with a PCA model. "
                         "SALT targets are normally placed near the slit centre, at about the same row where the "
                         "standard star of the same setup lands (given when available). Being a few arcsec off "
                         "centre is normal; other candidate rows near the slit edges are usually field stars or "
                         "edge artefacts.")}
    questions = {
        "target_extracted": Noul(instructions=(
            "Given `reduction_diagnostics`, is the extracted trace plausibly the intended science target "
            "(a point source near the slit centre with a seeing-like FWHM) rather than another object, "
            "a cosmic ray, or noise?")),
        "fluxcal_shape_ok": Noul(instructions=(
            "Is the flux calibration plausibly reliable in spectral shape (standard in the same configuration, "
            "small zeropoint fit rms, no gross problems)?")),
        "telluric_ok": Noul(instructions=(
            "Is the telluric correction plausibly sane or harmlessly skipped (no failed fit, chi2 not absurd)?")),
        "quality": Choice(instructions="Overall quality of this reduced spectrum for science use.",
                          criteria={"good": "high S/N, clean trace, calibrations consistent",
                                    "usable": "science-ready with caveats",
                                    "suspect": "likely problems worth a human look",
                                    "bad": "extraction or calibration clearly failed"}),
    }
    try:
        with TypeSafeClient(api_key=key) as client:
            resp = client.system_one(state=state, questions=questions)
        out = {name: round(a.noul, 3) for name, a in resp.nouls.items()}
        q = resp.choices["quality"]
        out["quality"] = q.choice
        out["quality_confidence"] = round(q.confidence, 3)
        out["quality_probabilities"] = {k: round(v, 3) for k, v in q.probabilities.items()}
        return out
    except Exception as exc:  # network / API problems must never break a reduction
        log.warning("TypeSafe review failed: %s", exc)
        return None
