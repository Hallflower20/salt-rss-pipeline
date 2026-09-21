"""End-to-end reduction of a SALT RSS product directory to 1D calibrated spectra."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import fluxcal, reddening, telluric as _telluric
from .extract import ExtractionResult, extract
from .frames import DEFAULT_READNOISE, Frame2D, load_science_frame, mask_ccd_gaps, prepare_standard_frame
from .fluxcal import SensFunc
from .products import FrameInfo, ProductDir
from .spectrum import Spectrum1D
from .wavecal import find_solution, load_solutions

log = logging.getLogger("saltrss")

STANDARDS_ENV = "SALTRSS_STANDARDS_DIR"


def quiet_pypeit(level=logging.WARNING) -> None:
    """pypeit logs at DEBUG by default; keep only warnings and errors.

    pypeit installs a custom Logger subclass, so it must be imported *before*
    ``logging.getLogger("pypeit")`` is first called.
    """
    import pypeit  # noqa: F401

    plog = logging.getLogger("pypeit")
    plog.setLevel(level)
    plog.propagate = False  # pypeit has its own handler; avoid duplicate lines via the root logger


@dataclass
class ReductionResult:
    """Everything produced while reducing one science frame."""

    spectrum: Spectrum1D                    # final calibrated spectrum
    counts: Spectrum1D                      # extracted counts
    fluxed: Spectrum1D                      # after flux calibration (before telluric)
    extraction: ExtractionResult
    frame: Frame2D
    info: FrameInfo
    sensfunc: SensFunc | None
    telluric: _telluric.TelluricResult | None = None
    ebv: float | None = None
    outputs: dict[str, Path] = field(default_factory=dict)
    qa: dict[str, Any] = field(default_factory=dict)

    @property
    def csv(self) -> Path | None:
        return self.outputs.get("csv")

    def summary(self) -> str:
        s = self.spectrum
        g = s.gpm
        snr = np.nanmedian(s.flux[g] * np.sqrt(s.ivar[g])) if g.any() else float("nan")
        lines = [f"{self.info.object}  {self.info.date_obs}  {self.info.config.key}  "
                 f"exptime={self.info.exptime:.0f}s airmass={self.info.airmass:.2f}",
                 f"  trace row {self.extraction.trace.center:.1f}, FWHM {self.extraction.trace.fwhm:.1f} px,"
                 f" {g.sum()}/{g.size} good pixels, median S/N per pixel {snr:.1f}"]
        if self.sensfunc is not None:
            m = self.sensfunc.meta
            lines.append(f"  standard {m.get('standard')} ({m.get('std_date')}), zeropoint rms {m.get('zp_rms', float('nan')):.3f} mag")
        if self.telluric is not None:
            lines.append(f"  telluric: {self.telluric.method} {self.telluric.meta.get('telluric_bands', '')}")
        if self.ebv is not None:
            lines.append(f"  dereddened with E(B-V) = {self.ebv:.4f}")
        for k, p in self.outputs.items():
            lines.append(f"  {k}: {p}")
        return "\n".join(lines)


# ----------------------------------------------------------------- helpers
def _standards_dir(standards_dir) -> Path | None:
    d = standards_dir or os.environ.get(STANDARDS_ENV)
    return Path(d).expanduser() if d else None


def _resolution(info: FrameInfo, spec: Spectrum1D) -> float:
    disp = float(np.median(np.diff(spec.wave)))
    return fluxcal.spectral_resolution(info, disp, spec.meta.get("fwhm_pix"), wave_ref=float(np.median(spec.wave)))


def get_sensfunc(product: ProductDir, info: FrameInfo, *, sensfunc_file=None, standard_file=None,
                 standards_dir=None, workdir: Path | None = None, rebuild: bool = False,
                 readnoise: float = DEFAULT_READNOISE, extinction: str | None = None,
                 extract_kwargs: dict | None = None, sensfunc_kwargs: dict | None = None,
                 std_trace_guess: float | None = None) -> tuple[SensFunc, ExtractionResult | None, Frame2D | None]:
    """Load, or build (and cache), the sensitivity function for ``info``'s configuration.

    Resolution order: explicit ``sensfunc_file`` -> cached file in ``workdir`` ->
    build from ``standard_file`` -> build from the best match in ``standards_dir``.
    """
    if sensfunc_file:
        return SensFunc.read(sensfunc_file), None, None
    cfg = info.config
    std_dir = _standards_dir(standards_dir)
    if standard_file:
        std_info = FrameInfo.from_file(standard_file)
    else:
        if std_dir is None:
            raise FileNotFoundError("No sensitivity function available: pass sensfunc_file, standard_file, "
                                    f"standards_dir, or set ${STANDARDS_ENV}")
        cands = fluxcal.find_standard_frames(std_dir, cfg)
        if not cands:
            raise LookupError(f"No standard-star frame with configuration {cfg.key} under {std_dir}")
        std_info = fluxcal.choose_standard(cands, info.jd)
    cache = None
    if workdir is not None:
        cache = Path(workdir) / "sensfunc" / f"sens_{cfg.key}_{std_info.object}_{std_info.night}.fits"
        if cache.exists() and not rebuild:
            log.info("Using cached sensitivity function %s", cache)
            return SensFunc.read(cache), None, None
    log.info("Building sensitivity function from %s (%s, %s)", std_info.path.name, std_info.object, std_info.date_obs)
    sol = find_solution(load_solutions(product.json("wavelength")), cfg, require_ybin=False)
    # Treat the standard exactly like the science frame: flat-field it only if the
    # pipeline flat-fielded the science frame (no flats -> "_bp_cr_cg_wr" products).
    apply_flat = "ff" in info.steps
    if not apply_flat:
        log.warning("Science frame was not flat-fielded by the pipeline; standard is reduced without a flat too")
    std_frame = prepare_standard_frame(std_info.path, product, solution=sol, readnoise=readnoise,
                                       apply_flat=apply_flat)
    ex = extract(std_frame, trace_guess=std_trace_guess, **(extract_kwargs or {}))
    ex.spectrum.meta["resolution"] = _resolution(std_info, ex.spectrum)
    sens = fluxcal.build_sensfunc(ex.spectrum, std_info, extinction=extinction, **(sensfunc_kwargs or {}))
    sens.meta["std_trace_row"] = float(ex.trace.center)
    sens.meta["std_ybin"] = int(std_info.config.ybin)
    sens.meta["flat_fielded"] = bool(apply_flat)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        sens.write(cache)
        log.info("Wrote %s", cache)
    return sens, ex, std_frame


def _output_stem(info: FrameInfo) -> str:
    obj = "".join(c if c.isalnum() or c in "-_+" else "_" for c in info.object) or "object"
    seq = info.path.name[5:18]  # P<date><seq>
    return f"{obj}_{seq}_{info.config.key}"


# ------------------------------------------------------------------ driver
def reduce_frame(frame_path, product: ProductDir | str | Path, *, output=None, output_dir=None,
                 sensfunc: SensFunc | None = None, sensfunc_file=None, standard_file=None,
                 standards_dir=None, workdir=None, telluric: str = "pypeit", telluric_kwargs: dict | None = None,
                 deredden: bool = False, ebv: float | None = None, dustmap: str = "sfd", r_v: float = 3.1,
                 extinction_law: str = "fitzpatrick99", trace_guess: float | None = None,
                 target_selection: str = "expected", expected_row: float | None = None,
                 extract_method: str = "horne", aperture_fwhm: float = 2.5, sky_sep_fwhm: float = 3.0,
                 sky_width: float = 30.0, profile_order: int = 4, readnoise: float = DEFAULT_READNOISE,
                 atm_extinction: str | None = None, write_fits: bool = True, qa_plot: bool = True,
                 typesafe_review: bool = False, include_mask: bool = False, verbose: bool = False) -> ReductionResult:
    """Reduce one rectified science frame to a calibrated 1D spectrum.

    Parameters
    ----------
    frame_path : path
        ``*_wr.fits`` science product.
    product : ProductDir or path
        Its product directory (calibrations for the standard come from here).
    output : path, optional
        CSV path.  Default ``<output_dir>/<object>_<frame>_<config>.csv``.
    sensfunc, sensfunc_file, standard_file, standards_dir
        Sensitivity function source; see :func:`get_sensfunc`.
    workdir : path, optional
        Where sensitivity functions are cached (default: ``output_dir/saltrss_work``).
    telluric : {"pypeit", "skycalc", "none"}
    deredden, ebv, dustmap, r_v, extinction_law
        Milky Way correction: ``ebv`` overrides the ``dustmap`` query.
    trace_guess, target_selection, expected_row
        Which object on the slit to extract (see :func:`saltrss.extract.find_trace`).
        By default the brightest peak within 60 rows of the standard star's row
        (or the slit centre) is taken.
    extract_method, aperture_fwhm, sky_sep_fwhm, sky_width, profile_order
        Extraction controls (see :func:`saltrss.extract.extract`).
    atm_extinction : str, optional
        Atmospheric extinction curve (default Sutherland).
    """
    if not verbose:
        quiet_pypeit()
    product = product if isinstance(product, ProductDir) else ProductDir(product)
    frame_path = Path(frame_path)
    frame = load_science_frame(frame_path, readnoise=readnoise)
    info = frame.info
    # the pipeline interpolates across the chip gaps ("_cg"); we do not want that data
    gaps = product.gaps_for(info.config)
    if gaps and product.has_json("wavelength"):
        try:
            sol = find_solution(load_solutions(product.json("wavelength")), info.config, require_ybin=False)
            mask_ccd_gaps(frame, gaps, sol)
        except LookupError:
            log.warning("No wavelength solution to locate the CCD gaps; gap-filled pixels not masked")
    out_dir = Path(output_dir) if output_dir else (Path(output).parent if output else Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = Path(workdir) if workdir else out_dir / "saltrss_work"
    stem = _output_stem(info)
    csv_path = Path(output) if output else out_dir / f"{stem}.csv"

    # 1. sensitivity function
    std_ex = std_frame = None
    if sensfunc is None:
        sensfunc, std_ex, std_frame = get_sensfunc(
            product, info, sensfunc_file=sensfunc_file, standard_file=standard_file,
            standards_dir=standards_dir, workdir=workdir, readnoise=readnoise, extinction=atm_extinction,
            extract_kwargs=dict(method=extract_method, aperture_fwhm=aperture_fwhm, sky_sep_fwhm=sky_sep_fwhm,
                                sky_width=sky_width, profile_order=profile_order))
    if not sensfunc.meta.get("config", "").startswith(info.config.key.split("GA")[0]):
        log.warning("Sensitivity function configuration %s differs from science %s",
                    sensfunc.meta.get("config"), info.config.key)

    # 2. extraction (target expected near the standard star's slit position)
    if expected_row is None and sensfunc.meta.get("std_trace_row") is not None:
        expected_row = float(sensfunc.meta["std_trace_row"]) * \
            float(sensfunc.meta.get("std_ybin", info.config.ybin)) / info.config.ybin
    ex = extract(frame, trace_guess=trace_guess, method=extract_method, aperture_fwhm=aperture_fwhm,
                 sky_sep_fwhm=sky_sep_fwhm, sky_width=sky_width, profile_order=profile_order,
                 expected_row=expected_row, target_selection=target_selection)
    counts = ex.spectrum
    counts.meta["resolution"] = _resolution(info, counts)

    # 3. flux calibration
    fluxed = sensfunc.apply(counts, atmext=fluxcal.extinction_curve(sensfunc.meta.get("extinction", atm_extinction)))
    fluxed.meta["resolution"] = counts.meta["resolution"]

    # 4. telluric
    tel = _telluric.correct(fluxed, method=telluric, **(telluric_kwargs or {}))
    spec = tel.spectrum

    # 5. reddening
    ebv_used = None
    if deredden or ebv is not None:
        ebv_used = float(ebv) if ebv is not None else reddening.ebv_from_dustmaps(info.ra_deg, info.dec_deg, dustmap=dustmap)
        spec = reddening.deredden(spec, ebv_used, r_v=r_v, law=extinction_law)

    result = ReductionResult(spectrum=spec, counts=counts, fluxed=fluxed, extraction=ex, frame=frame,
                             info=info, sensfunc=sensfunc, telluric=tel, ebv=ebv_used)
    result.qa["standard_extraction"] = std_ex
    result.qa["standard_frame"] = std_frame

    # 6. outputs
    spec.write_csv(csv_path, include_mask=include_mask)
    result.outputs["csv"] = csv_path
    if write_fits:
        from .io import write_fits_result

        fpath = csv_path.with_suffix(".fits")
        write_fits_result(result, fpath)
        result.outputs["fits"] = fpath
    if qa_plot:
        from .qa import plot_reduction

        ppath = csv_path.with_suffix(".png")
        plot_reduction(result, ppath)
        result.outputs["qa_plot"] = ppath
    if typesafe_review:
        from .qa import typesafe_review as _review

        result.qa["typesafe"] = _review(result)
    return result


def reduce(product_dir, output=None, *, science_file=None, combine: bool = True,
           upload=None, **kwargs) -> list[ReductionResult]:
    """Reduce every rectified science frame in a product directory.

    Parameters
    ----------
    product_dir : path
    output : path, optional
        CSV path when there is a single science frame, otherwise treated as
        the output *directory*.
    science_file : path, optional
        Reduce only this frame.
    combine : bool
        If several frames share a configuration, also write an
        inverse-variance weighted combination ``<object>_combined_<config>.csv``.
    upload : saltrss.skyportal.UploadOptions, optional
        When given, the finished spectra are posted to SkyPortal/Fritz once the
        reduction is complete (the combined spectrum when there is one).  The
        API token comes from ``$FRITZ_TOKEN``; an upload failure is logged but
        never fails the reduction.
    **kwargs
        Passed to :func:`reduce_frame`.

    Returns
    -------
    list of ReductionResult, one per successfully reduced frame.  Frames that
    fail are logged and skipped; a RuntimeError is raised only if none succeed.
    """
    product = ProductDir(product_dir)
    frames = [FrameInfo.from_file(science_file)] if science_file else product.science_frames
    if not frames:
        raise FileNotFoundError(f"No rectified science frames (*_wr.fits, OBSTYPE=OBJECT) in {product.path}")
    single = len(frames) == 1
    results: list[ReductionResult] = []
    output_dir = kwargs.pop("output_dir", None)
    if output is not None and (not single or Path(output).is_dir() or str(output).endswith(os.sep)
                               or Path(output).suffix.lower() not in (".csv", ".txt", ".dat")):
        output_dir, output = Path(output), None
    failures: list[tuple[FrameInfo, Exception]] = []
    for fi in frames:
        log.info("Reducing %s", fi.path.name)
        try:
            results.append(reduce_frame(fi.path, product, output=output if single else None,
                                        output_dir=output_dir, **kwargs))
        except Exception as exc:  # keep going with the other frames of the directory
            log.error("%s failed: %s: %s", fi.path.name, type(exc).__name__, exc)
            failures.append((fi, exc))
    if failures and not results:
        raise RuntimeError("All frames failed: " + "; ".join(f"{fi.path.name}: {exc}" for fi, exc in failures))
    if combine and len(results) > 1:
        groups: dict[str, list[ReductionResult]] = {}
        for r in results:
            groups.setdefault(r.info.config.key, []).append(r)
        for key, grp in groups.items():
            if len(grp) < 2:
                continue
            comb = combine_spectra([r.spectrum for r in grp])
            out_dir = grp[0].csv.parent
            path = out_dir / f"{_output_stem(grp[0].info).split('_')[0]}_combined_{key}.csv"
            comb.write_csv(path, include_mask=kwargs.get("include_mask", False))
            log.info("Wrote combined spectrum %s", path)
            grp[0].outputs["combined_csv"] = path
            grp[0].qa["combined_spectrum"] = comb
            grp[0].qa["combined_members"] = grp
    if upload is not None:
        from .skyportal import SkyPortalError, upload_results

        try:
            upload_results(results, upload)
        except SkyPortalError as exc:  # a reduction is worth keeping even if the upload is not possible
            log.error("SkyPortal upload skipped: %s", exc)
    return results


def combine_spectra(specs: list[Spectrum1D]) -> Spectrum1D:
    """Inverse-variance weighted mean on the grid of the first spectrum."""
    ref = specs[0]
    num = np.zeros_like(ref.wave)
    den = np.zeros_like(ref.wave)
    for s in specs:
        f = np.interp(ref.wave, s.wave, np.where(s.gpm, s.flux, np.nan), left=np.nan, right=np.nan)
        iv = np.interp(ref.wave, s.wave, np.where(s.gpm, s.ivar, 0.0), left=0.0, right=0.0)
        ok = np.isfinite(f) & (iv > 0)
        num[ok] += f[ok] * iv[ok]
        den[ok] += iv[ok]
    good = den > 0
    flux = np.where(good, num / np.where(good, den, 1.0), 0.0)
    return Spectrum1D(ref.wave, flux, den, good, unit=ref.unit,
                      meta={**ref.meta, "combined_from": [s.meta.get("source") for s in specs]})
