"""Command-line entry points."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _common_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if not verbose:
        from .pipeline import quiet_pypeit

        quiet_pypeit()
        logging.getLogger("matplotlib").setLevel(logging.WARNING)


def main_reduce(argv=None) -> int:
    from .pipeline import STANDARDS_ENV, reduce

    p = argparse.ArgumentParser(prog="saltrss-reduce",
                                description="Reduce SALT RSS longslit pipeline products to a calibrated 1D spectrum CSV.")
    p.add_argument("product_dir", help="SALT product directory (mbxgpP*_wr.fits + *_spec_ls_img.json)")
    p.add_argument("-o", "--output", help="output CSV (single frame) or output directory")
    p.add_argument("--science-file", help="reduce only this *_wr.fits frame")
    p.add_argument("--standards-dir", help=f"library of CAL_SPST mbxgpP*.fits frames (default ${STANDARDS_ENV})")
    p.add_argument("--standard-file", help="use this specific standard-star frame")
    p.add_argument("--sensfunc", dest="sensfunc_file", help="use a previously saved sensitivity function FITS")
    p.add_argument("--workdir", help="cache directory for sensitivity functions (default <output dir>/saltrss_work)")
    p.add_argument("--telluric", default="pypeit", choices=["pypeit", "skycalc", "none"])
    p.add_argument("--deredden", action="store_true", help="correct Milky Way reddening using dustmaps")
    p.add_argument("--ebv", type=float, help="use this E(B-V) instead of querying a dust map (implies --deredden)")
    p.add_argument("--dustmap", default="sfd", choices=["sfd", "planck", "csfd"])
    p.add_argument("--rv", type=float, default=3.1, help="R_V for the extinction law")
    p.add_argument("--extinction-law", default="fitzpatrick99", choices=["fitzpatrick99", "ccm89", "odonnell94", "fm07"])
    p.add_argument("--trace-guess", type=float, help="approximate row of the target (overrides selection)")
    p.add_argument("--target-selection", default="expected", choices=["expected", "brightest"],
                   help="'expected': brightest peak within 60 rows of the standard star's row / slit centre "
                        "(default); 'brightest': brightest object anywhere on the slit")
    p.add_argument("--expected-row", type=float, help="expected target row (default: standard star's row)")
    p.add_argument("--extract-method", default="horne", choices=["horne", "boxcar"])
    p.add_argument("--profile-order", type=int, default=4, help="polynomial order smoothing the Horne spatial profile")
    p.add_argument("--aperture-fwhm", type=float, default=2.5)
    p.add_argument("--sky-sep-fwhm", type=float, default=3.0)
    p.add_argument("--sky-width", type=float, default=30.0)
    p.add_argument("--readnoise", type=float, default=3.0)
    p.add_argument("--atm-extinction", help="atmospheric extinction curve (default: Sutherland; or a pypeit file name)")
    p.add_argument("--no-fits", action="store_true", help="do not write the multi-extension FITS")
    p.add_argument("--no-qa", action="store_true", help="do not write the QA PNG")
    p.add_argument("--no-combine", action="store_true", help="do not combine frames sharing a configuration")
    p.add_argument("--include-mask", action="store_true", help="add a mask column to the CSV")
    p.add_argument("--typesafe", action="store_true", help="ask TypeSafe for a QA verdict (needs API key)")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    _common_logging(a.verbose)

    results = reduce(a.product_dir, a.output, science_file=a.science_file, combine=not a.no_combine,
                     standards_dir=a.standards_dir, standard_file=a.standard_file, sensfunc_file=a.sensfunc_file,
                     workdir=a.workdir, telluric=a.telluric, deredden=a.deredden or a.ebv is not None, ebv=a.ebv,
                     dustmap=a.dustmap, r_v=a.rv, extinction_law=a.extinction_law, trace_guess=a.trace_guess,
                     target_selection=a.target_selection, expected_row=a.expected_row,
                     extract_method=a.extract_method, profile_order=a.profile_order, aperture_fwhm=a.aperture_fwhm,
                     sky_sep_fwhm=a.sky_sep_fwhm, sky_width=a.sky_width, readnoise=a.readnoise,
                     atm_extinction=a.atm_extinction, write_fits=not a.no_fits, qa_plot=not a.no_qa,
                     include_mask=a.include_mask, typesafe_review=a.typesafe, verbose=a.verbose)
    for r in results:
        print(r.summary())
        if r.qa.get("typesafe"):
            print("  TypeSafe review:", r.qa["typesafe"])
    return 0


def main_sensfunc(argv=None) -> int:
    from .pipeline import get_sensfunc, quiet_pypeit
    from .products import FrameInfo, ProductDir

    p = argparse.ArgumentParser(prog="saltrss-sensfunc",
                                description="Build and save a sensitivity function from a standard-star frame.")
    p.add_argument("product_dir", help="science product directory supplying flat/BPM/wavelength solution")
    p.add_argument("-o", "--output", required=True, help="output sensitivity function FITS")
    p.add_argument("--standard-file", help="standard-star mbxgpP*.fits (else best match in --standards-dir)")
    p.add_argument("--standards-dir")
    p.add_argument("--science-file", help="science frame defining the configuration (default: first in dir)")
    p.add_argument("--trace-guess", type=float)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)
    _common_logging(a.verbose)
    product = ProductDir(a.product_dir)
    info = FrameInfo.from_file(a.science_file) if a.science_file else product.science_frames[0]
    sens, _, _ = get_sensfunc(product, info, standard_file=a.standard_file, standards_dir=a.standards_dir,
                              std_trace_guess=a.trace_guess)
    sens.write(a.output)
    print(f"wrote {a.output}: {sens.meta}")
    return 0


def main_list(argv=None) -> int:
    """List frames in a product directory or standards library with their configurations."""
    from .products import ProductDir, scan_frames

    p = argparse.ArgumentParser(prog="saltrss-list", description="List RSS frames and configurations.")
    p.add_argument("path", help="product directory or standards library root")
    p.add_argument("--standards", action="store_true", help="recursive scan of mbxgpP*.fits basic products")
    a = p.parse_args(argv)
    if a.standards:
        files = sorted(Path(a.path).rglob("mbxgpP*.fits"))
        frames = [f for f in scan_frames(files) if "Flat" not in f.path.name]
    else:
        frames = ProductDir(a.path).frames
    for f in frames:
        flags = ("science" if f.is_science else "arc" if f.is_arc else f.obstype.lower())
        print(f"{f.path.name:45s} {f.object:22s} {f.date_obs} {f.config.key:32s} slit={f.slit_width_arcsec} "
              f"exp={f.exptime:6.0f} am={f.airmass:.2f} {flags} {'rectified' if f.is_rectified else ''}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main_reduce())
