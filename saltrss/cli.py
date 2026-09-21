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
    _add_upload_args(p)
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
    if a.upload:
        return _upload_reductions(results, a)
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


# --------------------------------------------------------------- SkyPortal
def _add_upload_args(p, with_flag: bool = True) -> None:
    """Options shared by ``saltrss-reduce --upload`` and ``saltrss-upload``.

    There is deliberately no ``--token`` option: the token is read from
    ``$FRITZ_TOKEN`` (or ``$SKYPORTAL_TOKEN``, or ``--token-file``) so it never
    lands in the shell history or in the process list.
    """
    from .skyportal import DEFAULT_ENDPOINT, DEFAULT_INSTRUMENT

    g = p.add_argument_group("SkyPortal upload")
    if with_flag:
        g.add_argument("--upload", action="store_true",
                       help="post the finished spectrum to SkyPortal (needs $FRITZ_TOKEN)")
    g.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help=f"SkyPortal instance (default {DEFAULT_ENDPOINT})")
    g.add_argument("--token-file", help="read the API token from this file instead of $FRITZ_TOKEN")
    g.add_argument("--obj-id", help="SkyPortal source name (default: the OBJECT header keyword)")
    g.add_argument("--resolve-obj-id", action="store_true",
                   help="if that name is not a source on the instance, use the one found within "
                        "--match-radius of the frame coordinates (SALT's OBJECT is often the TNS name, "
                        "e.g. AT2025vjw, while the instance calls it ZTF25ablxhsq)")
    g.add_argument("--match-radius", type=float, default=3.0, metavar="ARCSEC",
                   help="cone-search radius for --resolve-obj-id and for suggestions (default 3)")
    g.add_argument("--group", dest="groups", action="append", default=[], metavar="NAME_OR_ID",
                   help="share with this group (repeatable; name, nickname or id). "
                        "Default: the sharing groups of the token's owner")
    g.add_argument("--sitewide", action="store_true",
                   help="also share with the sitewide (public) group: visible to every user of the instance")
    g.add_argument("--instrument", default=DEFAULT_INSTRUMENT, help=f"instrument name (default {DEFAULT_INSTRUMENT})")
    g.add_argument("--instrument-id", type=int, help="instrument id (skips the name lookup)")
    g.add_argument("--spectrum-type", help="SkyPortal spectrum type, e.g. source or host (default: instance default)")
    g.add_argument("--label", help="plot-legend label (default: <instrument>-<date>)")
    g.add_argument("--origin", help="origin string (default: saltrss <version>)")
    g.add_argument("--reduced-by", action="append", default=[], metavar="USER",
                   help="SkyPortal user id, or 'me', credited as reducer (repeatable)")
    g.add_argument("--observed-by", action="append", default=[], metavar="USER", help="likewise for the observer")
    g.add_argument("--pi", action="append", default=[], metavar="USER", help="likewise for the PI")
    g.add_argument("--external-reducer", help="free-text reducer (needs --reduced-by as point of contact)")
    g.add_argument("--external-observer", help="free-text observer (needs --observed-by)")
    g.add_argument("--external-pi", help="free-text PI (needs --pi)")
    g.add_argument("--upload-frames", action="store_true",
                   help="upload every frame separately instead of their combination")
    g.add_argument("--dry-run", action="store_true", help="show what would be posted and stop")
    g.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation before posting")
    g.add_argument("--allow-insecure", action="store_true", help="permit a non-HTTPS endpoint (localhost testing)")


def _upload_options(a, dry_run: bool):
    from .skyportal import UploadOptions

    return UploadOptions(endpoint=a.endpoint, token_file=a.token_file, obj_id=a.obj_id,
                         instrument=a.instrument, instrument_id=a.instrument_id,
                         groups=tuple(a.groups), sitewide=a.sitewide, spectrum_type=a.spectrum_type,
                         label=a.label, origin=a.origin, reduced_by=tuple(a.reduced_by),
                         observed_by=tuple(a.observed_by), pi=tuple(a.pi),
                         external_reducer=a.external_reducer, external_observer=a.external_observer,
                         external_pi=a.external_pi, allow_insecure=a.allow_insecure, dry_run=dry_run)


def _confirm(prompt: str, assume_yes: bool) -> bool:
    """Ask before posting; without a terminal, require an explicit --yes."""
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print("Not a terminal: pass --yes to post without confirmation.", file=sys.stderr)
        return False
    return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")


def _check_obj_ids(records, client, resolve: bool = False, radius: float = 3.0) -> None:
    """Warn (or, with ``resolve``, fix) obj_ids the instance does not know.

    A SALT ``OBJECT`` keyword is usually the TNS name, which need not be the
    instance's source name, so a cone search around the frame coordinates
    suggests (or supplies) the right one.
    """
    checked: dict[str, str] = {}
    for r in records:
        obj_id = r["payload"]["obj_id"]
        if obj_id in checked:
            r["payload"]["obj_id"] = checked[obj_id]
            continue
        if client.source(obj_id) is not None:
            checked[obj_id] = obj_id
            continue
        near = client.sources_near(*r["coords"], radius_arcsec=radius) if r.get("coords") else []
        names = [str(s["id"]) for s in near]
        if resolve and len(names) == 1:
            print(f"obj_id {obj_id!r} is unknown; using {names[0]!r}, the source within "
                  f"{radius:g}\" of the frame coordinates")
            checked[obj_id] = names[0]
            r["payload"]["obj_id"] = names[0]
            continue
        hint = (f" Sources within {radius:g}\": {', '.join(names)}."
                f"{' Pass --resolve-obj-id to use it.' if len(names) == 1 else ' Pick one with --obj-id.'}"
                if names else " No source within %g\" of the frame either." % radius)
        print(f"! no source {obj_id!r} on {client.endpoint} (the upload will be rejected).{hint}",
              file=sys.stderr)
        checked[obj_id] = obj_id


def _post_records(records, client, assume_yes: bool, dry_run: bool,
                  resolve_obj_id: bool = False, match_radius: float = 3.0) -> int:
    """Print the prepared payloads, confirm, then POST them.  Returns an exit code."""
    from .skyportal import SkyPortalError, describe

    ready = [r for r in records if "payload" in r]
    for r in records:
        if "error" in r:
            print(f"! {r.get('frame') or r.get('file')}: {r['error']}", file=sys.stderr)
    if not ready:
        return 1
    _check_obj_ids(ready, client, resolve=resolve_obj_id, radius=match_radius)
    for r in ready:
        print(f"spectrum from {r.get('frame') or r.get('file')}:")
        print(describe(r["payload"], client.endpoint))
    if dry_run:
        print("\nDry run: nothing was posted.")
        return 0
    what = f"Post {len(ready)} spectrum/spectra to {client.endpoint}?"
    if not _confirm(what, assume_yes):
        print("Aborted; nothing was posted.")
        return 1
    failed = 0
    for r in ready:
        try:
            r["id"] = client.post_spectrum(r["payload"])
            r["url"] = client.endpoint + f"source/{r['payload']['obj_id']}"
            res = r.get("result")
            if res is not None:
                res.outputs["skyportal"] = r["url"]
                res.qa["skyportal_id"] = r["id"]
            print(f"posted spectrum id {r['id']}: {r['url']}")
        except SkyPortalError as exc:
            failed += 1
            print(f"! upload failed: {exc}", file=sys.stderr)
    return 1 if failed else 0


def _upload_reductions(results, a) -> int:
    """Prepare, show and (after confirmation) post the spectra of a finished reduction."""
    from .skyportal import SkyPortalError, upload_results

    try:
        client = _upload_options(a, True).client()
        # build the payloads first (dry run), then post them once confirmed
        records = upload_results(results, _upload_options(a, True), client,
                                 combined_only=not a.upload_frames)
        return _post_records(records, client, a.yes, a.dry_run, a.resolve_obj_id, a.match_radius)
    except SkyPortalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def main_upload(argv=None) -> int:
    """Post an already reduced spectrum (a saltrss .fits or .csv) to SkyPortal."""
    from .skyportal import SkyPortalError, upload_product

    p = argparse.ArgumentParser(
        prog="saltrss-upload",
        description="Submit a reduced spectrum to SkyPortal/Fritz. The API token is read from "
                    "$FRITZ_TOKEN (or $SKYPORTAL_TOKEN, or --token-file) and is never written anywhere.")
    p.add_argument("spectra", nargs="*", help="saltrss product files (spectrum.fits, or spectrum.csv)")
    p.add_argument("--observed-at", help="ISO UTC start of the exposure (required for a CSV)")
    p.add_argument("--list-groups", action="store_true", help="list the groups you can share with and exit")
    p.add_argument("--list-instruments", action="store_true", help="list the instruments of the instance and exit")
    p.add_argument("--whoami", action="store_true", help="show the token owner and exit")
    p.add_argument("-v", "--verbose", action="store_true")
    _add_upload_args(p, with_flag=False)
    a = p.parse_args(argv)
    _common_logging(a.verbose)
    try:
        client = _upload_options(a, a.dry_run).client()
        if a.whoami:
            me = client.whoami()
            print(f"{me.get('username')} (user id {me.get('id')}) on {client.endpoint}")
            return 0
        if a.list_groups:
            pub = client.public_group()
            for grp in client.accessible_groups():
                tag = "  [sitewide]" if grp["id"] == pub["id"] else ""
                print(f"{grp['id']:6d}  {grp['name']}{tag}")
            return 0
        if a.list_instruments:
            tel = {t["id"]: (t.get("nickname") or t.get("name")) for t in client.telescopes()}
            for inst in sorted(client.instruments(), key=lambda i: (str(tel.get(i.get("telescope_id"))), str(i.get("name")))):
                print(f"{inst['id']:6d}  {tel.get(inst.get('telescope_id'), '?')}/{inst.get('name')}")
            return 0
        if not a.spectra:
            p.error("give at least one spectrum file (or --list-groups / --list-instruments / --whoami)")
        records = []
        for path in a.spectra:
            opts = _upload_options(a, True)  # build the payload, never post here
            try:
                records.append(upload_product(path, opts, client, observed_at=a.observed_at))
            except SkyPortalError as exc:
                records.append({"file": path, "error": str(exc)})
        return _post_records(records, client, a.yes, a.dry_run, a.resolve_obj_id, a.match_radius)
    except SkyPortalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


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
