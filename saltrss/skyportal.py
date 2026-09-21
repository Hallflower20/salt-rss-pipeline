"""Submit reduced spectra to a SkyPortal instance (Fritz by default).

The API token is read from the ``FRITZ_TOKEN`` environment variable (or
``SKYPORTAL_TOKEN``, or a file given to :func:`find_token`).  It is sent only
in the ``Authorization`` header of requests to the configured endpoint; it is
never logged, never written into an output product, and never included in a
repr or an exception message (see :meth:`SkyPortalClient._redact`).

Endpoints used (https://skyportal.io/docs/api.html):

======================================  =======================================
``POST /api/spectrum``                  upload one spectrum
``GET  /api/groups``                    groups the token's owner can access
``GET  /api/groups/public``             the sitewide (public) group
``GET  /api/instrument?name=…``         instrument id by exact name
``GET  /api/telescope``                 telescopes, to disambiguate that name
``GET  /api/sources/<obj_id>``          check the object exists
``GET  /api/sources?ra=&dec=&radius=``  find the source by position
``GET  /api/internal/profile``          the token owner's user id
======================================  =======================================
"""
from __future__ import annotations

import logging
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urljoin, urlsplit

import numpy as np

from . import __version__
from .spectrum import Spectrum1D

log = logging.getLogger("saltrss.skyportal")

DEFAULT_ENDPOINT = "https://fritz.science/"
TOKEN_ENV_VARS = ("FRITZ_TOKEN", "SKYPORTAL_TOKEN")
DEFAULT_INSTRUMENT = "SALT/Spectrograph"   # Fritz has no "RSS": SALT's spectrograph is telescope 20 / instrument 40

#: saltrss flux units -> the three strings SkyPortal accepts.
SKYPORTAL_UNITS = {"erg/s/cm2/A": "erg/s/cm/cm/AA", "erg/s/cm/cm/AA": "erg/s/cm/cm/AA",
                   "Jy": "Jy", "AB": "AB"}


class SkyPortalError(RuntimeError):
    """An API call failed, or the request could not be built."""


# ------------------------------------------------------------------- token
def find_token(token: str | None = None, token_file: str | Path | None = None) -> str:
    """Return the API token from (in order) ``token``, ``token_file``, the environment.

    Raises
    ------
    SkyPortalError
        If no token is available.  The token itself never appears in the message.
    """
    if token:
        return token.strip()
    if token_file:
        path = Path(token_file).expanduser()
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            raise SkyPortalError(f"Cannot read token file {path}: {exc.strerror}") from None
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            log.warning("Token file %s is readable by group/others; chmod 600 it", path)
        value = path.read_text().strip()
        if not value:
            raise SkyPortalError(f"Token file {path} is empty")
        return value
    for var in TOKEN_ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            return value
    raise SkyPortalError(
        "No SkyPortal API token found. Put it in your ~/.bash_profile as\n"
        "    export FRITZ_TOKEN=<your token>\n"
        "(or pass --token-file). Tokens are created on the profile page of the instance.")


# ------------------------------------------------------------------ client
@dataclass
class SkyPortalClient:
    """Thin authenticated wrapper over a SkyPortal API.

    Parameters
    ----------
    endpoint : str
        Base URL of the instance, default ``https://fritz.science/``.
    token : str, optional
        API token; by default taken from ``$FRITZ_TOKEN`` / ``$SKYPORTAL_TOKEN``.
        It is excluded from the dataclass repr.
    timeout : float
        Per-request timeout in seconds.
    allow_insecure : bool
        Permit a plain ``http://`` endpoint (only sensible for localhost); by
        default a non-HTTPS endpoint is refused so the token is never sent in
        clear text.
    """

    endpoint: str = DEFAULT_ENDPOINT
    token: str | None = field(default=None, repr=False)
    timeout: float = 60.0
    allow_insecure: bool = False
    token_file: str | Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        import requests

        self.token = find_token(self.token, self.token_file)
        self.endpoint = (self.endpoint or DEFAULT_ENDPOINT).strip()
        if "://" not in self.endpoint:
            self.endpoint = "https://" + self.endpoint
        if not self.endpoint.endswith("/"):
            self.endpoint += "/"
        parts = urlsplit(self.endpoint)
        local = parts.hostname in ("localhost", "127.0.0.1", "::1")
        if parts.scheme != "https" and not (self.allow_insecure or local):
            raise SkyPortalError(f"Refusing to send the API token to a non-HTTPS endpoint ({self.endpoint}); "
                                 "pass allow_insecure=True if you really mean it")
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"token {self.token}",
                                      "User-Agent": f"saltrss/{__version__}"})
        self._groups_cache: list[dict] | None = None
        self._telescopes_cache: list[dict] | None = None

    # ------------------------------------------------------------- plumbing
    def _redact(self, text: Any) -> str:
        """Stringify ``text`` with any occurrence of the token replaced."""
        out = str(text)
        return out.replace(self.token, "<redacted token>") if self.token else out

    def request(self, method: str, path: str, *, params: dict | None = None,
                json_body: dict | None = None) -> Any:
        """Call ``/api/<path>`` and return the ``data`` payload of the response."""
        import requests

        url = urljoin(self.endpoint, "api/" + path.lstrip("/"))
        log.debug("%s %s", method, url)
        try:
            resp = self._session.request(method, url, params=params, json=json_body, timeout=self.timeout)
        except requests.RequestException as exc:
            raise SkyPortalError(f"{method} {url} failed: {self._redact(exc)}") from None
        try:
            payload = resp.json()
        except ValueError:
            raise SkyPortalError(f"{method} {url}: HTTP {resp.status_code}, "
                                 f"non-JSON response: {self._redact(resp.text[:200])}") from None
        if resp.status_code >= 400 or payload.get("status") != "success":
            message = payload.get("message") or payload.get("data") or resp.text[:400]
            hint = " (is the token valid for this endpoint?)" if resp.status_code in (401, 403) else ""
            raise SkyPortalError(f"{method} {url}: HTTP {resp.status_code}: {self._redact(message)}{hint}")
        return payload.get("data")

    def get(self, path: str, **params) -> Any:
        return self.request("GET", path, params=params or None)

    # -------------------------------------------------------------- lookups
    def whoami(self) -> dict:
        """Profile of the token's owner (``id`` is the numeric user id)."""
        return self.get("internal/profile")

    def user_id(self) -> int:
        return int(self.whoami()["id"])

    def accessible_groups(self) -> list[dict]:
        """Groups the token's owner can post to (cached for the client's life)."""
        if self._groups_cache is None:
            data = self.get("groups")
            self._groups_cache = list(data.get("user_accessible_groups") or data.get("user_groups") or [])
        return self._groups_cache

    def public_group(self) -> dict:
        """The instance's sitewide (public) group — visible to every user."""
        return self.get("groups/public")

    def resolve_groups(self, groups: Sequence[str | int] = (), sitewide: bool = False) -> list[int]:
        """Turn group names, nicknames or ids into a list of group ids.

        ``sitewide`` appends the instance's public group, which makes the
        spectrum visible to *every* user of the instance.
        """
        ids: list[int] = []
        named = [g for g in groups if not _is_int(g)]
        table = self.accessible_groups() if named else []
        for g in groups:
            if _is_int(g):
                ids.append(int(g))
                continue
            key = str(g).strip().lower()
            match = [row for row in table
                     if str(row.get("name", "")).lower() == key or str(row.get("nickname") or "").lower() == key]
            if not match:
                available = ", ".join(sorted(str(r.get("name")) for r in table)) or "(none)"
                raise SkyPortalError(f"No accessible group named {g!r}. Available: {available}")
            ids.append(int(match[0]["id"]))
        if sitewide:
            pub = self.public_group()
            log.warning("Including the sitewide group %r: the spectrum will be visible to every user of %s",
                        pub.get("name"), self.endpoint)
            ids.append(int(pub["id"]))
        return list(dict.fromkeys(ids))

    def instruments(self, name: str | None = None) -> list[dict]:
        data = self.get("instrument", **({"name": name} if name else {}))
        return data if isinstance(data, list) else [data]

    def telescopes(self) -> list[dict]:
        if self._telescopes_cache is None:
            data = self.get("telescope")
            self._telescopes_cache = data if isinstance(data, list) else [data]
        return self._telescopes_cache

    def resolve_instrument(self, name: str | None = DEFAULT_INSTRUMENT, instrument_id: int | None = None) -> int:
        """Instrument id from an explicit id, or from a name.

        Instrument names repeat across telescopes on a large instance (SALT's is
        simply called "Spectrograph"), so ``name`` may be qualified by the
        telescope as ``"SALT/Spectrograph"`` — the telescope's name or nickname
        before the slash.
        """
        if instrument_id is not None:
            return int(instrument_id)
        telescope, _, inst_name = str(name or "").rpartition("/")
        telescope, inst_name = telescope.strip(), inst_name.strip()
        found = self.instruments(inst_name)
        if telescope:
            key = telescope.lower()
            tel_ids = {t["id"] for t in self.telescopes()
                       if str(t.get("nickname") or "").lower() == key or str(t.get("name", "")).lower() == key}
            if not tel_ids:
                raise SkyPortalError(f"No telescope called {telescope!r} on {self.endpoint}")
            found = [i for i in found if i.get("telescope_id") in tel_ids]
        if not found:
            raise SkyPortalError(f"No instrument {name!r} on {self.endpoint}; "
                                 "'saltrss-upload --list-instruments' shows the available ones")
        if len(found) > 1:
            where = ", ".join(f"id {i['id']} (telescope {i.get('telescope_id')})" for i in found)
            raise SkyPortalError(f"{len(found)} instruments match {name!r} ({where}); "
                                 "qualify it as <telescope>/<instrument> or pass instrument_id")
        return int(found[0]["id"])

    def sources_near(self, ra_deg: float, dec_deg: float, radius_arcsec: float = 3.0) -> list[dict]:
        """Sources within ``radius_arcsec`` of a position (a SALT OBJECT name is
        often not the instance's obj_id: ``AT2025vjw`` is ``ZTF25ablxhsq`` on Fritz)."""
        if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
            return []
        data = self.get("sources", ra=float(ra_deg), dec=float(dec_deg),
                        radius=float(radius_arcsec) / 3600.0, numPerPage=10)
        found = data.get("sources", []) if isinstance(data, dict) else (data or [])
        return list(found)

    def source(self, obj_id: str) -> dict | None:
        """The source ``obj_id``, or None if it does not exist / is not visible."""
        try:
            return self.get(f"sources/{obj_id}")
        except SkyPortalError as exc:
            log.debug("source lookup failed: %s", exc)
            return None

    # ----------------------------------------------------------- the upload
    def post_spectrum(self, payload: dict) -> int:
        """POST ``/api/spectrum``; returns the new spectrum id."""
        data = self.request("POST", "spectrum", json_body=payload)
        return int(data["id"])


def _is_int(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, np.integer)) or (isinstance(value, str) and value.strip().isdigit())


# ------------------------------------------------------------- the payload
@dataclass
class UploadOptions:
    """Everything that decides *where* and *how* a spectrum is posted.

    ``groups`` accepts group ids or (nick)names; ``sitewide`` additionally
    shares with the instance's public group.  With no groups at all SkyPortal
    applies the token owner's default sharing groups.
    """

    endpoint: str = DEFAULT_ENDPOINT
    token: str | None = field(default=None, repr=False)
    token_file: str | Path | None = None
    obj_id: str | None = None
    instrument: str = DEFAULT_INSTRUMENT
    instrument_id: int | None = None
    groups: tuple[str | int, ...] = ()
    sitewide: bool = False
    spectrum_type: str | None = None
    label: str | None = None
    origin: str | None = None
    reduced_by: tuple[str | int, ...] = ()
    observed_by: tuple[str | int, ...] = ()
    pi: tuple[str | int, ...] = ()
    external_reducer: str | None = None
    external_observer: str | None = None
    external_pi: str | None = None
    altdata: dict[str, Any] = field(default_factory=dict)
    timeout: float = 60.0
    allow_insecure: bool = False
    dry_run: bool = False

    def client(self) -> SkyPortalClient:
        return SkyPortalClient(endpoint=self.endpoint, token=self.token, token_file=self.token_file,
                               timeout=self.timeout, allow_insecure=self.allow_insecure)


def _jsonable(value: Any) -> Any:
    """Recursively convert numpy scalars/arrays and Paths to JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return value.name
    return value


def observed_at_iso(info=None, header=None) -> str:
    """ISO UTC start time of the exposure, from JD/MJD or DATE-OBS + TIME-OBS."""
    from astropy.time import Time

    jd = getattr(info, "jd", None)
    if not jd and header is not None:
        if header.get("MJD-OBS"):
            jd = float(header["MJD-OBS"]) + 2400000.5
        elif header.get("JD"):
            jd = float(header["JD"])
    if jd:
        return Time(float(jd), format="jd", scale="utc").isot
    src = header if header is not None else getattr(info, "header", None) or {}
    date = (getattr(info, "date_obs", None) or src.get("DATE-OBS") or "").strip()
    if not date:
        raise SkyPortalError("Cannot determine the observation time; pass observed_at explicitly")
    if "T" in date:
        return Time(date, format="isot", scale="utc").isot
    time_obs = str(src.get("TIME-OBS") or src.get("UTC-OBS") or "").strip()
    if not time_obs:
        # Guessing midnight would put the spectrum up to a day off; make the caller say.
        raise SkyPortalError(f"Only a date ({date}) is available, no time of day: pass observed_at "
                             "(--observed-at, ISO UTC) explicitly")
    return Time(f"{date}T{time_obs}", format="isot", scale="utc").isot


def _points(spec: Spectrum1D) -> tuple[list[float], list[float], list[float] | None]:
    """Good (wavelength, flux, error) points, ascending in wavelength.

    Masked, non-finite and duplicate-wavelength pixels are dropped: JSON has no
    NaN, and SkyPortal plots every point it is given.
    """
    wave = np.asarray(spec.wave, dtype=float)
    flux = np.asarray(spec.flux, dtype=float)
    err = np.asarray(spec.err, dtype=float)
    good = np.asarray(spec.gpm, dtype=bool) & np.isfinite(wave) & np.isfinite(flux)
    if not good.any():
        raise SkyPortalError("The spectrum has no good pixels to upload")
    order = np.argsort(wave[good], kind="stable")
    w = wave[good][order]
    f = flux[good][order]
    e = err[good][order]
    if np.isfinite(e).all():
        return w.tolist(), f.tolist(), e.tolist()
    log.warning("%d/%d pixels have no usable uncertainty; uploading fluxes without errors",
                int((~np.isfinite(e)).sum()), e.size)
    return w.tolist(), f.tolist(), None


def build_payload(spec: Spectrum1D, *, obj_id: str, instrument_id: int, observed_at: str,
                  group_ids: Sequence[int] | str | None = None, spectrum_type: str | None = None,
                  label: str | None = None, origin: str | None = None,
                  altdata: dict | None = None, reduced_by: Sequence[int] = (),
                  observed_by: Sequence[int] = (), pi: Sequence[int] = (),
                  external_reducer: str | None = None, external_observer: str | None = None,
                  external_pi: str | None = None) -> dict:
    """Assemble the ``POST /api/spectrum`` body for ``spec``.

    Raises SkyPortalError if the request would be rejected for a reason we can
    see here (no obj_id, no good pixels, an external reducer without a point of
    contact, ...).
    """
    if not obj_id:
        raise SkyPortalError("No obj_id: pass --obj-id (the SkyPortal source name)")
    wave, flux, err = _points(spec)
    body: dict[str, Any] = {
        "obj_id": str(obj_id),
        "instrument_id": int(instrument_id),
        "observed_at": observed_at,
        "wavelengths": wave,
        "fluxes": flux,
    }
    if err is not None:
        body["errors"] = err
    unit = SKYPORTAL_UNITS.get(spec.unit)
    if unit:
        body["units"] = unit
    elif spec.unit:
        log.warning("Flux unit %r is not one of %s; uploading without a unit",
                    spec.unit, sorted(set(SKYPORTAL_UNITS.values())))
    if group_ids:
        body["group_ids"] = group_ids if isinstance(group_ids, str) else [int(g) for g in group_ids]
    for key, value in (("type", spectrum_type), ("label", label), ("origin", origin)):
        if value:
            body[key] = value
    # SkyPortal requires a real user as point of contact behind every free-text
    # "external" PI / reducer / observer.
    for key, ext_key, ids, external in (("pi", "external_pi", pi, external_pi),
                                        ("reduced_by", "external_reducer", reduced_by, external_reducer),
                                        ("observed_by", "external_observer", observed_by, external_observer)):
        if ids:
            body[key] = [int(i) for i in ids]
        if external:
            if not ids:
                raise SkyPortalError(f"{ext_key} needs at least one SkyPortal user as point of contact: "
                                     f"also pass {key} (e.g. 'me')")
            body[ext_key] = external
    if altdata:
        clean = _jsonable(altdata)
        if clean:
            body["altdata"] = clean
    return body


def result_altdata(result, **extra) -> dict:
    """Provenance of a :class:`~saltrss.pipeline.ReductionResult` for ``altdata``."""
    info = result.info
    spec = result.spectrum
    good = np.asarray(spec.gpm, dtype=bool)
    data: dict[str, Any] = {
        "pipeline": f"saltrss {__version__}",
        "source_file": info.path.name,
        "rss_config": info.config.key,
        "grating": info.config.grating,
        "slit_arcsec": info.slit_width_arcsec,
        "exptime_s": info.exptime,
        "airmass": info.airmass,
        "date_obs": info.date_obs,
        "wavelength_frame": "air",
        "wavelength_range_A": [float(np.min(spec.wave[good])), float(np.max(spec.wave[good]))] if good.any() else None,
        "n_pixels": int(good.sum()),
        "extract_method": result.counts.meta.get("extraction"),
        "trace_row": float(result.extraction.trace.center),
        "spatial_fwhm_pix": float(result.extraction.trace.fwhm),
    }
    if result.sensfunc is not None:
        meta = result.sensfunc.meta
        data.update(standard_star=meta.get("standard"), standard_date=meta.get("std_date"),
                    zeropoint_rms_mag=meta.get("zp_rms"))
    if result.telluric is not None:
        data["telluric"] = result.telluric.method
    if result.ebv is not None:
        data["mw_ebv"] = float(result.ebv)
    if result.qa.get("typesafe"):
        data["typesafe_qa"] = result.qa["typesafe"]
    data.update(extra)
    return data


def payload_from_result(result, options: UploadOptions, client: SkyPortalClient, *,
                        spectrum: Spectrum1D | None = None, observed_at: str | None = None,
                        altdata_extra: dict | None = None) -> dict:
    """Build the POST body for one reduced frame, resolving ids against ``client``."""
    spec = spectrum if spectrum is not None else result.spectrum
    obj_id = options.obj_id or result.info.object
    altdata = {**result_altdata(result, **(altdata_extra or {})), **(options.altdata or {})}
    return build_payload(
        spec, obj_id=obj_id,
        instrument_id=client.resolve_instrument(options.instrument, options.instrument_id),
        observed_at=observed_at or observed_at_iso(result.info),
        group_ids=client.resolve_groups(options.groups, options.sitewide),
        spectrum_type=options.spectrum_type, label=options.label,
        origin=options.origin or f"saltrss {__version__}", altdata=altdata,
        reduced_by=_resolve_users(client, options.reduced_by),
        observed_by=_resolve_users(client, options.observed_by),
        pi=_resolve_users(client, options.pi),
        external_reducer=options.external_reducer, external_observer=options.external_observer,
        external_pi=options.external_pi)


def _resolve_users(client: SkyPortalClient, users: Sequence[str | int]) -> list[int]:
    """User ids; the string ``"me"`` becomes the token owner's id."""
    out = []
    for u in users:
        out.append(client.user_id() if str(u).strip().lower() == "me" else int(u))
    return out


def describe(payload: dict, endpoint: str = DEFAULT_ENDPOINT) -> str:
    """One-paragraph human summary of a payload (never contains the token)."""
    wave = payload.get("wavelengths") or [float("nan")]
    groups = payload.get("group_ids")
    lines = [f"  endpoint     {endpoint}",
             f"  obj_id       {payload.get('obj_id')}",
             f"  observed_at  {payload.get('observed_at')} UTC",
             f"  instrument   id {payload.get('instrument_id')}",
             f"  spectrum     {len(wave)} points, {min(wave):.0f}-{max(wave):.0f} A, "
             f"units {payload.get('units', 'unspecified')}"
             f"{', with errors' if 'errors' in payload else ', no errors'}",
             f"  groups       {groups if groups else 'default sharing groups of the token owner'}"]
    for key in ("type", "label", "origin"):
        if payload.get(key):
            lines.append(f"  {key:<12} {payload[key]}")
    return "\n".join(lines)


# ------------------------------------------------------------------ driver
def upload_results(results: Iterable, options: UploadOptions, client: SkyPortalClient | None = None,
                   combined_only: bool = True) -> list[dict]:
    """Post the spectra of finished reductions.

    Parameters
    ----------
    results : iterable of ReductionResult
    options : UploadOptions
    client : SkyPortalClient, optional
        Re-used connection; built from ``options`` when omitted.
    combined_only : bool
        When several frames of one configuration were combined by
        :func:`saltrss.pipeline.reduce`, upload only that combination (the
        default) rather than each frame.

    Returns
    -------
    list of dict, one per spectrum, with ``obj_id``, ``payload`` and either
    ``id`` (the new SkyPortal spectrum id) or ``dry_run``/``error``.
    """
    results = list(results)
    if not results:
        return []
    client = client or options.client()
    jobs: list[tuple[Any, Spectrum1D | None, dict]] = []
    handled: set[int] = set()
    if combined_only:
        for r in results:
            comb = r.qa.get("combined_spectrum")
            if comb is None:
                continue
            members = r.qa.get("combined_members", [r])
            handled.update(id(m) for m in members)
            extra = {"combined_from": [m.info.path.name for m in members],
                     "total_exptime_s": float(sum(m.info.exptime for m in members)),
                     "n_frames_combined": len(members)}
            jobs.append((r, comb, extra))
    for r in results:
        if id(r) not in handled:
            jobs.append((r, None, {}))
    out = []
    for r, spec, extra in jobs:
        record: dict[str, Any] = {"result": r, "frame": r.info.path.name,
                                  "coords": (r.info.ra_deg, r.info.dec_deg)}
        try:
            observed_at = None
            if spec is not None:
                members = r.qa.get("combined_members", [r])
                observed_at = min(observed_at_iso(m.info) for m in members)
            payload = payload_from_result(r, options, client, spectrum=spec, observed_at=observed_at,
                                          altdata_extra=extra)
            record["payload"] = payload
            record["obj_id"] = payload["obj_id"]
            if options.dry_run:
                record["dry_run"] = True
                log.info("Dry run, not posting:\n%s", describe(payload, client.endpoint))
            else:
                record["id"] = client.post_spectrum(payload)
                url = urljoin(client.endpoint, f"source/{payload['obj_id']}")
                record["url"] = url
                log.info("Posted spectrum %s of %s to %s", record["id"], payload["obj_id"], url)
                r.outputs["skyportal"] = url
                r.qa["skyportal_id"] = record["id"]
        except SkyPortalError as exc:
            record["error"] = str(exc)
            log.error("Upload of %s failed: %s", r.info.path.name, exc)
        out.append(record)
    return out


def upload_product(path: str | Path, options: UploadOptions, client: SkyPortalClient | None = None,
                   observed_at: str | None = None, altdata: dict | None = None) -> dict:
    """Post a spectrum read back from a saltrss product file (``.fits`` or ``.csv``).

    The multi-extension FITS carries the metadata (object, time, configuration,
    calibration provenance); for a CSV, ``obj_id`` and ``observed_at`` have to
    be supplied through ``options`` / ``observed_at``.
    """
    from .io import read_spectrum_product

    path = Path(path)
    client = client or options.client()
    spec, header = read_spectrum_product(path)
    obj_id = options.obj_id or (str(header.get("OBJECT", "")).strip() if header else "")
    when = observed_at or (observed_at_iso(header=header) if header else None)
    if when is None:
        raise SkyPortalError("A CSV carries no observation time: pass --observed-at (ISO UTC)")
    meta = {"pipeline": f"saltrss {header.get('SALTRSS', __version__)}" if header else f"saltrss {__version__}",
            "source_file": str(header.get("SRCFILE", path.name)) if header else path.name,
            "rss_config": str(header.get("RSSCONF", "")) if header else None,
            "exptime_s": float(header["EXPTIME"]) if header and "EXPTIME" in header else None,
            "airmass": float(header["AIRMASS"]) if header and "AIRMASS" in header else None,
            "wavelength_frame": "air",
            "standard_star": str(header.get("STDNAME", "")) or None if header else None,
            "standard_date": str(header.get("STDDATE", "")) or None if header else None,
            "telluric": str(header.get("TELLURIC", "")) or None if header else None,
            "mw_ebv": float(header["EBV"]) if header and "EBV" in header else None,
            **(altdata or {}), **(options.altdata or {})}
    payload = build_payload(
        spec, obj_id=obj_id,
        instrument_id=client.resolve_instrument(options.instrument, options.instrument_id),
        observed_at=when, group_ids=client.resolve_groups(options.groups, options.sitewide),
        spectrum_type=options.spectrum_type, label=options.label,
        origin=options.origin or f"saltrss {__version__}", altdata=meta,
        reduced_by=_resolve_users(client, options.reduced_by),
        observed_by=_resolve_users(client, options.observed_by),
        pi=_resolve_users(client, options.pi),
        external_reducer=options.external_reducer, external_observer=options.external_observer,
        external_pi=options.external_pi)
    record: dict[str, Any] = {"file": str(path), "obj_id": payload["obj_id"], "payload": payload}
    if header is not None and "RA_DEG" in header and "DEC_DEG" in header:
        record["coords"] = (float(header["RA_DEG"]), float(header["DEC_DEG"]))
    if options.dry_run:
        record["dry_run"] = True
    else:
        record["id"] = client.post_spectrum(payload)
        record["url"] = urljoin(client.endpoint, f"source/{payload['obj_id']}")
    return record
