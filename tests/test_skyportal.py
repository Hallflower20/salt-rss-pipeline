"""SkyPortal upload: offline, against a fake HTTP session (no token, no network)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import requests
from astropy.io import fits
from astropy.table import Table

from saltrss import cli
from saltrss.skyportal import (DEFAULT_ENDPOINT, SkyPortalClient, SkyPortalError, UploadOptions, build_payload,
                               find_token, observed_at_iso, upload_results)
from saltrss.spectrum import Spectrum1D

TOKEN = "s3cret-token-value"

GROUPS = [{"id": 1, "name": "Sitewide Group", "nickname": None},
          {"id": 7, "name": "SALT Transients", "nickname": "salt"}]


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    """Stands in for ``requests.Session``; records every call."""

    instances: list["FakeSession"] = []
    fail_post = False
    missing_source = False

    def __init__(self):
        self.headers = {}
        self.calls = []
        FakeSession.instances.append(self)

    def request(self, method, url, params=None, json=None, timeout=None):
        self.calls.append(SimpleNamespace(method=method, url=url, params=params, body=json))
        path = url.split("/api/", 1)[1]
        if path == "groups":
            return FakeResponse({"status": "success", "data": {"user_accessible_groups": GROUPS}})
        if path == "groups/public":
            return FakeResponse({"status": "success", "data": GROUPS[0]})
        if path.startswith("instrument"):
            name = (params or {}).get("name")
            data = [{"id": 42, "name": "Spectrograph", "telescope_id": 20},
                    {"id": 43, "name": "Spectrograph", "telescope_id": 4},
                    {"id": 44, "name": "LRIS", "telescope_id": 4}]
            return FakeResponse({"status": "success", "data": [i for i in data if not name or i["name"] == name]})
        if path.startswith("telescope"):
            return FakeResponse({"status": "success", "data": [
                {"id": 20, "name": "South African Large Telescope", "nickname": "SALT"},
                {"id": 4, "name": "Keck I Telescope", "nickname": "Keck1"}]})
        if path == "internal/profile":
            return FakeResponse({"status": "success", "data": {"id": 99, "username": "observer"}})
        if path == "sources":                      # cone search
            return FakeResponse({"status": "success",
                                 "data": {"sources": [{"id": "ZTF25ablxhsq", "ra": 347.25325, "dec": -7.51876}]}})
        if path.startswith("sources/"):
            if FakeSession.missing_source:
                return FakeResponse({"status": "error", "message": "Source not found"}, status_code=400)
            return FakeResponse({"status": "success", "data": {"id": path.split("/")[1]}})
        if path == "spectrum" and method == "POST":
            if FakeSession.fail_post:
                return FakeResponse({"status": "error", "message": "bad obj_id"}, status_code=400)
            return FakeResponse({"status": "success", "data": {"id": 12345}})
        return FakeResponse({"status": "error", "message": f"unhandled {method} {path}"}, status_code=404)


@pytest.fixture(autouse=True)
def fake_http(monkeypatch):
    FakeSession.instances.clear()
    FakeSession.fail_post = False
    FakeSession.missing_source = False
    monkeypatch.setattr(requests, "Session", FakeSession)
    monkeypatch.setenv("FRITZ_TOKEN", TOKEN)
    monkeypatch.delenv("SKYPORTAL_TOKEN", raising=False)
    yield


@pytest.fixture
def client():
    return SkyPortalClient()


@pytest.fixture
def spectrum():
    wave = np.array([5000.0, 4000.0, 4500.0, 6000.0])       # deliberately unsorted
    flux = np.array([2e-16, 1e-16, np.nan, 4e-16])          # one NaN
    ivar = np.array([1e32, 1e32, 0.0, 1e32])
    gpm = np.array([True, True, False, False])              # last pixel masked (e.g. CCD gap)
    return Spectrum1D(wave, flux, ivar, gpm, unit="erg/s/cm2/A")


# ------------------------------------------------------------------- token
def test_token_from_environment(monkeypatch):
    assert find_token() == TOKEN
    monkeypatch.setenv("FRITZ_TOKEN", "  padded  ")
    assert find_token() == "padded"
    assert find_token("explicit") == "explicit"


def test_token_from_file_and_missing(tmp_path, monkeypatch):
    path = tmp_path / "token"
    path.write_text(TOKEN + "\n")
    assert find_token(token_file=path) == TOKEN
    monkeypatch.delenv("FRITZ_TOKEN")
    with pytest.raises(SkyPortalError, match="FRITZ_TOKEN"):
        find_token()
    with pytest.raises(SkyPortalError, match="Cannot read token file"):
        find_token(token_file=tmp_path / "nope")


def test_token_never_leaks(client):
    """The token must not show up in a repr, an exception, or a log message."""
    assert TOKEN not in repr(client)
    assert client._session.headers["Authorization"] == f"token {TOKEN}"
    assert TOKEN not in client._redact(f"failed with {TOKEN} in the message")
    with pytest.raises(SkyPortalError) as exc:
        client.get("nonexistent/route")
    assert TOKEN not in str(exc.value)


def test_insecure_endpoint_refused():
    with pytest.raises(SkyPortalError, match="non-HTTPS"):
        SkyPortalClient(endpoint="http://example.org/")
    assert SkyPortalClient(endpoint="http://localhost:5000/").endpoint == "http://localhost:5000/"
    assert SkyPortalClient(endpoint="fritz.science").endpoint == "https://fritz.science/"


# ----------------------------------------------------------------- lookups
def test_resolve_groups(client):
    assert client.resolve_groups(["SALT Transients"]) == [7]
    assert client.resolve_groups(["salt"]) == [7]          # nickname
    assert client.resolve_groups([7, "7"]) == [7]          # id, string id, de-duplicated
    assert client.resolve_groups(["SALT Transients"], sitewide=True) == [7, 1]
    assert client.resolve_groups([], sitewide=True) == [1]
    assert client.resolve_groups([]) == []
    with pytest.raises(SkyPortalError, match="SALT Transients"):
        client.resolve_groups(["Nope"])


def test_resolve_instrument(client):
    """SALT's instrument is just called "Spectrograph", so the telescope disambiguates it."""
    assert client.resolve_instrument("SALT/Spectrograph") == 42
    assert client.resolve_instrument("South African Large Telescope/Spectrograph") == 42
    assert client.resolve_instrument("LRIS") == 44
    assert client.resolve_instrument("Spectrograph", instrument_id=7) == 7
    with pytest.raises(SkyPortalError, match="2 instruments match"):
        client.resolve_instrument("Spectrograph")
    with pytest.raises(SkyPortalError, match="--list-instruments"):
        client.resolve_instrument("SALT/SALTICAM")
    with pytest.raises(SkyPortalError, match="No telescope"):
        client.resolve_instrument("VLT/Spectrograph")


def test_user_and_source(client):
    assert client.user_id() == 99
    assert client.source("AT2025vjw")["id"] == "AT2025vjw"


# ----------------------------------------------------------------- payload
def test_build_payload_filters_and_sorts(spectrum):
    body = build_payload(spectrum, obj_id="AT2025vjw", instrument_id=42,
                         observed_at="2025-09-13T19:51:55.560", group_ids=[7, 1])
    assert body["wavelengths"] == [4000.0, 5000.0]          # NaN and masked pixels dropped, ascending
    assert body["fluxes"] == [1e-16, 2e-16]
    assert np.allclose(body["errors"], [1e-16, 1e-16])
    assert body["units"] == "erg/s/cm/cm/AA"                 # saltrss unit -> SkyPortal unit
    assert body["group_ids"] == [7, 1]
    assert body["obj_id"] == "AT2025vjw" and body["instrument_id"] == 42
    assert not any(isinstance(v, float) and not np.isfinite(v) for v in body["fluxes"])
    json.dumps(body)                                        # must be strictly serialisable


def test_build_payload_validation(spectrum):
    with pytest.raises(SkyPortalError, match="obj_id"):
        build_payload(spectrum, obj_id="", instrument_id=42, observed_at="2025-01-01T00:00:00")
    with pytest.raises(SkyPortalError, match="point of contact"):
        build_payload(spectrum, obj_id="X", instrument_id=42, observed_at="2025-01-01T00:00:00",
                      external_reducer="A. Observer")
    body = build_payload(spectrum, obj_id="X", instrument_id=42, observed_at="2025-01-01T00:00:00",
                         external_reducer="A. Observer", reduced_by=[99])
    assert body["external_reducer"] == "A. Observer" and body["reduced_by"] == [99]
    empty = Spectrum1D(np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([False]))
    with pytest.raises(SkyPortalError, match="no good pixels"):
        build_payload(empty, obj_id="X", instrument_id=42, observed_at="2025-01-01T00:00:00")


def test_unknown_unit_is_dropped(spectrum):
    counts = Spectrum1D(spectrum.wave, spectrum.flux, spectrum.ivar, spectrum.gpm, unit="count")
    body = build_payload(counts, obj_id="X", instrument_id=42, observed_at="2025-01-01T00:00:00")
    assert "units" not in body


def test_observed_at_from_jd():
    info = SimpleNamespace(jd=2460932.32771991, date_obs="2025-09-13", header={})
    assert observed_at_iso(info).startswith("2025-09-13T19:51:55")
    hdr = fits.Header({"DATE-OBS": "2025-09-13", "TIME-OBS": "19:51:55.560"})
    assert observed_at_iso(header=hdr).startswith("2025-09-13T19:51:55")
    assert observed_at_iso(header=fits.Header({"MJD-OBS": 60931.82771991})).startswith("2025-09-13T19:51")
    with pytest.raises(SkyPortalError, match="observation time"):
        observed_at_iso(header=fits.Header())
    with pytest.raises(SkyPortalError, match="no time of day"):
        observed_at_iso(header=fits.Header({"DATE-OBS": "2025-09-13"}))


# ------------------------------------------------------------ upload_results
def fake_result(spectrum, name="mbxgpP202509130044_bp_ag_ff_cr_cg_wr.fits", jd=2460932.32771991, exptime=1500.0):
    info = SimpleNamespace(object="AT2025vjw", path=SimpleNamespace(name=name), date_obs="2025-09-13",
                           jd=jd, exptime=exptime, airmass=1.16, slit_width_arcsec=1.5,
                           ra_deg=347.2533333, dec_deg=-7.5188333,
                           config=SimpleNamespace(key="NM2x2PG0700GA4.5975CA22.75", grating="PG0700"),
                           header={})
    return SimpleNamespace(
        spectrum=spectrum, counts=SimpleNamespace(meta={"extraction": "horne"}), info=info,
        extraction=SimpleNamespace(trace=SimpleNamespace(center=999.9, fwhm=8.7)),
        sensfunc=SimpleNamespace(meta={"standard": "EG274", "std_date": "2025-08-30", "zp_rms": 0.04}),
        telluric=SimpleNamespace(method="pypeit"), ebv=0.05, qa={}, outputs={})


def test_upload_results_posts_and_records(spectrum, client):
    result = fake_result(spectrum)
    opts = UploadOptions(groups=["salt"], sitewide=True, label="SALT/RSS")
    records = upload_results([result], opts, client)
    assert len(records) == 1 and records[0]["id"] == 12345
    assert result.qa["skyportal_id"] == 12345
    assert result.outputs["skyportal"] == "https://fritz.science/source/AT2025vjw"
    post = [c for c in client._session.calls if c.method == "POST"][0]
    assert post.url == "https://fritz.science/api/spectrum"
    body = post.body
    assert body["obj_id"] == "AT2025vjw" and body["group_ids"] == [7, 1]
    assert body["instrument_id"] == 42                     # SALT/Spectrograph
    assert body["observed_at"].startswith("2025-09-13T19:51:55")
    assert body["altdata"]["standard_star"] == "EG274" and body["altdata"]["mw_ebv"] == 0.05
    assert body["altdata"]["wavelength_frame"] == "air"
    assert body["origin"].startswith("saltrss")
    json.dumps(body)


def test_upload_results_dry_run_posts_nothing(spectrum, client):
    records = upload_results([fake_result(spectrum)], UploadOptions(dry_run=True), client)
    assert records[0]["dry_run"] and "id" not in records[0]
    assert not [c for c in client._session.calls if c.method == "POST"]


def test_upload_results_uses_the_combination(spectrum, client):
    a, b = fake_result(spectrum, "frame_a.fits", jd=2460932.4), fake_result(spectrum, "frame_b.fits", jd=2460932.3)
    a.qa["combined_spectrum"] = spectrum
    a.qa["combined_members"] = [a, b]
    records = upload_results([a, b], UploadOptions(), client)
    assert len(records) == 1                                  # one spectrum, not two
    body = records[0]["payload"]
    assert body["altdata"]["combined_from"] == ["frame_a.fits", "frame_b.fits"]
    assert body["altdata"]["total_exptime_s"] == 3000.0
    assert body["observed_at"].startswith("2025-09-13T19:12")  # earliest frame of the combination
    assert len(upload_results([a, b], UploadOptions(), client, combined_only=False)) == 2


def test_upload_failure_is_reported_not_raised(spectrum, client):
    FakeSession.fail_post = True
    records = upload_results([fake_result(spectrum)], UploadOptions(), client)
    assert "bad obj_id" in records[0]["error"] and "id" not in records[0]


# -------------------------------------------------------------- product I/O
@pytest.fixture
def product_fits(tmp_path):
    path = tmp_path / "AT2025vjw.fits"
    hdr = fits.Header({"OBJECT": "AT2025vjw", "DATE-OBS": "2025-09-13", "MJD-OBS": 60931.82771991,
                       "EXPTIME": 1500.0, "AIRMASS": 1.16, "RSSCONF": "NM2x2PG0700GA4.5975CA22.75",
                       "RA_DEG": 347.2533333, "DEC_DEG": -7.5188333,
                       "SRCFILE": "mbxgpP202509130044_bp_ag_ff_cr_cg_wr.fits", "SALTRSS": "0.1.0",
                       "FLUXUNIT": "erg/s/cm2/A", "STDNAME": "EG274", "TELLURIC": "pypeit", "EBV": 0.05})
    tab = Table({"wavelength": [4000.0, 4500.0, 5000.0], "flux": [1e-16, 2e-16, 3e-16],
                 "fluxerr": [1e-17, 1e-17, 1e-17], "mask": np.array([0, 1, 0], np.int8)})
    fits.HDUList([fits.PrimaryHDU(header=hdr), fits.BinTableHDU(tab, name="SPECTRUM")]).writeto(path)
    return path


def test_read_spectrum_product_fits_and_csv(product_fits, tmp_path):
    from saltrss.io import read_spectrum_product

    spec, hdr = read_spectrum_product(product_fits)
    assert hdr["OBJECT"] == "AT2025vjw" and spec.unit == "erg/s/cm2/A"
    assert spec.gpm.tolist() == [True, False, True]
    csv = tmp_path / "spec.csv"
    spec.write_csv(csv)
    spec2, hdr2 = read_spectrum_product(csv)
    assert hdr2 is None and spec2.gpm.tolist() == [True, False, True]
    assert np.allclose(spec2.wave, spec.wave)


def test_upload_product(product_fits, client):
    from saltrss.skyportal import upload_product

    record = upload_product(product_fits, UploadOptions(groups=[7]), client)
    assert record["id"] == 12345
    body = [c for c in client._session.calls if c.method == "POST"][0].body
    assert body["obj_id"] == "AT2025vjw" and body["wavelengths"] == [4000.0, 5000.0]
    assert body["observed_at"].startswith("2025-09-13T19:51")
    assert body["altdata"]["standard_star"] == "EG274"


# --------------------------------------------------------------------- CLI
def test_cli_upload(product_fits, capsys):
    rc = cli.main_upload([str(product_fits), "--group", "salt", "--yes"])
    out = capsys.readouterr().out
    assert rc == 0 and "posted spectrum id 12345" in out
    assert TOKEN not in out
    body = [c for s in FakeSession.instances for c in s.calls if c.method == "POST"][0].body
    assert body["group_ids"] == [7]


def test_cli_upload_dry_run_and_listings(product_fits, capsys):
    assert cli.main_upload([str(product_fits), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Dry run: nothing was posted." in out and "AT2025vjw" in out
    assert not [c for s in FakeSession.instances for c in s.calls if c.method == "POST"]
    assert cli.main_upload(["--list-groups"]) == 0
    assert "[sitewide]" in capsys.readouterr().out
    assert cli.main_upload(["--list-instruments"]) == 0
    assert "SALT/Spectrograph" in capsys.readouterr().out
    assert cli.main_upload(["--whoami"]) == 0
    assert "observer" in capsys.readouterr().out


def test_cli_warns_about_an_unknown_source(product_fits, capsys):
    """A SALT OBJECT name (the TNS name) is often not the instance's obj_id."""
    FakeSession.missing_source = True
    assert cli.main_upload([str(product_fits), "--dry-run"]) == 0
    err = capsys.readouterr().err
    assert "no source 'AT2025vjw'" in err and "ZTF25ablxhsq" in err and "--resolve-obj-id" in err


def test_cli_resolves_the_obj_id_by_coordinates(product_fits, capsys):
    FakeSession.missing_source = True
    assert cli.main_upload([str(product_fits), "--dry-run", "--resolve-obj-id"]) == 0
    out = capsys.readouterr().out
    assert "using 'ZTF25ablxhsq'" in out and "obj_id       ZTF25ablxhsq" in out


def test_sources_near(client):
    assert [s["id"] for s in client.sources_near(347.2533, -7.5188)] == ["ZTF25ablxhsq"]
    assert client.sources_near(float("nan"), 0.0) == []


def test_cli_upload_needs_confirmation(product_fits, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert cli.main_upload([str(product_fits)]) == 1
    assert "Aborted" in capsys.readouterr().out
    assert not [c for s in FakeSession.instances for c in s.calls if c.method == "POST"]


def test_cli_upload_without_token(product_fits, monkeypatch, capsys):
    monkeypatch.delenv("FRITZ_TOKEN")
    assert cli.main_upload([str(product_fits), "--yes"]) == 2
    assert "FRITZ_TOKEN" in capsys.readouterr().err


def test_endpoint_and_instrument_defaults():
    assert DEFAULT_ENDPOINT == "https://fritz.science/"
    assert UploadOptions().endpoint == DEFAULT_ENDPOINT
    assert UploadOptions().instrument == "SALT/Spectrograph"
    assert SkyPortalClient().resolve_instrument(UploadOptions().instrument) == 42
