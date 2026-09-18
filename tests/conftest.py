"""Synthetic SALT RSS product directory used by the unit tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from numpy.polynomial import legendre as L

NX, NY = 400, 120
W0, DISP = 4000.0, 2.0           # true wavelength solution: w(x) = W0 + DISP*x + CURV*x^2
CURV = 2.0e-5
STAR_ROW = 60.0
STAR_SIGMA = 2.5
SKY_LINES = [4200.0, 4500.0, 4650.0]
CONFIG_KEY = "NM2x2PG0900GA13.625CA27.25"
REAL_DATA = Path("/hildafs/projects/phy220048p/xhall/data/SALT")


def true_wave(x):
    return W0 + DISP * np.asarray(x, float) + CURV * np.asarray(x, float) ** 2


def z_shift(row):
    """Row-dependent wavelength zero point (power series in row), as in the pipeline JSON."""
    return 5000.0 + 0.03 * (np.asarray(row, float) - NY // 2) + 2e-4 * (np.asarray(row, float) - NY // 2) ** 2


def z_coef():
    # convert z_shift(row) to plain power-series coefficients in row
    r = np.arange(NY, dtype=float)
    return np.polynomial.polynomial.polyfit(r, z_shift(r), 2).tolist()


def make_solution_json():
    pm = np.linspace(20, NX - 20, 15)
    wm = true_wave(pm)
    x0, x1 = pm.min(), pm.max()
    xn = 2 * (pm - x0) / (x1 - x0) - 1
    w_coef = L.legfit(xn, wm, 5)
    xn_all = lambda x: 2 * (np.asarray(x, float) - x0) / (x1 - x0) - 1  # noqa: E731
    w_min, w_max = float(L.legval(xn_all(0), w_coef)), float(L.legval(xn_all(NX - 1), w_coef))
    entry = {"reference": "mbxgpP202501010012", "lamp": "Ar",
             "fit": {"function": "legendre", "order": 5, "niter": 5},
             "pm": pm.tolist(), "wm": wm.tolist(), "w_coef": w_coef.tolist(), "w_rms": 0.01,
             "w_min": w_min, "w_max": w_max, "z_coef": z_coef(), "z_rms": 0.01, "n": NX,
             "added": "2025-01-02 10:00:00"}
    return {"obs_date": "20250101", "solutions": {CONFIG_KEY + "BV1": entry}}


def star_flam(wave):
    """Synthetic 'true' F_lambda of the standard, erg/s/cm^2/A (smooth, blue)."""
    return 3e-13 * (wave / 4500.0) ** -2.5


def response(wave):
    """Instrument response: counts per (erg/s/cm^2/A) per second per Angstrom."""
    return 5e15 * np.exp(-0.5 * ((wave - 4600.0) / 700.0) ** 2) + 5e14


def sky_model(wave):
    sky = 40.0 + 0.0 * wave
    for wl in SKY_LINES:
        sky = sky + 600.0 * np.exp(-0.5 * ((wave - wl) / 2.5) ** 2)
    return sky


def base_header(obj="STAR", obstype="OBJECT", propid="CAL_SPST", exptime=60.0, airmass=1.2,
                date="2025-01-01", jd=2460677.5, maskid="PL0400N002", lamp="NONE"):
    h = fits.Header()
    h["OBJECT"] = obj
    h["OBSTYPE"] = obstype
    h["PROPID"] = propid
    h["GRATING"] = "PG0900"
    h["GR-ANGLE"] = 13.625
    h["CAMANG"] = 27.25
    h["CCDSUM"] = "2 2"
    h["FILTER"] = "PC03400"
    h["EXPTIME"] = exptime
    h["AIRMASS"] = airmass
    h["RA"] = "03:10:30.98"
    h["DEC"] = "-68:36:02.20"
    h["DATE-OBS"] = date
    h["JD"] = jd
    h["MASKID"] = maskid
    h["MASKTYP"] = "LONGSLIT"
    h["LAMPID"] = lamp
    h["BVISITID"] = "1"
    return h


def raw_frame(flam_func, exptime, airmass, rng, flat, star_row=STAR_ROW, include_star=True):
    """Unrectified (tilted) frame in counts: (star + sky) * flat, Poisson noise."""
    from saltrss.fluxcal import sutherland_extinction

    ext = sutherland_extinction()
    x = np.arange(NX, dtype=float)
    rows = np.arange(NY, dtype=float)
    img = np.zeros((NY, NX))
    ref = z_shift(NY // 2)
    for r in range(NY):
        wave_r = true_wave(x) + (z_shift(r) - ref)
        sky = sky_model(wave_r) * exptime / 60.0
        img[r] = sky
        if include_star:
            counts_per_pix = flam_func(wave_r) * response(wave_r) * exptime * DISP / ext.correction_factor(wave_r, airmass)
            prof = np.exp(-0.5 * ((r - star_row) / STAR_SIGMA) ** 2) / (STAR_SIGMA * np.sqrt(2 * np.pi))
            img[r] += counts_per_pix * prof
    img = rng.poisson(np.clip(img, 0, None)).astype(float) + rng.normal(0, 3.0, img.shape)
    return img * flat


@pytest.fixture(scope="session")
def synthetic_product(tmp_path_factory) -> Path:
    from saltrss.wavecal import WavelengthSolution

    rng = np.random.default_rng(12345)
    d = tmp_path_factory.mktemp("product")
    sol_json = make_solution_json()
    (d / "wavelength_solutions_nm_spec_ls_img.json").write_text(json.dumps(sol_json))
    (d / "ccd_gaps_spec_ls_img.json").write_text(json.dumps(
        {"wrk": {"2x2": {"1": {"x0": 150, "x1": 156}}}, "db": {}}))
    flat_name = "mbxgpP20250101FlatNM2x2FASLPC03400PG0900GA13.625CA27.25.fits"
    (d / "combined_flats_spec_ls_img.json").write_text(json.dumps(
        {"wrk": {}, "db": {CONFIG_KEY: flat_name}}))
    (d / "auto_gain_correct_spec_ls_img.json").write_text(json.dumps(
        {"wrk": {}, "db": {CONFIG_KEY: [{"amp": 1, "x0": 0, "x1": 200, "correction": 1.01}]}}))
    x = np.arange(NX)
    flat = 1.0 + 0.2 * np.sin(x / 40.0)[None, :] * np.ones((NY, 1))
    fits.HDUList([fits.PrimaryHDU(header=base_header("FLAT", "FLAT", "CAL_FLAT")),
                  fits.ImageHDU(flat.astype(np.float32), name="SCI")]).writeto(d / flat_name)
    bpm = np.zeros((NY, NX), np.uint8)
    bpm[5, 100] = 1
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(bpm, name="BPM")]).writeto(d / "mRSSBpm2x2.fits")

    # standard star (basic product, unrectified)
    std = raw_frame(star_flam, 60.0, 1.2, rng, flat)
    fits.HDUList([fits.PrimaryHDU(header=base_header("STAR", exptime=60.0, airmass=1.2)),
                  fits.ImageHDU(std.astype(np.float32), name="SCI")]).writeto(d / "mbxgpP202501010010.fits")

    # science target: fainter, different exptime/airmass; rectified with the solution like the pipeline
    sol = WavelengthSolution.from_json_entry(*next(iter(sol_json["solutions"].items())))
    sci_flam = lambda w: 0.02 * star_flam(w) * (1 + 0.3 * np.exp(-0.5 * ((w - 4400) / 30) ** 2))  # noqa: E731
    for seq, (exptime, airmass, row) in {11: (600.0, 1.4, STAR_ROW), 13: (600.0, 1.3, STAR_ROW)}.items():
        raw = raw_frame(sci_flam, exptime, airmass, rng, flat, star_row=row)
        rect, _, rmask = sol.rectify(raw / flat, mask=bpm.astype(bool))
        hdr = base_header("TARGET", propid="2025-1-SCI-001", exptime=exptime, airmass=airmass, maskid="PL0150N004")
        sci_hdr = fits.Header()
        sci_hdr["CRPIX1"] = 1
        sci_hdr["CRVAL1"] = sol.w_min
        sci_hdr["CDELT1"] = sol.dispersion
        sci_hdr["CTYPE1"] = "LINEAR"
        fits.HDUList([fits.PrimaryHDU(header=hdr),
                      fits.ImageHDU(rect.astype(np.float32), header=sci_hdr, name="SCI"),
                      fits.ImageHDU(rmask.astype(np.uint8), name="BPM"),
                      fits.ImageHDU(np.zeros_like(bpm), name="CRMASK")]
                     ).writeto(d / f"mbxgpP2025010100{seq}_bp_ag_ff_cr_cg_wr.fits")
        fits.HDUList([fits.PrimaryHDU(header=hdr),
                      fits.ImageHDU(raw.astype(np.float32), name="SCI")]).writeto(d / f"mbxgpP2025010100{seq}.fits")
    # arc: emission lines only, raw and rectified
    arc = raw_frame(star_flam, 10.0, 1.0, rng, flat, include_star=False)
    ahdr = base_header("ARC", "ARC", "CAL_ARC", exptime=10.0, lamp="Ar")
    fits.HDUList([fits.PrimaryHDU(header=ahdr), fits.ImageHDU(arc.astype(np.float32), name="SCI")]
                 ).writeto(d / "mbxgpP202501010012.fits")
    rect, _, _ = sol.rectify(arc / flat)
    fits.HDUList([fits.PrimaryHDU(header=ahdr), fits.ImageHDU(rect.astype(np.float32), name="SCI")]
                 ).writeto(d / "mbxgpP202501010012_wr.fits")
    return d


@pytest.fixture(scope="session")
def reference_spectrum():
    """pypeit Spectrum standing in for the archive entry of the synthetic standard."""
    from pypeit.core.spectrum import Spectrum

    wave = np.arange(3300.0, 9000.0, 5.0)
    return Spectrum(wave, star_flam(wave) / 1e-17,
                    meta={"Name": "SYNTH", "File": "synth.dat", "ra_deg": 47.6, "dec_deg": -68.6})


@pytest.fixture(scope="session")
def solution(synthetic_product):
    from saltrss.wavecal import load_solutions

    return next(iter(load_solutions(json.loads(
        (synthetic_product / "wavelength_solutions_nm_spec_ls_img.json").read_text())).values()))


def pytest_configure(config):
    import logging

    import pypeit  # noqa: F401  (must precede getLogger("pypeit"): pypeit uses a Logger subclass)

    logging.getLogger("pypeit").setLevel(logging.WARNING)
