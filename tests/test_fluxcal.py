import numpy as np
import pytest

from saltrss import fluxcal
from saltrss.extract import extract
from saltrss.frames import load_science_frame, prepare_standard_frame
from saltrss.products import FrameInfo, ProductDir, RSSConfig
from tests.conftest import star_flam


def test_sutherland_extinction():
    ext = fluxcal.sutherland_extinction()
    f = ext.correction_factor(np.array([3500.0, 5500.0, 8000.0]), airmass=1.0)
    assert np.all(f > 1) and f[0] > f[1] > f[2]
    assert np.isclose(f[1], 10 ** (0.4 * 0.165))
    assert np.isclose(ext.correction_factor(np.array([5500.0]), airmass=2.0)[0], 10 ** (0.8 * 0.165))


def test_spectral_resolution(synthetic_product):
    info = FrameInfo.from_file(synthetic_product / "mbxgpP202501010010.fits")  # 4'' slit
    r_slit = fluxcal.spectral_resolution(info, 1.0, wave_ref=5000.0)
    assert np.isclose(r_slit, 5000.0 / (4.0 / (0.1267 * 2)))
    assert fluxcal.spectral_resolution(info, 1.0, fwhm_pix=8.0, wave_ref=5000.0) == pytest.approx(5000 / 8.0)


def test_find_standard_frames(synthetic_product):
    cfg = RSSConfig("PG0900", 13.625, 27.25, 2, 2)
    found = fluxcal.find_standard_frames(synthetic_product, cfg)
    names = sorted(f.path.name for f in found)
    assert "mbxgpP202501010010.fits" in names
    assert not any("_wr" in n or "Flat" in n for n in names)
    assert fluxcal.find_standard_frames(synthetic_product, RSSConfig("PG0700", 4.6, 22.75, 2, 2)) == []


@pytest.fixture(scope="module")
def sensfunc(synthetic_product, solution, reference_spectrum):
    pd = ProductDir(synthetic_product)
    info = FrameInfo.from_file(synthetic_product / "mbxgpP202501010010.fits")
    fr = prepare_standard_frame(info.path, pd, solution=solution)
    ex = extract(fr)
    return fluxcal.build_sensfunc(ex.spectrum, info, reference=reference_spectrum, hydrogen_mask_wid=15.0)


def test_sensfunc_self_consistency(sensfunc, synthetic_product, solution):
    pd = ProductDir(synthetic_product)
    info = FrameInfo.from_file(synthetic_product / "mbxgpP202501010010.fits")
    ex = extract(prepare_standard_frame(info.path, pd, solution=solution))
    fl = sensfunc.apply(ex.spectrum)
    assert fl.unit == "erg/s/cm2/A"
    g = fl.gpm & (fl.wave > fl.wave.min() + 80) & (fl.wave < fl.wave.max() - 80)
    ratio = fl.flux[g] / star_flam(fl.wave[g])
    assert abs(np.median(ratio) - 1) < 0.03
    assert np.percentile(np.abs(ratio - 1), 90) < 0.08
    assert sensfunc.meta["standard"] == "STAR" and sensfunc.meta["extinction"] == "sutherland"
    assert sensfunc.meta["zp_rms"] < 0.1


def test_sensfunc_applies_to_other_exptime_airmass(sensfunc, synthetic_product):
    """Science frame has 10x longer exposure and a different airmass; extinction + exptime must be handled."""
    fr = load_science_frame(ProductDir(synthetic_product).science_frames[0].path)
    ex = extract(fr)
    fl = sensfunc.apply(ex.spectrum)
    truth = 0.02 * star_flam(fl.wave) * (1 + 0.3 * np.exp(-0.5 * ((fl.wave - 4400) / 30) ** 2))
    g = fl.gpm & (fl.wave > fl.wave.min() + 80) & (fl.wave < fl.wave.max() - 80)
    ratio = fl.flux[g] / truth[g]
    assert abs(np.median(ratio) - 1) < 0.05
    # a 30% emission bump at 4400 A survives calibration
    i = np.argmin(np.abs(fl.wave - 4400))
    cont = np.median(fl.flux[(np.abs(fl.wave - 4400) > 60) & (np.abs(fl.wave - 4400) < 120)])
    assert fl.flux[i] / cont > 1.15


def test_sensfunc_io_roundtrip(sensfunc, tmp_path):
    p = tmp_path / "sens.fits"
    sensfunc.write(p)
    s2 = fluxcal.SensFunc.read(p)
    assert np.allclose(s2.zeropoint, sensfunc.zeropoint) and np.array_equal(s2.gpm, sensfunc.gpm)
    assert s2.meta["standard"] == "STAR"
    assert s2.wave_range == sensfunc.wave_range
