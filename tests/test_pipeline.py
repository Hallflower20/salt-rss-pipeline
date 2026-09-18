import os
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from saltrss import fluxcal, pipeline
from saltrss.extract import extract
from saltrss.frames import prepare_standard_frame
from saltrss.products import FrameInfo, ProductDir
from saltrss.spectrum import Spectrum1D
from tests.conftest import REAL_DATA, star_flam


@pytest.fixture(scope="module")
def sens_file(synthetic_product, solution, reference_spectrum, tmp_path_factory):
    pd = ProductDir(synthetic_product)
    info = FrameInfo.from_file(synthetic_product / "mbxgpP202501010010.fits")
    ex = extract(prepare_standard_frame(info.path, pd, solution=solution))
    sf = fluxcal.build_sensfunc(ex.spectrum, info, reference=reference_spectrum, hydrogen_mask_wid=15.0)
    p = tmp_path_factory.mktemp("sens") / "sens.fits"
    sf.write(p)
    return p


def test_combine_spectra():
    w = np.linspace(4000, 5000, 100)
    a = Spectrum1D(w, np.full(100, 2.0), np.full(100, 1.0))
    b = Spectrum1D(w, np.full(100, 4.0), np.full(100, 3.0))
    b.gpm[10] = False
    c = pipeline.combine_spectra([a, b])
    assert np.isclose(c.flux[0], (2 * 1 + 4 * 3) / 4) and np.isclose(c.ivar[0], 4.0)
    assert np.isclose(c.flux[10], 2.0) and c.gpm.all()


def test_reduce_frame_end_to_end(synthetic_product, sens_file, tmp_path):
    pd = ProductDir(synthetic_product)
    out = tmp_path / "target.csv"
    res = pipeline.reduce_frame(pd.science_frames[0].path, pd, output=out, sensfunc_file=sens_file,
                                telluric="none", ebv=0.05, write_fits=True, qa_plot=True)
    assert res.csv == out and out.exists()
    tab = Table.read(out, format="ascii.csv")
    assert tab.colnames == ["wavelength", "flux", "fluxerr"]
    assert len(tab) == res.spectrum.wave.size
    good = np.isfinite(tab["flux"])
    assert good.sum() > 0.9 * len(tab)
    truth = 0.02 * star_flam(np.asarray(tab["wavelength"]))
    a_lam = __import__("extinction").fitzpatrick99(np.ascontiguousarray(tab["wavelength"], dtype=float), 3.1 * 0.05, 3.1)
    truth = truth * 10 ** (0.4 * a_lam)
    mid = good & (tab["wavelength"] > 4150) & (tab["wavelength"] < 4700) & (np.abs(tab["wavelength"] - 4400) > 80)
    assert abs(np.median(tab["flux"][mid] / truth[mid]) - 1) < 0.06
    assert res.ebv == 0.05 and res.telluric.method == "none"
    assert res.outputs["fits"].exists() and res.outputs["qa_plot"].exists()
    with fits.open(res.outputs["fits"]) as h:
        names = [x.name for x in h]
        for n in ("SPECTRUM", "COUNTS", "FLUXED", "TELLURIC", "SKY", "SENSFUNC"):
            assert n in names
        assert h[0].header["EBV"] == 0.05
    assert "trace row" in res.summary()


def test_reduce_directory_with_combination(synthetic_product, sens_file, tmp_path):
    results = pipeline.reduce(synthetic_product, tmp_path, sensfunc_file=sens_file, telluric="none",
                              write_fits=False, qa_plot=False)
    assert len(results) == 2
    csvs = sorted(p.name for p in tmp_path.glob("*.csv"))
    assert any("combined" in c for c in csvs) and len(csvs) == 3
    comb = Table.read(results[0].outputs["combined_csv"], format="ascii.csv")
    assert len(comb) == results[0].spectrum.wave.size


def test_get_sensfunc_from_standards_dir(synthetic_product, reference_spectrum, tmp_path, monkeypatch):
    """Automatic standard lookup + caching (archive lookup patched to the synthetic reference)."""
    from pypeit.core import standard as _std

    monkeypatch.setattr(_std, "get_standard_spectrum", lambda **kw: reference_spectrum)
    pd = ProductDir(synthetic_product)
    info = pd.science_frames[0]
    sens, ex, fr = pipeline.get_sensfunc(pd, info, standards_dir=synthetic_product, workdir=tmp_path)
    assert sens.meta["standard"] == "STAR" and ex is not None and fr is not None
    cached = list((tmp_path / "sensfunc").glob("sens_*.fits"))
    assert len(cached) == 1
    sens2, ex2, _ = pipeline.get_sensfunc(pd, info, standards_dir=synthetic_product, workdir=tmp_path)
    assert ex2 is None and np.allclose(sens2.zeropoint, sens.zeropoint)


def test_missing_sensfunc_source_raises(synthetic_product, monkeypatch):
    monkeypatch.delenv(pipeline.STANDARDS_ENV, raising=False)
    pd = ProductDir(synthetic_product)
    with pytest.raises(FileNotFoundError):
        pipeline.get_sensfunc(pd, pd.science_frames[0])


@pytest.mark.realdata
@pytest.mark.network
def test_real_at2025abne(tmp_path):
    prod = REAL_DATA / "AT2025abne" / "20251026_RSS"
    stds = REAL_DATA / "standards"
    if not prod.is_dir() or not stds.is_dir():
        pytest.skip("real SALT data not available")
    res = pipeline.reduce(prod, tmp_path, science_file=prod / "mbxgpP202510260083_bp_ag_ff_cr_cg_wr.fits",
                          standards_dir=stds, telluric="none", qa_plot=False)
    s = res[0].spectrum
    g = s.gpm
    assert g.sum() > 3000
    snr = np.nanmedian(s.flux[g] * np.sqrt(s.ivar[g]))
    assert snr > 20
    assert 1e-16 < np.nanmedian(s.flux[g]) < 1e-13


def test_reduce_continues_after_frame_failure(synthetic_product, sens_file, tmp_path, monkeypatch):
    import saltrss.pipeline as pl

    real = pl.reduce_frame
    calls = []

    def flaky(frame_path, *a, **k):
        calls.append(frame_path)
        if len(calls) == 1:
            raise ValueError("boom")
        return real(frame_path, *a, **k)

    monkeypatch.setattr(pl, "reduce_frame", flaky)
    results = pl.reduce(synthetic_product, tmp_path, sensfunc_file=sens_file, telluric="none",
                        write_fits=False, qa_plot=False)
    assert len(results) == 1 and len(calls) == 2
