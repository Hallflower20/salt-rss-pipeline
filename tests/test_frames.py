import numpy as np
import pytest
from astropy.io import fits

from saltrss.frames import estimate_variance, load_science_frame, prepare_standard_frame
from saltrss.products import ProductDir
from tests.conftest import NX, NY


def test_estimate_variance():
    v = estimate_variance(np.array([0.0, 100.0]), readnoise=3.0)
    assert np.allclose(v, [9.0, 109.0])
    v2 = estimate_variance(np.array([50.0]), readnoise=3.0, scale=2.0)  # data already multiplied by 2
    assert np.allclose(v2, (25.0 + 9.0) * 4.0)


def test_load_science_frame(synthetic_product):
    pd = ProductDir(synthetic_product)
    fr = load_science_frame(pd.science_frames[0].path)
    assert fr.shape == (NY, NX) and fr.wave.size == NX
    assert fr.mask.dtype == bool and fr.var.shape == fr.shape
    assert fr.info.exptime == 600.0
    cut = fr.cutout(50, 70)
    assert cut.shape == (20, NX) and cut.meta["row_offset"] == 50


def test_prepare_standard_frame(synthetic_product, solution):
    pd = ProductDir(synthetic_product)
    fr = prepare_standard_frame(synthetic_product / "mbxgpP202501010010.fits", pd, solution=solution)
    assert fr.info.steps == ("bp", "ff", "cg", "wr")
    assert fr.shape == (NY, NX)
    assert np.allclose(fr.wave, solution.linear_grid())
    # gap columns are masked in every row (the rectification shifts them slightly)
    assert fr.mask[:, 152].all()
    # flat removed: without cosmic-ray cleaning the result equals rectify(raw/flat)
    raw = fits.getdata(synthetic_product / "mbxgpP202501010010.fits", 1).astype(float)
    flat = fits.getdata(pd.flat_for(fr.info.config), 1).astype(float)
    expect, _, _ = solution.rectify(raw / flat)
    fr2 = prepare_standard_frame(synthetic_product / "mbxgpP202501010010.fits", pd, solution=solution,
                                 clean_cosmics=False)
    good = ~fr2.mask
    assert np.allclose(fr2.data[good], expect[good], rtol=1e-5)
    # optional gain step multiplies the given columns
    fr3 = prepare_standard_frame(synthetic_product / "mbxgpP202501010010.fits", pd, solution=solution,
                                 clean_cosmics=False, apply_gain=True)
    assert np.isclose(np.nanmedian((fr3.data / fr2.data)[:, 20:100]), 1.01, atol=1e-3)


def test_mask_ccd_gaps(synthetic_product, solution):
    from saltrss.frames import mask_ccd_gaps

    pd = ProductDir(synthetic_product)
    fr = load_science_frame(pd.science_frames[0].path)
    before = fr.mask.sum()
    n = mask_ccd_gaps(fr, pd.gaps_for(fr.info.config), solution, pad=0.0)
    assert n > 0 and fr.mask.sum() == before + n
    w0, w1 = solution.wave_at(150), solution.wave_at(156)
    inside = (fr.wave >= w0) & (fr.wave <= w1)
    assert fr.mask[fr.shape[0] // 2, inside].all()          # centre row: gap columns masked
    just_outside = np.argmin(np.abs(fr.wave - (w1 + 6.0)))
    assert not fr.mask[fr.shape[0] // 2, just_outside]      # but not far beyond it
    # rows far from the centre are shifted by z(row); the masked block follows them
    r = 5
    assert fr.mask[r, np.argmin(np.abs(fr.wave - (0.5 * (w0 + w1) + solution.row_shift(r, fr.shape[0] // 2))))]
    assert fr.meta["gaps_masked"][0][0] == pytest.approx(w0)
