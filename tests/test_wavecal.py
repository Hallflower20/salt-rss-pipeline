import json

import numpy as np
import pytest
from astropy.io import fits

from saltrss.products import RSSConfig
from saltrss.wavecal import WavelengthSolution, find_solution, load_solutions, wcs_grid
from tests.conftest import NX, NY, true_wave, z_shift


def test_solution_reproduces_endpoints_and_lines(solution):
    assert solution.domain_is_consistent(1e-3)
    x = np.linspace(0, NX - 1, 50)
    assert np.allclose(solution.wave_at(x), true_wave(x), atol=0.05)
    assert np.allclose(solution.linear_grid()[[0, -1]], [solution.w_min, solution.w_max])
    assert np.isclose(solution.dispersion, (solution.w_max - solution.w_min) / (NX - 1))


def test_row_shift_convention(solution):
    ref = NY // 2
    rows = np.array([0, 30, ref, 90, NY - 1])
    assert np.allclose(solution.row_shift(rows, ref), z_shift(rows) - z_shift(ref), atol=1e-6)
    assert solution.row_shift(ref, ref) == 0.0


def test_domain_fallback_solver(solution):
    entry = {"w_coef": solution.w_coef.tolist(), "z_coef": solution.z_coef.tolist(), "n": solution.n,
             "w_min": solution.w_min, "w_max": solution.w_max}  # no pm -> must solve for the domain
    sol2 = WavelengthSolution.from_json_entry("NM2x2PG0900GA13.625CA27.25", entry)
    assert np.allclose(sol2.x_domain, solution.x_domain, atol=0.05)


def test_rectify_straightens_lines(synthetic_product, solution):
    raw = fits.getdata(synthetic_product / "mbxgpP202501010012.fits", 1).astype(float)
    rect, var, mask = solution.rectify(raw, var=np.ones_like(raw), mask=np.zeros_like(raw, bool))
    grid = solution.linear_grid()
    wl = 4500.0
    j = np.argmin(np.abs(grid - wl))
    peaks = [j - 6 + np.argmax(rect[r, j - 6:j + 7]) for r in range(5, NY - 5, 10)]
    assert np.ptp(peaks) <= 1, "sky line not aligned after rectification"
    # raw peaks must have been tilted by several pixels across the slit
    raw_peaks = [np.argmax(raw[r]) for r in (5, NY - 6)]
    assert abs(raw_peaks[0] - raw_peaks[1]) >= 2
    assert mask.dtype == bool and mask.shape == rect.shape
    assert np.all(np.isfinite(rect))


def test_rectified_product_matches_wcs(synthetic_product, solution):
    hdr = fits.getheader(synthetic_product / "mbxgpP202501010011_bp_ag_ff_cr_cg_wr.fits", 1)
    assert np.allclose(wcs_grid(hdr), solution.linear_grid(), atol=1e-6)


def test_find_solution(synthetic_product):
    sols = load_solutions(json.loads((synthetic_product / "wavelength_solutions_nm_spec_ls_img.json").read_text()))
    cfg = RSSConfig("PG0900", 13.63, 27.25, 2, 2)
    assert find_solution(sols, cfg).key.startswith("NM2x2PG0900")
    with pytest.raises(LookupError):
        find_solution(sols, RSSConfig("PG0700", 4.6, 22.75, 2, 2))


def test_rectify_row_scale(synthetic_product, solution):
    """A frame with half the rows (2x4 binning) rectified with a 2x2 solution: the
    row shift polynomial must be evaluated at row * 2."""
    raw = fits.getdata(synthetic_product / "mbxgpP202501010012.fits", 1).astype(float)
    binned = raw[0::2]                      # every second row ~ coarser spatial binning
    rect, _, _ = solution.rectify(binned, row_scale=2.0)
    grid = solution.linear_grid()
    j = np.argmin(np.abs(grid - 4500.0))
    peaks = [j - 6 + np.argmax(rect[r, j - 6:j + 7]) for r in range(2, binned.shape[0] - 2, 5)]
    assert np.ptp(peaks) <= 1


def test_find_solution_across_binning(synthetic_product):
    sols = load_solutions(json.loads((synthetic_product / "wavelength_solutions_nm_spec_ls_img.json").read_text()))
    cfg24 = RSSConfig("PG0900", 13.625, 27.25, 2, 4)
    with pytest.raises(LookupError):
        find_solution(sols, cfg24)
    assert find_solution(sols, cfg24, require_ybin=False).config.ybin == 2
