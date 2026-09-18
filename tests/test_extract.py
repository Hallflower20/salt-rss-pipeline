import numpy as np
import pytest

from saltrss.extract import extract, find_peaks, find_trace, spatial_profile
from saltrss.frames import load_science_frame, prepare_standard_frame
from saltrss.products import ProductDir
from tests.conftest import STAR_ROW, STAR_SIGMA, SKY_LINES, response, star_flam, DISP
from saltrss.fluxcal import sutherland_extinction


@pytest.fixture(scope="module")
def std_frame(synthetic_product, solution):
    return prepare_standard_frame(synthetic_product / "mbxgpP202501010010.fits", ProductDir(synthetic_product),
                                  solution=solution)


def test_find_trace(std_frame):
    tr = find_trace(std_frame)
    assert abs(tr.center - STAR_ROW) < 0.5
    assert abs(tr.fwhm - 2.3548 * STAR_SIGMA) < 1.0
    assert tr.peaks and abs(tr.peaks[0] - STAR_ROW) <= 1
    assert np.all(np.abs(tr.positions - STAR_ROW) < 1.0)


def test_find_peaks_orders_by_brightness():
    prof = np.zeros(200)
    prof[50] = 100; prof[120] = 300
    assert find_peaks(prof, min_snr=1.0)[:2] == [120.0, 50.0]


@pytest.mark.parametrize("method", ["horne", "boxcar"])
def test_extract_recovers_counts(std_frame, method):
    ex = extract(std_frame, method=method, aperture_fwhm=3.0)
    s = ex.spectrum
    ext = sutherland_extinction()
    expect = star_flam(s.wave) * response(s.wave) * 60.0 * DISP / ext.correction_factor(s.wave, 1.2)
    g = s.gpm & (s.wave > s.wave.min() + 60) & (s.wave < s.wave.max() - 60)
    ratio = s.flux[g] / expect[g]
    assert abs(np.median(ratio) - 1) < 0.03, f"{method}: median flux ratio {np.median(ratio)}"
    assert np.std(ratio) < 0.05
    # uncertainties are of the right order: chi2/dof ~ 1
    chi = (s.flux[g] - expect[g]) * np.sqrt(s.ivar[g])
    assert 0.5 < np.std(chi) < 2.0
    assert ex.sky is not None and ex.sky_windows is not None
    # sky lines removed: residual at line centres not > 5 sigma above continuum trend
    for wl in SKY_LINES:
        i = np.argmin(np.abs(s.wave - wl))
        assert abs(s.flux[i] - expect[i]) < 6 / np.sqrt(s.ivar[i])


def test_extract_faint_science(synthetic_product):
    fr = load_science_frame(ProductDir(synthetic_product).science_frames[0].path)
    ex = extract(fr)
    assert abs(ex.trace.center - STAR_ROW) < 1.0
    assert ex.spectrum.gpm.sum() > 0.9 * ex.spectrum.wave.size
    assert ex.spectrum.meta["exptime"] == 600.0 and ex.spectrum.unit == "count"


def test_spatial_profile_shape(std_frame):
    prof = spatial_profile(std_frame)
    assert prof.shape == (std_frame.shape[0],) and np.argmax(prof) == int(STAR_ROW)


def test_target_selection_prefers_expected_position(std_frame):
    """A brighter object far from the expected row must not steal the trace."""
    import copy

    fr = copy.deepcopy(std_frame)
    fr.data[20:26, :] += 5 * fr.data[57:63, :]   # bright interloper near the slit edge
    tr_b = find_trace(fr, target_selection="brightest")
    assert abs(tr_b.center - 23) < 2
    tr_e = find_trace(fr, expected_row=fr.shape[0] // 2, search_window=20)
    assert abs(tr_e.center - STAR_ROW) < 1
    assert tr_e.peaks[0] == pytest.approx(23, abs=1) and len(tr_e.peaks) >= 2
    tr_none = find_trace(fr, expected_row=100, search_window=5)   # nothing there -> expected row used
    assert tr_none.guess == 100


def test_undetected_target_uses_straight_trace(std_frame):
    """Tracing empty sky must not run away (and sky windows must stay on the frame)."""
    tr = find_trace(std_frame, guess=20.0, window=8)
    assert np.all(tr.positions == 20.0) or np.all(np.abs(tr.positions - 20.0) <= 8)
    ex = extract(std_frame, tr)           # must not raise "background regions overlapped"
    assert ex.spectrum.wave.size == std_frame.shape[1]
