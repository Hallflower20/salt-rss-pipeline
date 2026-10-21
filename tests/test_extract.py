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


def test_tilted_trace_is_straight_and_clips_bad_bins(std_frame):
    """A tilted trace is fitted by a line; a few corrupted bins are rejected, not chased."""
    from saltrss.frames import Frame2D

    rng = np.random.default_rng(1)
    ny, nx = 120, 400
    x, y = np.arange(nx), np.arange(ny)[:, None]
    true = 54.0 + 12.0 * x / (nx - 1)                     # 12-row tilt across the frame
    row = np.broadcast_to(true, (ny, nx)).copy()
    row[:, 150:170] += 7.0                                # two bins pulled 7 rows off the trace
    row[:, 250:260] -= 7.0                                # and one the other way
    data = 500.0 * np.exp(-0.5 * ((y - row) / STAR_SIGMA) ** 2) + rng.normal(0, 1.0, (ny, nx))
    fr = Frame2D(data, np.ones((ny, nx)), np.zeros((ny, nx), bool), std_frame.wave[:nx], std_frame.info)

    tr = find_trace(fr, guess=60.0)
    assert np.max(np.abs(tr.positions - true)) < 0.3
    unclipped = find_trace(fr, guess=60.0, clip_sigma=0)
    assert np.max(np.abs(unclipped.positions - true)) > np.max(np.abs(tr.positions - true))


def _with_neighbour(frame, row, factor=5.0):
    """Copy of ``frame`` with a star ``factor`` times brighter than the target at ``row``."""
    import copy

    fr = copy.deepcopy(frame)
    peak = np.median(frame.data[int(STAR_ROW)] - frame.data[int(STAR_ROW) + 30])
    y = np.arange(fr.shape[0], dtype=float)[:, None]
    fr.data = fr.data + factor * peak * np.exp(-0.5 * ((y - row) / STAR_SIGMA) ** 2)
    return fr


def test_trace_rows_keep_a_brighter_neighbour_out(std_frame):
    """A brighter star within the search window wins unless the trace is confined to the target's rows."""
    fr = _with_neighbour(std_frame, 30.0)
    assert abs(find_trace(fr, expected_row=STAR_ROW).center - 30) < 1     # the problem
    tr = find_trace(fr, expected_row=STAR_ROW, trace_rows=(48, 72))
    assert abs(tr.center - STAR_ROW) < 0.5 and tr.rows == (48, 72)
    assert np.all(np.abs(tr.positions - STAR_ROW) < 1.0)
    # also when the guess is given: the fit cannot be pulled out of the rows
    assert abs(find_trace(fr, guess=60, trace_rows=(72, 48)).center - STAR_ROW) < 0.5


def test_trace_rows_validation(std_frame):
    with pytest.raises(ValueError, match="outside trace_rows"):
        find_trace(std_frame, guess=30, trace_rows=(48, 72))
    with pytest.raises(ValueError, match="too narrow"):
        find_trace(std_frame, trace_rows=(58, 62))
    with pytest.raises(ValueError, match="off the frame"):
        find_trace(std_frame, trace_rows=(100, 130))
    with pytest.raises(ValueError, match="pair of rows"):
        find_trace(std_frame, trace_rows=(60,))


def test_sky_rows_keep_a_neighbour_out_of_the_sky(std_frame, caplog):
    fr = _with_neighbour(std_frame, 30.0)
    clean = extract(std_frame).spectrum
    g = clean.gpm & (clean.flux > 0.2 * np.median(clean.flux))
    with caplog.at_level("WARNING", logger="saltrss"):
        extract(fr, trace_rows=(48, 72))                  # lower automatic sky window holds the neighbour
    assert "Object(s) at row(s) [30] fall in the sky windows" in caplog.text
    sky_rows = [(42, 50), (70, 78)]
    ex = extract(fr, trace_rows=(48, 72), sky_rows=sky_rows, aperture_fwhm=2.0)
    assert ex.sky_windows == ((42.0, 50.0), (70.0, 78.0)) and ex.meta["sky_rows"] == sky_rows
    assert abs(np.median(ex.spectrum.flux[g] / clean.flux[g]) - 1) < 0.02
    # the sky is the (clipped) median of exactly those rows
    rows = np.r_[42:51, 70:79]
    ref = np.median(std_frame.data[rows], axis=0)
    assert np.median(np.abs(ex.sky.flux / ref - 1)) < 0.01


def test_sky_rows_validation(std_frame, caplog):
    with pytest.raises(ValueError, match="off the frame"):
        extract(std_frame, sky_rows=[(110, 125)])
    with pytest.raises(ValueError, match="empty"):
        extract(std_frame, sky_rows=[])
    with caplog.at_level("WARNING", logger="saltrss"):
        extract(std_frame, sky_rows=[(30, 40), (62, 70)])
    assert "inside the" in caplog.text


def _tilted_frame(std_frame, true, nx=400, ny=120, amp=500.0):
    from saltrss.frames import Frame2D

    rng = np.random.default_rng(2)
    y = np.arange(ny)[:, None]
    data = amp * np.exp(-0.5 * ((y - true) / STAR_SIGMA) ** 2) + rng.normal(0, 1.0, (ny, nx))
    return Frame2D(data, np.ones((ny, nx)), np.zeros((ny, nx), bool), std_frame.wave[:nx], std_frame.info)


def test_sky_rows_follow_a_tilted_trace(std_frame):
    """Sky rows apply at the trace centre and shift with the trace, keeping their clearance."""
    nx = 400
    true = 54.0 + 12.0 * np.arange(nx) / (nx - 1)          # 12-row tilt, centre row 60
    ex = extract(_tilted_frame(std_frame, true), sky_rows=[(46, 50), (70, 74)], aperture_fwhm=2.0)
    assert np.max(np.abs(ex.trace.positions - true)) < 0.3
    # fixed rows 46-50 would sit 4-8 rows (< 2 sigma at the near edge) from the star at the left end
    assert np.max(np.abs(ex.sky.flux)) < 2.0


def test_trace_ignores_a_run_of_lost_bins(std_frame):
    """Bins where a faint target vanishes (here: another source takes over) must not tilt the trace."""
    nx = 400
    true = np.where(np.arange(nx) < 130, 54.0, STAR_ROW)    # first quarter of the traced range at row 54
    tr = find_trace(_tilted_frame(std_frame, true), guess=STAR_ROW)
    assert np.all(np.abs(tr.positions - STAR_ROW) < 0.3)
    unclipped = find_trace(_tilted_frame(std_frame, true), guess=STAR_ROW, clip_sigma=0)
    assert np.max(np.abs(unclipped.positions - STAR_ROW)) > 1.0


def test_clipped_trace_ignores_fallback_bins_and_keeps_real_curvature():
    """Patterns seen on real frames (2026ofr, AT2025adgo) for the robust start of the trace fit."""
    from types import SimpleNamespace

    from saltrss.extract import _clipped_trace

    x = np.linspace(500, 2700, 40)
    # undetected target: FitTrace repeats the last successful bin for every failed one, so
    # there are only three measurements and the caller keeps FitTrace's own (flagged) fit
    y = np.r_[[1148.1] * 10, [1002.6] * 25, [1030.0] * 5]
    assert _clipped_trace(SimpleNamespace(_bin_peaks=(x, y)), 1, 3172, 3.0, fwhm=5.1) is None
    # bright, slightly curved trace: bins 2.5 rows off a line (< FWHM/2) are not discarded up front
    y = 1006.0 - 0.0025 * (x - 1600) + np.where(x < 850, 2.5, 0.0)
    pos = _clipped_trace(SimpleNamespace(_bin_peaks=(x, y)), 1, 3172, 3.0, fwhm=13.2)
    lsq = np.polyval(np.polyfit(x, y, 1), np.arange(3172))
    assert np.allclose(pos, lsq, atol=0.05)
