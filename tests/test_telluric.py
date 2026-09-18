import numpy as np
import pytest

from saltrss import telluric
from saltrss.spectrum import Spectrum1D


def flat_spec(w0=6000.0, w1=8000.0, n=2000, level=5e-16):
    wave = np.linspace(w0, w1, n)
    flux = np.full(n, level)
    ivar = np.full(n, 1 / (0.01 * level) ** 2)
    return Spectrum1D(wave, flux, ivar, unit="erg/s/cm2/A", meta={"airmass": 1.3, "exptime": 100.0, "resolution": 900.0})


def test_bands_and_coverage():
    wave = np.linspace(3600, 7000, 100)
    assert telluric.covered_bands(wave, np.ones(100, bool)) == [(6270.0, 6330.0), (6860.0, 6960.0)]
    assert telluric.covered_bands(np.linspace(3600, 6000, 50), np.ones(50, bool)) == []
    m = telluric.in_bands(wave, [(6860, 6960)], pad=10)
    assert wave[m].min() >= 6850 and wave[m].max() <= 6970


def test_none_and_apply_transmission():
    s = flat_spec()
    r = telluric.correct(s, method="none")
    assert r.method == "none" and np.all(r.transmission == 1) and np.allclose(r.spectrum.flux, s.flux)
    trans = np.where((s.wave > 7600) & (s.wave < 7700), 0.5, 1.0)
    out = telluric.apply_transmission(s, trans, "test")
    m = trans < 1
    assert np.allclose(out.flux[m], 2 * s.flux[m]) and np.allclose(out.ivar[m], s.ivar[m] / 4)
    assert np.allclose(out.flux[~m], s.flux[~m])
    trans[100] = 0.0  # opaque pixel -> masked
    out2 = telluric.apply_transmission(s, trans, "test")
    assert not out2.gpm[100]


def test_no_coverage_returns_unchanged():
    s = Spectrum1D(np.linspace(3600, 6000, 500), np.ones(500), np.ones(500), unit="erg/s/cm2/A",
                   meta={"airmass": 1.2, "exptime": 1.0})
    r = telluric.correct_pypeit(s)
    assert r.method == "none" and "reason" in r.meta


@pytest.mark.network
@pytest.mark.slow
def test_pypeit_fit_recovers_flat_continuum():
    """Absorb a flat spectrum with a pypeit PCA telluric realisation, then fit it back out."""
    from pypeit.core import telluric as _tel
    from pypeit.core.wave import airtovac
    from astropy import units as u

    s = flat_spec(6500.0, 7800.0, 1300)
    wave_vac = airtovac(s.wave * u.AA).value
    td = _tel.read_telluric_pca(telluric.DEFAULT_TELGRID, wave_min=wave_vac.min(), wave_max=wave_vac.max())
    theta = np.concatenate([np.asarray(td["coefs_tell_pca"])[0, :4], [900.0, 0.0, 1.0]])
    model = _tel.eval_telluric(theta, td)
    trans = np.interp(wave_vac, td["wave_grid"][:model.size], model)  # pypeit trims one edge pixel
    rng = np.random.default_rng(1)
    absorbed = Spectrum1D(s.wave, s.flux * trans + rng.normal(0, 0.01 * 5e-16, s.wave.size), s.ivar,
                          unit=s.unit, meta=s.meta)
    r = telluric.correct_pypeit(absorbed, polyorder=2)
    assert r.method == "pypeit" and r.meta["tell_success"]
    band = telluric.in_bands(s.wave, [(6860, 6960), (7580, 7700)])
    before = np.std(absorbed.flux[band] / 5e-16)
    after = np.std(r.spectrum.flux[band] / 5e-16)
    assert after < 0.5 * before
    assert abs(np.median(r.spectrum.flux[band]) / 5e-16 - 1) < 0.05
