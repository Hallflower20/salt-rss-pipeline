import numpy as np
import pytest

from saltrss import reddening
from saltrss.spectrum import Spectrum1D


def spec():
    w = np.linspace(3600, 9000, 200)
    return Spectrum1D(w, np.ones_like(w), np.ones_like(w), unit="erg/s/cm2/A")


def test_deredden_identity_and_direction():
    s = spec()
    assert np.allclose(reddening.deredden(s, 0.0).flux, s.flux)
    d = reddening.deredden(s, 0.1)
    assert d.meta["ebv"] == 0.1 and d.meta["extinction_law"] == "fitzpatrick99"
    assert np.all(d.flux > 1) and d.flux[0] > d.flux[-1]           # more correction in the blue
    assert np.allclose(d.ivar, s.ivar / (d.flux / s.flux) ** 2)


def test_extinction_mag_matches_library():
    import extinction as ext

    w = np.array([4000.0, 5500.0, 8000.0])
    a = reddening.extinction_mag(w, 0.2, r_v=3.1)
    assert np.allclose(a, ext.fitzpatrick99(w, 0.62, 3.1))
    a_ccm = reddening.extinction_mag(w, 0.2, r_v=3.1, law="ccm89")
    assert np.allclose(a_ccm, ext.ccm89(w, 0.62, 3.1))
    # A_V = R_V * E(B-V) at V band to a few percent
    assert abs(a[1] / 0.62 - 1) < 0.05


@pytest.mark.network
def test_dustmaps_query():
    # towards the SMC-ish southern sky the SFD reddening is small but positive
    e = reddening.ebv_from_dustmaps(73.22, -1.44)
    assert 0.0 < e < 0.5
