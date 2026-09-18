import numpy as np

from saltrss.products import FrameInfo, ProductDir, RSSConfig, parse_steps, _strip_flat_key


def test_parse_steps():
    assert parse_steps("mbxgpP202510260083_bp_ag_ff_cr_cg_wr.fits") == ["bp", "ag", "ff", "cr", "cg", "wr"]
    assert parse_steps("mbxgpP202510260084_wr.fits") == ["wr"]
    assert parse_steps("mbxgpP202510260084.fits") == []
    assert parse_steps("mbxgpP20250621FlatNM2x2FASLPC03400PG0900GA13.625CA27.25.fits") == []


def test_config_key_roundtrip():
    c = RSSConfig.from_key("NM2x2PG0900GA13.625CA27.25BV44177")
    assert c.grating == "PG0900" and c.gr_angle == 13.625 and c.cam_angle == 27.25
    assert c.xbin == 2 and c.ybin == 2
    assert c.key == "NM2x2PG0900GA13.625CA27.25"
    assert RSSConfig.from_key("NM2x2PG0700GA4.6CA22.75").key == "NM2x2PG0700GA4.6CA22.75"


def test_config_matching_tolerances():
    a = RSSConfig("PG0700", 4.6, 22.75, 2, 2)
    b = RSSConfig("PG0700", 4.5975, 22.75, 2, 4)
    assert a.matches(b)                       # angle within tolerance, spatial binning ignored
    assert not a.matches(b, require_ybin=True)
    assert not a.matches(RSSConfig("PG0900", 4.6, 22.75, 2, 2))
    assert not a.matches(RSSConfig("PG0700", 4.6, 22.75, 4, 2))  # spectral binning differs


def test_strip_flat_key():
    assert _strip_flat_key("NM2x2FASLPC03400PG0900GA13.625CA27.25BV44177") == "NM2x2PG0900GA13.625CA27.25BV44177"


def test_product_dir_discovery(synthetic_product):
    pd = ProductDir(synthetic_product)
    names = sorted(f.path.name for f in pd.science_frames)
    assert names == ["mbxgpP202501010011_bp_ag_ff_cr_cg_wr.fits", "mbxgpP202501010013_bp_ag_ff_cr_cg_wr.fits"]
    assert len(pd.arc_frames) == 2
    sci = pd.science_frames[0]
    assert sci.is_rectified and sci.is_science and not sci.is_standard
    assert sci.slit_width_arcsec == 1.5
    assert pd.flat_for(sci.config).name.startswith("mbxgpP20250101Flat")
    assert pd.gaps_for(sci.config) == [(150, 156)]
    assert pd.gain_corrections_for(sci.config)[0]["correction"] == 1.01
    assert pd.bpm_file.name == "mRSSBpm2x2.fits"
    std = FrameInfo.from_file(synthetic_product / "mbxgpP202501010010.fits")
    assert std.is_standard and std.slit_width_arcsec == 4.0
    assert np.isclose(std.ra_deg, 47.629, atol=0.01) and np.isclose(std.dec_deg, -68.6006, atol=0.01)
