from astropy.table import Table

from saltrss import cli


def test_list(synthetic_product, capsys):
    assert cli.main_list([str(synthetic_product)]) == 0
    out = capsys.readouterr().out
    assert "mbxgpP202501010011_bp_ag_ff_cr_cg_wr.fits" in out and "science" in out and "rectified" in out
    assert cli.main_list([str(synthetic_product), "--standards"]) == 0
    assert "STAR" in capsys.readouterr().out


def test_reduce_cli(synthetic_product, tmp_path, reference_spectrum, monkeypatch, capsys):
    from pypeit.core import standard as _std

    monkeypatch.setattr(_std, "get_standard_spectrum", lambda **kw: reference_spectrum)
    sens = tmp_path / "sens.fits"
    assert cli.main_sensfunc([str(synthetic_product), "-o", str(sens), "--standards-dir", str(synthetic_product)]) == 0
    assert sens.exists()
    out = tmp_path / "spec.csv"
    rc = cli.main_reduce([str(synthetic_product), "--science-file",
                          str(synthetic_product / "mbxgpP202501010011_bp_ag_ff_cr_cg_wr.fits"),
                          "-o", str(out), "--sensfunc", str(sens), "--telluric", "none", "--no-fits", "--no-qa",
                          "--include-mask", "--trace-rows", "45:75", "--sky-rows", "25:40", "--sky-rows", "80:95"])
    assert rc == 0 and out.exists()
    tab = Table.read(out, format="ascii.csv")
    assert tab.colnames == ["wavelength", "flux", "fluxerr", "mask"]
    assert "median S/N" in capsys.readouterr().out


def test_rows_argument():
    import argparse

    import pytest

    assert cli._rows("499:504") == (499, 504)
    for bad in ("499", "499-504", "a:b", "1:2:3"):
        with pytest.raises(argparse.ArgumentTypeError):
            cli._rows(bad)
