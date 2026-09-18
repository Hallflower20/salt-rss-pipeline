# saltrss — SALT RSS longslit products → calibrated 1D spectra

`saltrss` turns the wavelength-rectified 2D products delivered by the SAAO RSS
longslit pipeline (`mbxgpP*_bp_ag_ff_cr_cg_wr.fits` plus the
`*_spec_ls_img.json` reduction dictionaries) into a sky-subtracted,
flux-calibrated, telluric-corrected and optionally dereddened 1D spectrum,
written as a CSV with `wavelength, flux, fluxerr`.

It is deliberately thin: the numerical work is done by established packages.

| Step | Implementation |
| --- | --- |
| Product discovery, header parsing | `astropy.io.fits` (`saltrss.products`) |
| Wavelength solution / rectification of standards | the pipeline's own JSON solution (`saltrss.wavecal`, conventions verified against the pipeline's `_wr` arcs to < 0.01 px) |
| Object tracing, sky windows, boxcar | `specreduce` (`FitTrace`, `Background`, `BoxcarExtract`) |
| Optimal extraction | Horne (1986) with empirical profile and outlier rejection (`saltrss.extract.horne_extract`) |
| Standard-star reference spectra | `pypeit.core.standard` archives (CALSPEC, ESO, X-shooter, …) matched by coordinates |
| Sensitivity function | `pypeit.core.flux_calib.fit_zeropoint` (b-spline zeropoint, Balmer/telluric masks) |
| Atmospheric extinction | Sutherland curve from PySALT via `pypeit.core.atmextinction` |
| Telluric correction | `pypeit.core.telluric.Telluric` PCA model (site independent) or ESO SkyCalc |
| Cosmic rays (optional) | `astroscrappy` |
| Milky Way reddening | `dustmaps` (SFD/Planck/CSFD) + `extinction` (Fitzpatrick 99, CCM 89, …) |
| QA judgment (optional) | TypeSafe System One (`typesafe-sdk`) |

## Installation

A ready environment lives at `/hildafs/projects/phy220048p/share/envs/salt-rss`
(Python 3.12, pypeit 2.0.1, specreduce 1.9). To recreate it elsewhere:

```bash
conda create -p <prefix> python=3.12 -y
<prefix>/bin/pip install --no-deps pypeit          # pyqt6 wheels may not exist for old glibc
<prefix>/bin/pip install "numpy>=2.4" "astropy>=7" scipy matplotlib PyYAML PyERFA fast-histogram \
    configobj scikit-learn IPython "ginga>=5.5.1" qtpy pygithub bottleneck specutils scikit-image
<prefix>/bin/pip install -e .[test]                # this package
<prefix>/bin/pip install skycalc_ipy typesafe-sdk  # optional
```

Reference files (standard spectra, the 6 MB telluric PCA grid, dust maps) are
downloaded on first use into the pypeit and dustmaps caches.

## Quick start

```bash
export SALTRSS_STANDARDS_DIR=/hildafs/projects/phy220048p/xhall/data/SALT/standards
saltrss-reduce /path/to/<object>/<night>_RSS -o spectrum.csv --deredden
```

This

1. finds the rectified science frame(s) in the product directory;
2. picks the standard-star frame in the library with the same grating / grating
   angle / camera angle / spectral binning closest in time, replays the pipeline
   steps on it with the calibrations of *this* product directory (bad-pixel
   mask, combined flat, CCD-gap mask, wavelength solution), extracts it and fits
   the sensitivity function (cached in `saltrss_work/sensfunc/`);
3. traces the target — the brightest object within 60 rows of where the
   standard star landed on the slit (SALT acquires onto a fixed slit position;
   field stars elsewhere on the slit are ignored) — subtracts the sky estimated
   from windows either side and extracts optimally;
4. flux-calibrates with the Sutherland extinction curve;
5. fits and divides out telluric absorption in the covered O2/H2O bands;
6. (with `--deredden`) queries SFD (×0.86, Schlafly & Finkbeiner 2011) and removes
   Milky Way extinction with Fitzpatrick (1999), R_V = 3.1;
7. writes `spectrum.csv`, `spectrum.fits` (all intermediate spectra, telluric
   model, sky, zeropoint) and `spectrum.png` (QA figure).

Useful options: `--science-file`, `--trace-guess ROW` or
`--target-selection brightest` (which object on the slit), `--standard-file` /
`--sensfunc file.fits`, `--telluric skycalc|none`, `--ebv 0.05`,
`--extract-method boxcar`, `--typesafe` (experimental QA verdict from TypeSafe;
needs `TYPESAFE_API_KEY` or `JEV_API_KEY`), `--include-mask`. `saltrss-list DIR [--standards]` shows frames and
configurations; `saltrss-sensfunc` builds a reusable sensitivity function.

### Python

```python
import saltrss

results = saltrss.reduce("/data/SALT/AT2025abne/20251026_RSS", output="out/",
                         standards_dir="/data/SALT/standards", deredden=True)
spec = results[0].spectrum          # saltrss.Spectrum1D: wave, flux, ivar, gpm, meta
print(results[0].summary())

# building blocks
from saltrss.products import ProductDir
from saltrss.frames import load_science_frame
from saltrss.extract import extract
pd = ProductDir("/data/SALT/AT2025abne/20251026_RSS")
frame = load_science_frame(pd.science_frames[0].path)
counts = extract(frame, trace_guess=1012).spectrum
```

## Output

CSV columns: `wavelength` (Å, **air**, the SALT pipeline's native system),
`flux` and `fluxerr` (erg s⁻¹ cm⁻² Å⁻¹). Bad pixels (CCD gaps, opaque telluric
pixels, masked columns) are written as NaN; `--include-mask` adds a 0/1 column.

## Caveats

* **Only the wavelength axis is rectified.** The SAAO ``_wr`` step aligns arc
  lines along columns; it does not straighten the spatial axis. An object's row
  therefore still drifts with wavelength (atmospheric differential refraction
  plus camera distortion, typically 5-15 pixels across the PG0700 range), which
  is why the fitted trace in the QA figure is curved. The extraction follows it.
* **CCD gaps.** The pipeline's ``_cg`` step interpolates across the two chip
  gaps without flagging them. `saltrss` re-masks those columns (row by row,
  from the gap table and the wavelength solution), so they are NaN in the CSV.

* **Absolute flux scale.** SALT's moving pupil changes the effective aperture
  during a track and standards are taken on other nights through a 4″ slit, so
  the absolute scale can be off by tens of per cent; the spectral *shape* is what
  the calibration delivers. Rescale to photometry if you need absolute fluxes.
* Standards are not delivered wavelength-calibrated. They are rectified with the
  solution of the science night for the same configuration; flexure of a few
  pixels between nights is harmless for a smooth sensitivity function (Balmer
  lines are masked with a 25 Å half-width).
* The blue setups (PG0900 to 6730 Å) only contain the weak O2 γ band, so the
  telluric step has little to do there; the red PG0700 setups reach the B band
  and the 7200 Å water band.
* Uncertainties are propagated from a Poisson + read-noise model of the
  rectified frame (the pipeline does not deliver a variance plane).

## Tests

```bash
pytest                       # synthetic data, offline, ~15 s
pytest -m network            # telluric PCA fit recovery, dustmaps query
pytest -m realdata           # needs the SALT data tree on hildafs
```

## Layout

```
saltrss/
  products.py   product directory index, FrameInfo, RSSConfig
  wavecal.py    WavelengthSolution (JSON Legendre + row-shift), rectify()
  frames.py     Frame2D, load_science_frame(), prepare_standard_frame()
  extract.py    find_trace(), horne_extract(), extract()
  fluxcal.py    standards lookup, SensFunc, build_sensfunc()
  telluric.py   correct() -> pypeit PCA / SkyCalc / none
  reddening.py  ebv_from_dustmaps(), deredden()
  pipeline.py   reduce(), reduce_frame(), get_sensfunc(), combine_spectra()
  qa.py         plot_reduction(), typesafe_review()
  io.py         multi-extension FITS output
  cli.py        saltrss-reduce / saltrss-sensfunc / saltrss-list
  data/suth_extinct.dat   Sutherland extinction (PySALT, BSD)
```
