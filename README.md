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
| Publication to SkyPortal/Fritz | `POST /api/spectrum` (`saltrss.skyportal`) |

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
needs `TYPESAFE_API_KEY` or `JEV_API_KEY`), `--include-mask`, `--upload` (see
below). `saltrss-list DIR [--standards]` shows frames and
configurations; `saltrss-sensfunc` builds a reusable sensitivity function;
`saltrss-upload` posts a finished spectrum to SkyPortal.

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

## Uploading to SkyPortal / Fritz

A finished spectrum can be posted to a SkyPortal instance (default
`https://fritz.science/`) as soon as the reduction is done:

```bash
export FRITZ_TOKEN=<your API token>        # in ~/.bash_profile
saltrss-reduce /path/to/<object>/<night>_RSS -o out/ --upload --group "SALT Transients"
```

or later, from the product file:

```bash
saltrss-upload out/AT2025vjw_*.fits --group "SALT Transients" --sitewide
```

The upload shows exactly what it is about to post and asks for confirmation
(`--dry-run` stops after showing it, `--yes` skips the question — required when
stdin is not a terminal). `saltrss-upload --list-groups`, `--list-instruments`
and `--whoami` answer the questions you need to fill the other options. The
existence of the source is checked before anything is posted: SALT's `OBJECT`
keyword is the TNS name, which is often not the instance's source name
(`AT2025vjw` is `ZTF25ablxhsq` on Fritz), so a cone search around the frame
coordinates names the candidates — `--resolve-obj-id` takes the match when there
is exactly one, otherwise pick it with `--obj-id`.

SALT's instrument on Fritz is called simply `Spectrograph` (telescope `SALT`),
not `RSS`, so the default `--instrument SALT/Spectrograph` is qualified by the
telescope; a bare name works too when it is unique on the instance.

| Option | Meaning |
| --- | --- |
| `--endpoint URL` | SkyPortal instance, default `https://fritz.science/` |
| `--group NAME_OR_ID` | share with this group; repeat for several. Names, nicknames and ids all work. Without any, SkyPortal applies the token owner's default sharing groups |
| `--sitewide` | *also* share with the instance's sitewide (public) group — visible to **every** user |
| `--obj-id` | source name on SkyPortal; defaults to the `OBJECT` header keyword |
| `--resolve-obj-id`, `--match-radius` | when that name is not a source on the instance, take the one within `--match-radius` (3″) of the frame coordinates |
| `--instrument` / `--instrument-id` | instrument, as `<telescope>/<instrument>` or a bare name (default `SALT/Spectrograph`, which is id 40 on Fritz), or its id |
| `--spectrum-type`, `--label`, `--origin` | SkyPortal spectrum type (e.g. `source`), plot-legend label, origin string |
| `--reduced-by`, `--observed-by`, `--pi` | credit SkyPortal users (`me` resolves to the token owner); an `--external-…` free-text name needs one of these as point of contact |
| `--upload-frames` | post each frame instead of their inverse-variance combination (the default when frames of one configuration were combined) |

Masked pixels (CCD gaps, opaque telluric bands) are dropped rather than sent as
NaN, points go up in wavelength, and `units` is set to `erg/s/cm/cm/AA`. The
provenance of the reduction — configuration, exposure, standard star and its
zeropoint rms, telluric method, E(B-V), trace row, and that the wavelengths are
**air** — travels with the spectrum in `altdata`.

### The token

The token is read from `$FRITZ_TOKEN` (then `$SKYPORTAL_TOKEN`, or
`--token-file`). There is deliberately **no `--token` option**, so it cannot end
up in your shell history or in `ps` output. It is sent only in the
`Authorization` header, and never appears in a log line, an error message, a
`repr`, or any output product; a non-HTTPS endpoint is refused outright
(`--allow-insecure` is for a local test instance). Keep it out of the
repository: `saltrss` never writes it anywhere.

```python
from saltrss.skyportal import UploadOptions
import saltrss

saltrss.reduce("/data/SALT/AT2025vjw/20250925_RSS", output="out/",
               upload=UploadOptions(groups=["SALT Transients"], sitewide=False))
```

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
  io.py         multi-extension FITS output, product read-back
  skyportal.py  SkyPortal/Fritz client, payload building, upload
  cli.py        saltrss-reduce / saltrss-sensfunc / saltrss-list / saltrss-upload
  data/suth_extinct.dat   Sutherland extinction (PySALT, BSD)
```
