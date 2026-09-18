"""saltrss: reduce SALT RSS longslit pipeline products to calibrated 1D spectra.

The package takes the wavelength-rectified 2D products written by the SAAO RSS
longslit pipeline (``*_bp_ag_ff_cr_cg_wr.fits``) together with the sidecar
calibration dictionaries (``*_spec_ls_img.json``) and produces a
sky-subtracted, flux-calibrated, telluric-corrected (and optionally
dereddened) 1D spectrum as a CSV with ``wavelength, flux, fluxerr``.

Heavy lifting is delegated to well-tested libraries:

* `specreduce` -- tracing, background (sky) estimation, Horne extraction
* `pypeit`     -- standard-star archive, zeropoint (sensitivity) fitting,
                  atmospheric extinction handling, PCA telluric fitting
* `astroscrappy` -- cosmic-ray cleaning of standard-star frames
* `dustmaps` + `extinction` -- Milky Way reddening correction
"""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("saltrss")
except PackageNotFoundError:  # pragma: no cover - source checkout
    __version__ = "0.0.0+src"

from .pipeline import reduce, ReductionResult  # noqa: E402,F401
from .spectrum import Spectrum1D  # noqa: E402,F401

__all__ = ["reduce", "ReductionResult", "Spectrum1D", "__version__"]
