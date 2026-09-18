"""Discovery and parsing of SALT RSS longslit pipeline product directories.

A *product directory* is what SALT delivers for one night: basic-reduced
frames ``mbxgpP<date><seq>.fits``, pipeline products with processing-step
suffixes (``_bp_ag_ff_cr_cg_wr``), combined flats, a bad-pixel mask and the
``*_spec_ls_img.json`` reduction dictionaries.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy import units as u

# Pipeline suffix meaning (from https://astronomers.salt.ac.za/software/rss-pipeline/)
STEP_NAMES = {"bp": "bad pixel", "ag": "auto gain", "ff": "flat field",
              "cr": "cosmic ray", "cg": "ccd gap", "wr": "wavelength rectified"}

_CONFIG_RE = re.compile(
    r"NM(?P<xbin>\d)x(?P<ybin>\d)(?P<grating>PG\d{4})GA(?P<grang>[\d.]+)CA(?P<camang>[\d.]+)(?P<bv>BV\d+)?$")


@dataclass(frozen=True)
class RSSConfig:
    """Spectroscopic configuration that determines the wavelength solution."""

    grating: str
    gr_angle: float
    cam_angle: float
    xbin: int
    ybin: int
    filter: str = ""

    @property
    def key(self) -> str:
        """Key in the same style as the pipeline JSON dictionaries (no BV suffix)."""
        return f"NM{self.xbin}x{self.ybin}{self.grating}GA{self.gr_angle:g}CA{self.cam_angle:g}"

    def matches(self, other: "RSSConfig", angle_tol: float = 0.05,
                require_ybin: bool = False) -> bool:
        """True if ``other`` shares grating, angles (within ``angle_tol`` deg)
        and spectral binning.  Spatial binning only matters if ``require_ybin``."""
        return (self.grating == other.grating
                and abs(self.gr_angle - other.gr_angle) <= angle_tol
                and abs(self.cam_angle - other.cam_angle) <= angle_tol
                and self.xbin == other.xbin
                and (not require_ybin or self.ybin == other.ybin))

    @classmethod
    def from_key(cls, key: str) -> "RSSConfig":
        m = _CONFIG_RE.match(key.strip())
        if m is None:
            raise ValueError(f"Not a recognised RSS configuration key: {key!r}")
        return cls(m["grating"], float(m["grang"]), float(m["camang"]),
                   int(m["xbin"]), int(m["ybin"]))


def _sexagesimal_to_deg(ra: str, dec: str) -> tuple[float, float]:
    c = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
    return float(c.ra.deg), float(c.dec.deg)


@dataclass
class FrameInfo:
    """Header metadata of one RSS frame that the pipeline needs."""

    path: Path
    object: str
    obstype: str
    propid: str
    config: RSSConfig
    exptime: float
    airmass: float
    ra_deg: float
    dec_deg: float
    date_obs: str
    jd: float
    maskid: str
    masktype: str
    steps: tuple[str, ...] = ()
    shape: tuple[int, int] | None = None
    header: fits.Header = field(default=None, repr=False)

    @property
    def is_rectified(self) -> bool:
        return "wr" in self.steps

    @property
    def is_arc(self) -> bool:
        return self.obstype.upper() == "ARC" or self.header.get("LAMPID", "NONE").strip().upper() not in ("NONE", "")

    @property
    def is_science(self) -> bool:
        return self.obstype.upper() == "OBJECT" and not self.is_arc

    @property
    def is_standard(self) -> bool:
        return self.propid.upper().startswith("CAL_SPST")

    @property
    def slit_width_arcsec(self) -> float | None:
        """Slit width from the long-slit mask barcode, e.g. ``PL0150N004`` -> 1.5''."""
        m = re.match(r"PL(\d{4})", self.maskid or "")
        return int(m.group(1)) / 100.0 if m else None

    @property
    def night(self) -> str:
        return self.date_obs.replace("-", "")

    @classmethod
    def from_header(cls, header: fits.Header, path: Path | str, shape=None) -> "FrameInfo":
        path = Path(path)
        ccdsum = str(header.get("CCDSUM", "2 2")).split()
        cfg = RSSConfig(grating=str(header["GRATING"]).strip(),
                        gr_angle=float(header["GR-ANGLE"]),
                        cam_angle=float(header["CAMANG"]),
                        xbin=int(ccdsum[0]), ybin=int(ccdsum[1]),
                        filter=str(header.get("FILTER", "")).strip())
        try:
            ra, dec = _sexagesimal_to_deg(str(header["RA"]), str(header["DEC"]))
        except Exception:  # pragma: no cover - malformed coordinates
            ra = dec = float("nan")
        steps = tuple(parse_steps(path.name))
        return cls(path=path, object=str(header.get("OBJECT", "")).strip(),
                   obstype=str(header.get("OBSTYPE", "")).strip(),
                   propid=str(header.get("PROPID", "")).strip(), config=cfg,
                   exptime=float(header.get("EXPTIME", 0.0)),
                   airmass=float(header.get("AIRMASS", 1.0)),
                   ra_deg=ra, dec_deg=dec, date_obs=str(header.get("DATE-OBS", "")).strip(),
                   jd=float(header.get("JD", 0.0)), maskid=str(header.get("MASKID", "")).strip(),
                   masktype=str(header.get("MASKTYP", "")).strip(), steps=steps, shape=shape,
                   header=header)

    @classmethod
    def from_file(cls, path: Path | str) -> "FrameInfo":
        with fits.open(path) as hdul:
            shape = None
            for hdu in hdul[1:]:
                if hdu.data is not None and hdu.data.ndim == 2:
                    shape = tuple(hdu.data.shape)
                    break
            return cls.from_header(hdul[0].header, path, shape=shape)


_FRAME_RE = re.compile(r"^mbxgpP(?P<date>\d{8})(?P<seq>\d{4})(?P<steps>(?:_[a-z]{2})*)\.fits$")


def parse_steps(filename: str) -> list[str]:
    """Return the pipeline step suffixes in a product filename, e.g.
    ``['bp','ag','ff','cr','cg','wr']``."""
    m = _FRAME_RE.match(Path(filename).name)
    if m is None:
        return []
    return [s for s in m["steps"].split("_") if s]


def is_frame_file(filename: str) -> bool:
    return _FRAME_RE.match(Path(filename).name) is not None


class ProductDir:
    """Index of one SALT RSS product directory.

    Parameters
    ----------
    path : str or Path
        Directory containing the ``mbxgpP*`` files and JSON dictionaries.
    """

    JSON_NAMES = {
        "wavelength": "wavelength_solutions_nm_spec_ls_img.json",
        "gaps": "ccd_gaps_spec_ls_img.json",
        "gain": "auto_gain_correct_spec_ls_img.json",
        "flats": "combined_flats_spec_ls_img.json",
    }

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.is_dir():
            raise FileNotFoundError(f"Product directory not found: {self.path}")
        self._frames: dict[Path, FrameInfo] | None = None

    # ---------------------------------------------------------------- files
    @property
    def frame_files(self) -> list[Path]:
        return sorted(p for p in self.path.iterdir() if is_frame_file(p.name))

    @property
    def frames(self) -> list[FrameInfo]:
        if self._frames is None:
            self._frames = {p: FrameInfo.from_file(p) for p in self.frame_files}
        return list(self._frames.values())

    @property
    def science_frames(self) -> list[FrameInfo]:
        """Fully processed, rectified science frames (``..._wr.fits`` with OBSTYPE OBJECT)."""
        return [f for f in self.frames if f.is_rectified and f.is_science]

    @property
    def arc_frames(self) -> list[FrameInfo]:
        return [f for f in self.frames if f.is_arc]

    @property
    def flat_files(self) -> list[Path]:
        return sorted(self.path.glob("mbxgpP*Flat*.fits"))

    @property
    def bpm_file(self) -> Path | None:
        cands = sorted(self.path.glob("mRSSBpm*.fits"))
        return cands[0] if cands else None

    def json(self, kind: str) -> dict:
        """Load one of the reduction dictionaries (``wavelength``, ``gaps``, ``gain``, ``flats``)."""
        p = self.path / self.JSON_NAMES[kind]
        if not p.exists():
            raise FileNotFoundError(f"{p.name} not found in {self.path}")
        with open(p) as fh:
            return json.load(fh)

    def has_json(self, kind: str) -> bool:
        return (self.path / self.JSON_NAMES[kind]).exists()

    # ------------------------------------------------------- calibrations
    def flat_for(self, config: RSSConfig) -> Path | None:
        """Combined flat matching ``config`` (via the flats dictionary, then by filename)."""
        if self.has_json("flats"):
            db = self.json("flats")
            for section in ("wrk", "db"):
                for key, fname in db.get(section, {}).items():
                    try:
                        cfg = RSSConfig.from_key(_strip_flat_key(key))
                    except ValueError:
                        continue
                    if cfg.matches(config, require_ybin=True) and (self.path / fname).exists():
                        return self.path / fname
        for p in self.flat_files:
            m = re.search(r"Flat(NM\d+x\d+)FA\w+?(PG\d{4}GA[\d.]+CA[\d.]+)", p.name)
            if m:
                try:
                    cfg = RSSConfig.from_key(m.group(1) + m.group(2))
                except ValueError:
                    continue
                if cfg.matches(config, require_ybin=True):
                    return p
        return None

    def gaps_for(self, config: RSSConfig) -> list[tuple[int, int]]:
        """CCD gap column ranges ``[(x0, x1), ...]`` for the binning of ``config``."""
        if not self.has_json("gaps"):
            return []
        db = self.json("gaps")
        binkey = f"{config.xbin}x{config.ybin}"
        for section in ("wrk", "db"):
            entry = db.get(section, {}).get(binkey)
            if entry:
                return [(int(v["x0"]), int(v["x1"])) for v in entry.values()]
        return []

    def gain_corrections_for(self, config: RSSConfig) -> list[dict]:
        """Per-amplifier auto-gain corrections ``[{amp, x0, x1, correction}, ...]``."""
        if not self.has_json("gain"):
            return []
        db = self.json("gain")
        for section in ("db", "wrk"):
            for key, entries in db.get(section, {}).items():
                try:
                    cfg = RSSConfig.from_key(_strip_flat_key(key))
                except ValueError:
                    continue
                if cfg.matches(config, require_ybin=True):
                    return list(entries)
        return []

    def __repr__(self) -> str:
        return f"ProductDir({str(self.path)!r}, n_frames={len(self.frame_files)})"


def _strip_flat_key(key: str) -> str:
    """``NM2x2FASLPC03400PG0900GA13.625CA27.25BV44177`` -> ``NM2x2PG0900GA13.625CA27.25BV44177``."""
    return re.sub(r"FA\w+?(?=PG\d{4})", "", key)


def scan_frames(paths: Iterable[Path | str]) -> list[FrameInfo]:
    """Read headers of many frame files, skipping unreadable ones."""
    import warnings

    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p in paths:
            try:
                out.append(FrameInfo.from_file(p))
            except Exception:
                continue
    return out


def frame_is_readable(path: Path | str) -> bool:
    """True if the SCI data of a frame can be fully read (catches truncated downloads)."""
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=Warning)
            with fits.open(path) as hdul:
                data = hdul[1].data
                return data is not None and np.isfinite(data[-1, -1]) is not None
    except Exception:
        return False
