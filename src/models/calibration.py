"""Learned corrections to the projection model, and the file they live in.

Three corrections are stored per sport and market. They compose in the order
they are applied:

1. ``mean_factor`` -- the projection's systematic bias. If the model says 60
   rushing yards and the position really averages 57, the factor is 0.95.
2. ``dispersion``  -- how wide the real spread is, measured from residuals,
   replacing the prior guesses in ``config/bookmaker_keys.json``. A prior that
   is too tight is what makes a longshot leg look cheaper than it is.
3. ``platt_a`` / ``platt_b`` -- a logistic recalibration of whatever
   miscalibration survives the first two, fitted on the probability scale.

``family`` is an override: a count market priced as Poisson is promoted to a
Negative Binomial when the measured variance rejects the Poisson assumption
that variance equals the mean.

The corrections are fitted by :mod:`src.learning.train` and written to
``data/calibration.json``. Every probability in the engine is produced by
:class:`~src.models.distributions.DistributionSpec`, so loading that one file
corrects the whole pipeline -- the browser included, because the static bundle
exports the already-corrected ``p_model`` per leg.

This module deliberately depends on nothing but the config and numpy, so that
``distributions`` can import it without a cycle.
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from config.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: Where fitted corrections are read from and written to.
CALIBRATION_PATH = PROJECT_ROOT / "data" / "calibration.json"

#: File format version. Bumped when a field's meaning changes.
FORMAT_VERSION = 1

#: Guard rails. A fit outside these is more likely a data problem than a real
#: bias, so it is clipped rather than trusted.
MEAN_FACTOR_BOUNDS = (0.70, 1.40)
PLATT_SLOPE_BOUNDS = (0.25, 4.00)
PLATT_INTERCEPT_BOUNDS = (-3.0, 3.0)

#: Below this many residuals a market's fit is noise, so it is left alone.
MIN_SAMPLES = 60


def logit(p: np.ndarray | float) -> np.ndarray:
    """Log-odds, with the ends pulled in so infinities cannot appear."""
    clipped = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def expit(x: np.ndarray | float) -> np.ndarray:
    """Inverse of :func:`logit`."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


@dataclass(frozen=True)
class MarketCalibration:
    """Everything learned about one sport's market."""

    sport: str
    market: str
    samples: int = 0
    mean_factor: float = 1.0
    dispersion: float | None = None
    family: str | None = None
    platt_a: float = 1.0
    platt_b: float = 0.0
    #: Measured variance / mean for count markets, kept for the report.
    variance_multiple: float | None = None

    @property
    def shifts_mean(self) -> bool:
        return abs(self.mean_factor - 1.0) > 1e-9

    @property
    def shifts_probability(self) -> bool:
        return abs(self.platt_a - 1.0) > 1e-9 or abs(self.platt_b) > 1e-9

    def adjust_mean(self, mean: float) -> float:
        """Apply the fitted bias to a projected mean."""
        return float(mean) * self.mean_factor

    def adjust_probability(self, probability: float) -> float:
        """Apply the fitted logistic recalibration to a probability."""
        if not self.shifts_probability:
            return float(probability)
        corrected = expit(self.platt_a * logit(probability) + self.platt_b)
        return float(np.clip(corrected, 1e-9, 1 - 1e-9))

    def clipped(self) -> "MarketCalibration":
        """Pull every fitted parameter back inside its guard rails."""
        return replace(
            self,
            mean_factor=float(np.clip(self.mean_factor, *MEAN_FACTOR_BOUNDS)),
            platt_a=float(np.clip(self.platt_a, *PLATT_SLOPE_BOUNDS)),
            platt_b=float(np.clip(self.platt_b, *PLATT_INTERCEPT_BOUNDS)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "samples": int(self.samples),
            "mean_factor": round(self.mean_factor, 6),
            "platt_a": round(self.platt_a, 6),
            "platt_b": round(self.platt_b, 6),
        }
        if self.dispersion is not None:
            payload["dispersion"] = round(float(self.dispersion), 6)
        if self.family is not None:
            payload["family"] = self.family
        if self.variance_multiple is not None:
            payload["variance_multiple"] = round(float(self.variance_multiple), 6)
        return payload

    @classmethod
    def from_dict(cls, sport: str, market: str, payload: dict[str, Any]) -> "MarketCalibration":
        return cls(
            sport=sport,
            market=market,
            samples=int(payload.get("samples", 0)),
            mean_factor=float(payload.get("mean_factor", 1.0)),
            dispersion=_optional_float(payload.get("dispersion")),
            family=payload.get("family"),
            platt_a=float(payload.get("platt_a", 1.0)),
            platt_b=float(payload.get("platt_b", 0.0)),
            variance_multiple=_optional_float(payload.get("variance_multiple")),
        )


@dataclass(frozen=True)
class SportCalibration:
    """One sport's markets, plus how the fit that produced them scored."""

    sport: str
    markets: dict[str, MarketCalibration]
    fitted_at: str = ""
    seasons: tuple[int, ...] = ()
    metrics: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metrics is None:
            object.__setattr__(self, "metrics", {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "fitted_at": self.fitted_at,
            "seasons": list(self.seasons),
            "metrics": self.metrics,
            "markets": {key: value.to_dict() for key, value in sorted(self.markets.items())},
        }

    @classmethod
    def from_dict(cls, sport: str, payload: dict[str, Any]) -> "SportCalibration":
        markets = {
            market: MarketCalibration.from_dict(sport, market, body)
            for market, body in (payload.get("markets") or {}).items()
        }
        return cls(
            sport=sport,
            markets=markets,
            fitted_at=str(payload.get("fitted_at", "")),
            seasons=tuple(int(s) for s in payload.get("seasons") or ()),
            metrics=dict(payload.get("metrics") or {}),
        )


@dataclass(frozen=True)
class Calibration:
    """Every sport's corrections: the whole of ``data/calibration.json``."""

    sports: dict[str, SportCalibration]

    @property
    def empty(self) -> bool:
        return not any(sport.markets for sport in self.sports.values())

    def get(self, sport: str | None, market: str) -> MarketCalibration | None:
        """The correction for one market, or ``None`` when nothing was fitted.

        College football falls back to the NFL fit for player markets: the
        market keys are the same and there is far less college history to
        learn from. Game markets never fall back -- a college final score
        swings much wider than an NFL one, so borrowing that spread would be
        worse than the prior.
        """
        if not sport:
            return None
        for candidate in _lookup_order(sport, market):
            fitted = self.sports.get(candidate)
            if fitted is not None and market in fitted.markets:
                return fitted.markets[market]
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": FORMAT_VERSION,
            "sports": {key: value.to_dict() for key, value in sorted(self.sports.items())},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Calibration":
        sports = {
            sport: SportCalibration.from_dict(sport, body)
            for sport, body in (payload.get("sports") or {}).items()
        }
        return cls(sports=sports)

    @classmethod
    def blank(cls) -> "Calibration":
        return cls(sports={})

    def merged_with(self, other: "Calibration") -> "Calibration":
        """``other`` wins per sport; sports it does not mention are kept."""
        sports = dict(self.sports)
        sports.update(other.sports)
        return Calibration(sports=sports)

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path or CALIBRATION_PATH)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n")
        return target

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Calibration":
        source = Path(path or CALIBRATION_PATH)
        if not source.exists():
            return cls.blank()
        try:
            payload = json.loads(source.read_text())
        except (OSError, json.JSONDecodeError):
            logger.warning("calibration file at %s is unreadable; ignoring it", source)
            return cls.blank()
        version = int(payload.get("version", 0))
        if version != FORMAT_VERSION:
            logger.warning(
                "calibration file is version %s, expected %s; ignoring it",
                version,
                FORMAT_VERSION,
            )
            return cls.blank()
        return cls.from_dict(payload)


#: Sports that borrow another sport's fit when they have none of their own.
FALLBACK_SPORT: dict[str, str] = {"ncaaf": "nfl"}

#: Markets whose fit is league-specific and must never be borrowed.
NO_FALLBACK_MARKETS = frozenset({"spreads", "totals", "h2h"})


def _lookup_order(sport: str, market: str) -> tuple[str, ...]:
    fallback = FALLBACK_SPORT.get(sport)
    if fallback is None or market in NO_FALLBACK_MARKETS:
        return (sport,)
    return (sport, fallback)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------
# the process-wide active calibration
# ----------------------------------------------------------------------
_ACTIVE: Calibration | None = None


def active_calibration() -> Calibration:
    """The corrections in force, loaded from disk on first use."""
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = Calibration.load()
        if not _ACTIVE.empty:
            fitted = ", ".join(
                f"{sport}:{len(body.markets)}" for sport, body in sorted(_ACTIVE.sports.items())
            )
            logger.info("loaded learned calibration (%s)", fitted)
    return _ACTIVE


def set_active_calibration(calibration: Calibration | None) -> None:
    """Install corrections directly, or pass ``None`` to reload from disk."""
    global _ACTIVE
    _ACTIVE = calibration


def calibration_for(sport: str | None, market: str) -> MarketCalibration | None:
    """Shorthand used by :mod:`src.models.distributions`."""
    return active_calibration().get(sport, market)


@contextmanager
def using_calibration(calibration: Calibration | None):
    """Run a block with ``calibration`` in force, then restore what was there.

    Training uses this to score a candidate fit through the very code path
    production uses, rather than a second implementation of it.
    """
    global _ACTIVE
    previous = _ACTIVE
    _ACTIVE = calibration if calibration is not None else Calibration.blank()
    try:
        yield
    finally:
        _ACTIVE = previous
