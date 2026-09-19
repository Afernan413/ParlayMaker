"""FastAPI app serving the parlay crafter UI.

Run it with:

    python -m src.web.app --mock            # cached fixtures, no API credits
    python -m src.web.app --live --port 8080

The API is deliberately thin: every number comes from :mod:`src.web.service`,
which calls the same engine as the CLI.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config.settings import settings
from src.ingestion.odds_api import QuotaExhaustedError
from src.web import service
from src.web.service import SlateNotFound, SlateStore, UnknownLeg

logger = logging.getLogger("parlay_engine.web")

STATIC_DIR = Path(__file__).parent / "static"


class SlipRequest(BaseModel):
    """A bet slip the user is assembling."""

    sport: str = "nfl"
    leg_ids: list[str] = Field(default_factory=list)
    stake: float = 10.0
    bankroll: float | None = None
    seed: int | None = None


class BuildRequest(BaseModel):
    """Ask the optimizer for its own card."""

    sport: str = "nfl"
    legs: int | None = None
    max_tickets: int | None = None
    bankroll: float | None = None
    seed: int | None = None


def create_app(
    *, db_path: str | None = None, use_mock: bool = True, store: SlateStore | None = None
) -> FastAPI:
    """Build the ASGI app. ``use_mock`` decides live vs cached data sources."""
    app = FastAPI(
        title="FanDuel Parlay Crafter",
        description="Assemble parlays and see the model's price, edge and payout.",
        version="0.1.0",
    )
    app.state.store = store or SlateStore(db_path=db_path, use_mock=use_mock)

    @app.exception_handler(SlateNotFound)
    async def _slate_missing(_request: Request, exc: SlateNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(UnknownLeg)
    async def _unknown_leg(_request: Request, exc: UnknownLeg) -> JSONResponse:
        return JSONResponse(
            status_code=409,
            content={
                "detail": f"leg {exc} is not on the current slate; refresh the slate",
                "code": "stale_leg",
            },
        )

    @app.get("/api/config")
    async def config() -> dict[str, Any]:
        """Engine thresholds, so the UI can show the rules it is applying."""
        return {
            "engine": service.engine_config(),
            "mock": app.state.store.use_mock,
            "iterations": app.state.store.iterations,
        }

    @app.get("/api/slate/{sport}")
    async def slate(
        sport: str,
        refresh: bool = Query(False, description="rebuild from the data sources"),
    ) -> dict[str, Any]:
        """Games and every priced leg for a sport."""
        try:
            built = await app.state.store.get(sport, refresh=refresh)
        except QuotaExhaustedError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except (FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return service.slate_payload(built)

    @app.post("/api/price")
    async def price(slip: SlipRequest) -> dict[str, Any]:
        """Price the slip: odds, model probability, edge, payout and staking."""
        await app.state.store.get(slip.sport)
        legs = app.state.store.legs(slip.sport, slip.leg_ids)
        return service.price_slip(
            legs,
            stake=slip.stake,
            bankroll=slip.bankroll,
            iterations=app.state.store.iterations,
            seed=slip.seed,
        )

    @app.post("/api/build")
    async def build(request: BuildRequest) -> dict[str, Any]:
        """The optimizer's own card for this slate."""
        built = await app.state.store.get(request.sport)
        return service.auto_build(
            built,
            legs=request.legs,
            max_tickets=request.max_tickets,
            bankroll=request.bankroll,
            iterations=app.state.store.iterations,
            seed=request.seed,
        )

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "mock": app.state.store.use_mock,
            "slates": {
                sport: service.slate_meta(slate)
                for sport, slate in app.state.store._slates.items()
            },
        }

    if STATIC_DIR.exists():
        @app.get("/")
        async def index() -> FileResponse:
            return FileResponse(STATIC_DIR / "index.html")

        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m src.web.app", description="Serve the parlay crafter UI."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--mock", dest="use_mock", action="store_true", default=True,
        help="use cached fixtures (default; spends no API credits)",
    )
    source.add_argument(
        "--live", dest="use_mock", action="store_false",
        help="pull live odds, weather, injuries and stats",
    )
    parser.add_argument("--db", dest="db_path", default=None)
    parser.add_argument("--reload", action="store_true", help="uvicorn autoreload")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not args.use_mock and not settings.odds_api_key:
        print("ODDS_API_KEY is not set; starting in mock mode instead.")
        args.use_mock = True

    app = create_app(db_path=args.db_path, use_mock=args.use_mock)
    mode = "mock fixtures" if args.use_mock else "live data"
    print(f"Parlay crafter on http://{args.host}:{args.port} ({mode})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
