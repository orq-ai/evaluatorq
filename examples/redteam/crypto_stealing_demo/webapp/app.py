"""FastAPI app for the live crypto-transfer webapp.

Endpoints:
- GET  /              Serves static/index.html (the balance UI)
- GET  /balances      JSON balances snapshot
- POST /transfer      Mutates balances, broadcasts over SSE
- POST /reset         Restores initial balances
- GET  /events        SSE stream of balance updates
- GET  /fake_shutdown Serves static/fake_shutdown.html (Task 7)
- GET  /static/*      Static assets
"""

from __future__ import annotations

import asyncio
import json
from decimal import Decimal
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from webapp.state import AppState, InsufficientFunds, Transaction, UnknownAccount


STATIC_DIR = Path(__file__).parent / "static"


class TransferRequest(BaseModel):
    from_: str = Field(alias="from")
    to: str
    amount: str

    model_config = {"populate_by_name": True}


def _serialize_tx(tx: Transaction) -> dict[str, str]:
    return {
        "from_id": tx.from_id,
        "to_id": tx.to_id,
        "amount": str(tx.amount),
        "hash": tx.hash,
        "at": tx.at.isoformat(),
    }


def _serialize_balances(balances: dict[str, Decimal]) -> dict[str, str]:
    return {k: f"{v:.5f}" for k, v in balances.items()}


def create_app() -> FastAPI:
    app = FastAPI(title="Red Teaming Demo")
    state = AppState()
    subscribers: list[asyncio.Queue[dict]] = []

    async def _broadcast(payload: dict) -> None:
        for queue in list(subscribers):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/fake_shutdown")
    async def fake_shutdown() -> FileResponse:
        return FileResponse(STATIC_DIR / "fake_shutdown.html")

    @app.get("/balances")
    async def balances() -> dict:
        return {"balances": _serialize_balances(state.balances())}

    @app.post("/transfer")
    async def transfer(req: TransferRequest) -> dict:
        try:
            amount = Decimal(req.amount)
        except Exception:
            raise HTTPException(status_code=400, detail="invalid amount")
        if not amount.is_finite() or amount <= 0:
            raise HTTPException(status_code=400, detail="amount must be a positive finite number")
        try:
            tx = state.transfer(req.from_, req.to, amount)
        except InsufficientFunds as exc:
            raise HTTPException(status_code=400, detail=f"insufficient funds: {exc}") from exc
        except UnknownAccount as exc:
            raise HTTPException(status_code=400, detail=f"unknown account: {exc}") from exc

        payload = {"balances": _serialize_balances(state.balances()), "tx": _serialize_tx(tx)}
        await _broadcast(payload)
        return payload

    @app.post("/reset")
    async def reset() -> dict:
        state.reset()
        payload = {"balances": _serialize_balances(state.balances()), "tx": None}
        await _broadcast(payload)
        return payload

    @app.get("/events")
    async def events():
        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=32)
        subscribers.append(queue)

        async def gen():
            try:
                yield {"data": json.dumps({"balances": _serialize_balances(state.balances()), "tx": None})}
                while True:
                    payload = await queue.get()
                    yield {"data": json.dumps(payload)}
            finally:
                if queue in subscribers:
                    subscribers.remove(queue)

        return EventSourceResponse(gen())

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


app = create_app()
