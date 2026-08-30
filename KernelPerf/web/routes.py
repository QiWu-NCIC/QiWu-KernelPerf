from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def install_web_ui(app: FastAPI, static_dir: str | Path, contest_db: Any) -> None:
    """Install only presentation routes; API and scheduling stay outside web/."""
    root = Path(static_dir)
    app.mount("/static", StaticFiles(directory=root), name="static")

    @app.middleware("http")
    async def disable_frontend_caching(request: Request, call_next):
        response = await call_next(request)
        if (
            request.url.path in {"/", "/submit"}
            or request.url.path.startswith("/contest")
            or request.url.path.startswith("/static/")
        ):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(root / "index.html")

    @app.get("/submit")
    def submit() -> FileResponse:
        return FileResponse(root / "submit.html")

    @app.get("/contest")
    def contest_index() -> FileResponse:
        return FileResponse(root / "contest.html")

    @app.get("/contest/{contest_id}")
    def contest_ranking(contest_id: str) -> FileResponse:
        if contest_db.get_contest(contest_id) is None:
            raise HTTPException(status_code=404, detail="contest not found")
        return FileResponse(root / "contest-ranking.html")
