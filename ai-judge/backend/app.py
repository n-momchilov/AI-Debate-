from __future__ import annotations

import json
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.models.schemas import Argument, CaseInput, DebateTranscript, Verdict, DebateRoles
from backend.utils.debate_coordinator import DebateCoordinator
from backend.agents.emotional_lawyer import EmotionalLawyer
from backend.agents.logical_lawyer import LogicalLawyer
from backend.agents.judge import JudgeAgent


DATA_LOCK = threading.Lock()


def _debates_path() -> Path:
    return Path(settings.DEBATES_FILE)


def _stats_path() -> Path:
    return Path(settings.STATISTICS_FILE)


def _load_debates() -> Dict[str, Any]:
    path = _debates_path()
    if not path.exists():
        return {"cases": {}, "debates": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                # migrate from [] seed to structured object
                return {"cases": {}, "debates": {}}
            if not isinstance(data, dict):
                return {"cases": {}, "debates": {}}
            data.setdefault("cases", {})
            data.setdefault("debates", {})
            return data
    except Exception:
        return {"cases": {}, "debates": {}}


def _save_debates(data: Dict[str, Any]) -> None:
    path = _debates_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_stats() -> Dict[str, int]:
    path = _stats_path()
    if not path.exists():
        return {"emotional_wins": 0, "logical_wins": 0, "total_debates": 0}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {"emotional_wins": 0, "logical_wins": 0, "total_debates": 0}
            data.setdefault("emotional_wins", 0)
            data.setdefault("logical_wins", 0)
            data.setdefault("total_debates", 0)
            return data
    except Exception:
        return {"emotional_wins": 0, "logical_wins": 0, "total_debates": 0}


def _save_stats(data: Dict[str, int]) -> None:
    path = _stats_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def create_app() -> FastAPI:
    app = FastAPI(title="AI Judge - Legal Debate Arena", version="0.1.0")

    # CORS: allow localhost
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://localhost:8000",
            "http://127.0.0.1",
            "http://127.0.0.1:8000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.time()
        try:
            response = await call_next(request)
            return response
        finally:
            dt = (time.time() - t0) * 1000.0
            method = request.method
            path = request.url.path
            status = getattr(response, "status_code", 0) if 'response' in locals() else 0
            print(f"[REQ] {method} {path} -> {status} in {dt:.1f}ms")

    # Error handlers
    @app.exception_handler(Exception)
    async def handle_ex(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    # Static files (serve frontend)
    FRONTEND_DIR = (Path(__file__).resolve().parents[1] / "frontend").resolve()
    if FRONTEND_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
    else:
        # Fallback: try relative to current working directory
        fallback = Path("frontend").resolve()
        if fallback.exists():
            app.mount("/static", StaticFiles(directory=str(fallback)), name="static")
        else:
            print(f"[WARN] Frontend directory not found at {FRONTEND_DIR} or {fallback}; /static disabled.")

    # Convenience redirects
    @app.get("/")
    async def root_index():
        # Redirect root to the frontend index for convenience
        return RedirectResponse(url="/static/index.html", status_code=307)

    @app.get("/favicon.ico")
    async def favicon():
        # Avoid noisy 404s if no favicon is provided
        return Response(status_code=204)

    # Routes
    @app.post("/api/case/create")
    async def create_case(case: CaseInput) -> Dict[str, str]:
        case_id = f"case-{uuid.uuid4().hex[:8]}"
        with DATA_LOCK:
            data = _load_debates()
            data["cases"][case_id] = case.model_dump()
            _save_debates(data)
        return {"case_id": case_id}

    def _placeholder_verdict() -> Dict[str, Any]:
        return {
            "emotional_score": 0,
            "logical_score": 0,
            "winner": "tie",
            "reasoning": "Pending — debate is still in progress; a detailed verdict will appear when all rounds finish.",
            "criteria_scores": {"relevance": 0, "coherence": 0, "evidence": 0, "persuasiveness": 0, "rebuttal": 0},
        }

    def _run_debate_bg(debate_id: str):
        # Execute full debate and persist results
        try:
            with DATA_LOCK:
                store = _load_debates()
                record = store["debates"].get(debate_id)
                if not record:
                    return
                case_id = record["case_id"]
                roles_dict = record.get("roles") or {}
                case_obj = store["cases"].get(case_id)
            if not case_obj:
                return
            emo_role = str(roles_dict.get("emotional_role", "prosecution")).lower()
            if emo_role not in {"prosecution", "defense"}:
                emo_role = "prosecution"
            log_role = "defense" if emo_role == "prosecution" else "prosecution"

            coordinator = DebateCoordinator(EmotionalLawyer(role=emo_role), LogicalLawyer(role=log_role), JudgeAgent())
            case = CaseInput(**case_obj)
            transcript = coordinator.run_debate(case)

            # Persist results and update stats
            with DATA_LOCK:
                store = _load_debates()
                store["debates"][debate_id] = {
                    "debate_id": transcript.debate_id,
                    "case_id": case_id,
                    "status": transcript.status,
                    "timestamp": transcript.timestamp,
                    "rounds": [
                        [arg.model_dump() for arg in rnd] for rnd in transcript.rounds
                    ],
                    "verdict": transcript.verdict.model_dump(),
                }
                _save_debates(store)

                stats = _load_stats()
                stats["total_debates"] = stats.get("total_debates", 0) + 1
                winner = transcript.verdict.winner
                if winner == "emotional":
                    stats["emotional_wins"] = stats.get("emotional_wins", 0) + 1
                elif winner == "logical":
                    stats["logical_wins"] = stats.get("logical_wins", 0) + 1
                _save_stats(stats)
        except Exception as e:
            # Mark as failed
            with DATA_LOCK:
                store = _load_debates()
                rec = store["debates"].get(debate_id)
                if rec:
                    rec["status"] = "failed"
                    rec["error"] = str(e)
                    try:
                        if isinstance(rec.get("verdict"), dict):
                            rec["verdict"]["reasoning"] = (
                                f"Debate generation failed: {str(e)}"
                            )
                        else:
                            rec["verdict"] = _placeholder_verdict()
                            rec["verdict"]["reasoning"] = (
                                f"Debate generation failed: {str(e)}"
                            )
                    except Exception:
                        # Ensure verdict exists even if above fails
                        rec["verdict"] = _placeholder_verdict()
                    _save_debates(store)

    @app.post("/api/debate/start/{case_id}")
    async def start_debate(
        case_id: str,
        background_tasks: BackgroundTasks,
        roles: DebateRoles | None = Body(None),
    ) -> Dict[str, str]:
        with DATA_LOCK:
            data = _load_debates()
            if case_id not in data["cases"]:
                raise HTTPException(status_code=404, detail="Case not found")
            role_cfg = roles or DebateRoles()
            emo_role = role_cfg.emotional_role
            log_role = "defense" if emo_role == "prosecution" else "prosecution"
            debate_id = f"deb-{uuid.uuid4().hex[:8]}"
            data["debates"][debate_id] = {
                "debate_id": debate_id,
                "case_id": case_id,
                "status": "in_progress",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                 "roles": {"emotional_role": emo_role, "logical_role": log_role},
                "rounds": [[], [], []],
                "verdict": _placeholder_verdict(),
            }
            _save_debates(data)

        background_tasks.add_task(_run_debate_bg, debate_id)
        return {"debate_id": debate_id, "status": "in_progress"}

    @app.get("/api/debate/{debate_id}")
    async def get_debate(debate_id: str) -> DebateTranscript:
        with DATA_LOCK:
            data = _load_debates()
            rec = data["debates"].get(debate_id)
            if not rec:
                raise HTTPException(status_code=404, detail="Debate not found")
            case_id = rec["case_id"]
            case_obj = data["cases"].get(case_id)
            if not case_obj:
                raise HTTPException(status_code=404, detail="Case not found for debate")

        # Build DebateTranscript (allow in_progress with empty rounds)
        rounds: List[List[Argument]] = []
        for idx, rnd in enumerate(rec.get("rounds", [[], [], []]), start=1):
            args: List[Argument] = []
            for a in rnd:
                try:
                    args.append(Argument(**a))
                except Exception:
                    # In-progress or malformed; skip
                    continue
            rounds.append(args)

        verdict_dict = rec.get("verdict", _placeholder_verdict())
        try:
            verdict = Verdict(**verdict_dict)
        except Exception:
            verdict = Verdict(**_placeholder_verdict())

        transcript = DebateTranscript(
            debate_id=rec["debate_id"],
            case=CaseInput(**case_obj),
            rounds=rounds if len(rounds) == settings.NUM_ROUNDS else [[], [], []],
            verdict=verdict,
            timestamp=rec.get("timestamp", datetime.now(timezone.utc).isoformat()),
            status=rec.get("status", "in_progress"),
        )
        return transcript

    @app.get("/api/statistics")
    async def get_statistics() -> Dict[str, int]:
        with DATA_LOCK:
            return _load_stats()

    @app.get("/api/debates")
    async def list_debates() -> List[Dict[str, Any]]:
        """Return a lightweight list of past debates for the history panel.

        Each item includes: debate_id, case_id, case_title, timestamp, status,
        winner, emotional_score, logical_score.
        """
        with DATA_LOCK:
            store = _load_debates()
        items: List[Dict[str, Any]] = []
        debates = store.get("debates", {}) or {}
        cases = store.get("cases", {}) or {}
        for key, rec in debates.items():
            if not isinstance(rec, dict):
                continue
            case_id = rec.get("case_id")
            case_obj = cases.get(case_id, {}) if isinstance(cases, dict) else {}
            verdict = rec.get("verdict") or {}
            items.append(
                {
                    "debate_id": rec.get("debate_id", key),
                    "case_id": case_id,
                    "case_title": case_obj.get("title", ""),
                    "timestamp": rec.get("timestamp"),
                    "status": rec.get("status", "in_progress"),
                    "winner": verdict.get("winner"),
                    "emotional_score": verdict.get("emotional_score"),
                    "logical_score": verdict.get("logical_score"),
                }
            )

        # Sort by timestamp descending when available
        def _sort_key(item: Dict[str, Any]):
            return item.get("timestamp") or ""

        items.sort(key=_sort_key, reverse=True)
        return items

    return app


app = create_app()
