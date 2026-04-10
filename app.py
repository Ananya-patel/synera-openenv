"""
Synera Triage OpenEnv — FastAPI application.

Endpoints (OpenEnv spec):
  POST /reset  → Observation
  POST /step   → {observation, reward, done, info}
  GET  /state  → dict
  GET  /health → {"status": "ok"}
  GET  /tasks  → task descriptions

Environment variables:
  API_BASE_URL   LLM endpoint for inference.py
  MODEL_NAME     LLM model identifier for inference.py
  HF_TOKEN       HF / API authentication token for inference.py
  TASK_ID        Default task (task1 | task2 | task3)  default: task1
  SEED           Random seed for reproducibility       default: 42
  PORT           HTTP port                             default: 7860
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from env.environment import SyneraTriageEnv, VALID_TASKS
from models.schemas import Action, Observation, Reward

app = FastAPI(
    title="Synera Clinical Triage Environment",
    description=(
        "OpenEnv-compliant RL environment. An AI agent observes wearable vital streams "
        "from hospital ward patients and must triage correctly across 3 tasks: "
        "artifact rejection (easy), exertion classification (medium), "
        "trajectory acceleration triage (hard)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_env: Optional[SyneraTriageEnv] = None


def _get_env() -> SyneraTriageEnv:
    global _env
    if _env is None:
        task = os.getenv("TASK_ID", "task1")
        seed = int(os.getenv("SEED", "42"))
        _env = SyneraTriageEnv(task_id=task, seed=seed)
    return _env


# ── Request/response models ────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    seed: Optional[int] = None


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    score: float          # ← add this
    done: bool
    info: dict

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "environment": "synera-triage-v1", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return SyneraTriageEnv.task_descriptions()


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest = ResetRequest()):
    global _env
    task_id = req.task_id or os.getenv("TASK_ID", "task1")
    seed = req.seed if req.seed is not None else int(os.getenv("SEED", "42"))
    if task_id not in VALID_TASKS:
        raise HTTPException(400, f"task_id must be one of {VALID_TASKS}")
    _env = SyneraTriageEnv(task_id=task_id, seed=seed)
    return _env.reset()


@app.post("/step", response_model=StepResponse)
def step(action: Action):
    env = _get_env()
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    return StepResponse(
        observation=obs,
        reward=reward,
        score=reward.score,   # ← add this
        done=done,
        info=info,
    )


@app.get("/state")
def state():
    return _get_env().state()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)