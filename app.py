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
from fastapi import FastAPI, HTTPException
import uvicorn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.schemas import Action, Observation, Reward
from env.environment import SyneraTriageEnv

app = FastAPI()

# One env instance per task — validator will hit each separately
_envs: dict[str, SyneraTriageEnv] = {}

def _get_env(task_id: str) -> SyneraTriageEnv:
    if task_id not in _envs:
        _envs[task_id] = SyneraTriageEnv(task_id=task_id, seed=42)
    return _envs[task_id]


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(task_id: str = "task1"):
    env = _get_env(task_id)
    obs = env.reset()
    return {"observation": obs.model_dump(), "task_id": task_id}


@app.post("/step")
def step(action: Action, task_id: str = "task1"):
    env = _get_env(task_id)
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),   # includes score field
        "score": reward.score,           # also hoisted to top level
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(task_id: str = "task1"):
    env = _get_env(task_id)
    return env.state()


@app.get("/tasks")
def tasks():
    return SyneraTriageEnv.task_descriptions()


def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()