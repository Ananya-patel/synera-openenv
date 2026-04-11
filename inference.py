"""
inference.py — Synera Triage OpenEnv chain-of-thought clinical reasoning agent.

REQUIRED by hackathon spec:
- Uses OpenAI API client (not Ollama/LangChain)
- Reads API_BASE_URL, MODEL_NAME, HF_TOKEN from environment variables
- Emits structured [START], [STEP], [END] stdout logs as JSON
- File must be named inference.py and live in project root
- Must complete in < 20 minutes on vcpu=2, memory=8gb

Usage:
  export API_BASE_URL=https://api.openai.com/v1
  export MODEL_NAME=gpt-4o-mini
  export HF_TOKEN=your_token_here
  export ENV_BASE_URL=http://localhost:7860   # or your HF Space URL
  python inference.py
"""

import json
import os
import re
import sys
import time

import httpx
from openai import OpenAI

# ── Config ─────────────────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

if not HF_TOKEN:
    print(json.dumps({"event": "ERROR", "error": "HF_TOKEN (or OPENAI_API_KEY) not set"}), flush=True)
    # Still print END events so the validator can read scores
    for _tid in ["task1", "task2", "task3"]:
        print(json.dumps({"event": "END", "task_id": _tid, "score": 0.1, "task_score": 0.1,
                          "error": "HF_TOKEN not set"}), flush=True)
    sys.exit(1)

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
http   = httpx.Client(base_url=ENV_BASE_URL, timeout=60)

TASKS = ["task1", "task2", "task3"]

# ── System prompt (chain-of-thought clinical reasoner) ─────────────────────────
#
# Design principle: do NOT hand the model a rule lookup table.
# Instead, give it clinical context and ask it to reason like a clinician.
# The model must infer artifact/exertion/trajectory patterns from the data
# itself — this is what Phase 2 agentic evaluation (Nemotron 3 Super) tests.

SYSTEM_PROMPT = """You are Synera, an autonomous AI clinician monitoring hospital patients
via continuous wearable vital sensors. Your role is real-time triage: determine which patients
need immediate attention and which are physiologically stable.

VITAL SIGN CONTEXT
- heart_rate (BPM): normal resting adult 60–100. Changes >40 BPM between consecutive readings
  are physiologically impossible in healthy tissue and indicate sensor malfunction.
- spo2 (%): blood oxygen saturation. Normal ≥95%. Downward trends are serious.
- temperature (°C): normal 36.5–37.5°C. Fever >38°C with rising HR suggests systemic response.
- motion_score (0–10): inertial measurement unit. 0=at rest, 10=vigorous exercise.
  Physical exertion NORMALLY elevates heart rate; this is benign and must not trigger alerts.
- baseline_hr_mean / baseline_hr_std: this patient's personal calibrated resting HR baseline,
  NOT a population average. Deviation must be measured against their own baseline.

CLINICAL REASONING STEPS (apply to each patient)
Step 1 — Sensor integrity: Is the latest heart_rate reading physically plausible given the
  previous reading in the window? An abrupt jump that no human body could produce in one
  5-second interval is a sensor artifact — discard it clinically.

Step 2 — Exertion context: If the patient is moving (high motion_score) and their HR is
  elevated proportionally, this is exercise physiology, not pathology. Do not alert.

Step 3 — Trajectory analysis (the hard part): With a clean, resting reading, look at the
  SHAPE of the HR trend across the window — not just the current value.
  Ask: is the rate of HR increase itself increasing? (i.e., acceleration, not just elevation)
  A patient whose HR is climbing faster and faster, while deviating significantly from their
  personal baseline and at rest, is showing early signs of physiological decompensation.
  This warrants a critical alert.

Step 4 — Multi-patient prioritisation: Rank ALL patients by urgency. Actively deteriorating
  patients outrank exertion patients, who outrank stable patients. Identical urgency patients
  can share adjacent ranks — use your best clinical judgement on tie-breaking.

OUTPUT INSTRUCTIONS
First, reason through each patient inside <reasoning>...</reasoning> tags.
Then output ONLY a valid JSON object with no markdown, no extra text:

{"decisions": [
  {
    "patient_id": "<copy exactly>",
    "is_artifact": <true|false>,
    "is_exertion": <true|false>,
    "alert_tier": <0|1|2|3>,
    "priority_rank": <1..N, 1=most urgent>
  }
]}

alert_tier meanings: 0=stable, 1=watch (mild deviation), 2=advisory, 3=critical/SYNERA_STATE"""


# ── Observation formatter ───────────────────────────────────────────────────────
# Present data as a clinical briefing rather than raw JSON — this activates the
# model's medical knowledge and produces richer chain-of-thought.

def _format_obs(obs: dict) -> str:
    """Render observation as a human-readable clinical briefing."""
    task_id = obs.get("task_id", "unknown")
    step    = obs.get("episode_step", 0)
    lines   = [f"=== SYNERA TRIAGE BRIEFING | {task_id.upper()} | Step {step} ===\n"]

    for p in obs.get("patients", []):
        pid      = p["patient_id"]
        window   = p.get("window", [])
        b_mean   = p.get("baseline_hr_mean", 70)
        b_std    = p.get("baseline_hr_std", 5)
        calib    = p.get("calibration_complete", False)
        step_num = p.get("step_number", 0)

        lines.append(f"Patient {pid}  (baseline HR: {b_mean:.1f} ± {b_std:.1f} BPM,"
                     f" calibrated: {calib}, step: {step_num})")

        if not window:
            lines.append("  No readings available yet.\n")
            continue

        lines.append(f"  Recent vital window ({len(window)} readings, oldest → newest):")
        for i, r in enumerate(window):
            marker = " ←latest" if i == len(window) - 1 else ""
            dev    = ""
            if calib and b_std > 0:
                sigma = (r["heart_rate"] - b_mean) / b_std
                dev   = f"  [{sigma:+.1f}σ from baseline]"
            lines.append(
                f"    [{i+1:2d}] HR={r['heart_rate']:6.1f} BPM  "
                f"SpO2={r['spo2']:5.1f}%  Temp={r['temperature']:.1f}°C  "
                f"Motion={r['motion_score']}{dev}{marker}"
            )

        # Summarise HR trend for the model
        hrs = [r["heart_rate"] for r in window]
        if len(hrs) >= 3:
            d1 = [hrs[i+1] - hrs[i] for i in range(len(hrs)-1)]
            trend = "accelerating" if sum(d for d in d1[-3:]) > 0 and d1[-1] > d1[0] else \
                    "rising steadily" if sum(d for d in d1) > 0 else \
                    "falling" if sum(d for d in d1) < 0 else "flat/stable"
            lines.append(f"  HR trend: {trend}  (recent deltas: {[round(d,1) for d in d1[-4:]]})")

        lines.append("")

    lines.append("Provide your <reasoning> then the JSON decision for ALL patients above.")
    return "\n".join(lines)


# ── Response parsing ────────────────────────────────────────────────────────────

def _safe_parse(raw: str, patients: list) -> tuple[dict, str]:
    """
    Parse LLM response containing optional <reasoning> block then JSON.
    Returns (action_dict, reasoning_text).
    Falls back to safe no-op on parse failure.
    """
    reasoning = ""
    text = raw.strip()

    # Extract reasoning block if present
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
        text = text[reasoning_match.end():].strip()

    # Strip markdown code fences
    if text.startswith("```"):
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence_match:
            text = fence_match.group(1).strip()
        else:
            text = "\n".join(text.split("\n")[1:]).strip()

    # Find the JSON object (scan forward to first '{')
    brace_idx = text.find("{")
    if brace_idx > 0:
        text = text[brace_idx:]

    try:
        action = json.loads(text)
        # Validate required fields exist
        for d in action.get("decisions", []):
            d.setdefault("is_artifact", False)
            d.setdefault("is_exertion", False)
            d.setdefault("alert_tier", 0)
            d.setdefault("priority_rank", 1)
        return action, reasoning
    except Exception:
        fallback = {
            "decisions": [
                {"patient_id": p["patient_id"], "is_artifact": False,
                 "is_exertion": False, "alert_tier": 0, "priority_rank": i + 1}
                for i, p in enumerate(patients)
            ]
        }
        return fallback, reasoning


def ask_agent(obs: dict) -> tuple[dict, str]:
    """
    Send observation to LLM as a clinical briefing.
    Returns (action_dict, reasoning_text).
    """
    briefing = _format_obs(obs)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": briefing},
        ],
        temperature=0.0,
        max_tokens=1200,
    )
    return _safe_parse(resp.choices[0].message.content, obs.get("patients", []))


def _clamp_score(value: float) -> float:
    """Clamp a score to be strictly in (0, 1) — never 0.0 or 1.0."""
    return round(max(0.05, min(0.95, float(value))), 4)


def run_task(task_id: str) -> dict:
    """Run one full episode and return summary dict. Always prints [END] event."""
    # Safe defaults — used if an exception fires before/during the episode
    task_score     = 0.1
    step_count     = 0
    total_reward   = 0.0
    episode_rewards: list[float] = []
    reasoning_steps = 0
    error_msg: str | None = None

    try:
        reset_resp = http.post("/reset", json={"task_id": task_id, "seed": 42})
        reset_resp.raise_for_status()
        obs = reset_resp.json()

        done = False

        # ── [START] log ────────────────────────────────────────────────────────
        print(json.dumps({
            "event": "START",
            "task_id": task_id,
            "n_patients": len(obs.get("patients", [])),
            "seed": 42,
            "model": MODEL_NAME,
        }), flush=True)

        while not done:
            t0 = time.time()
            action, reasoning = ask_agent(obs)

            step_resp = http.post("/step", json=action)
            step_resp.raise_for_status()
            data = step_resp.json()

            obs    = data["observation"]
            reward = data["reward"]
            done   = data["done"]
            info   = data.get("info", {})

            step_reward = float(reward.get("total", reward.get("score", 0.5)))
            total_reward += step_reward
            episode_rewards.append(step_reward)
            step_count += 1
            if reasoning:
                reasoning_steps += 1

            step_score = _clamp_score(reward.get("score", step_reward))

            # ── [STEP] log — JSON format ────────────────────────────────────
            step_log: dict = {
                "event": "STEP",
                "task_id": task_id,
                "step": step_count,
                "score": step_score,
                "reward": round(step_reward, 4),
                "done": done,
                "latency_ms": round((time.time() - t0) * 1000),
            }
            if reasoning:
                step_log["reasoning_snippet"] = (
                    reasoning[:300].replace("\n", " ")
                    + ("..." if len(reasoning) > 300 else "")
                )
            print(json.dumps(step_log), flush=True)

        avg_reward = total_reward / max(1, step_count)
        task_score = _clamp_score(avg_reward)

    except Exception as exc:
        error_msg = str(exc)
        print(json.dumps({"event": "ERROR", "task_id": task_id, "error": error_msg}),
              flush=True)
        # Compute partial score from whatever steps completed
        if step_count > 0 and episode_rewards:
            avg = total_reward / step_count
            task_score = _clamp_score(avg)
        else:
            task_score = 0.1   # safe default strictly in (0, 1)

    avg_reward = total_reward / max(1, step_count)

    summary: dict = {
        "event":          "END",
        "task_id":        task_id,
        "score":          task_score,       # key the validator reads
        "task_score":     task_score,       # alias kept for backward compat
        "steps":          step_count,
        "total_reward":   round(total_reward, 4),
        "avg_reward":     round(avg_reward, 4),
        "reasoning_steps": reasoning_steps,
        "reasoning_rate": round(reasoning_steps / max(1, step_count), 3),
    }
    if episode_rewards:
        summary["min_reward"] = round(min(episode_rewards), 4)
        summary["max_reward"] = round(max(episode_rewards), 4)
    if error_msg:
        summary["error"] = error_msg

    # ── [END] log — ALWAYS printed, even on failure ─────────────────────────
    print(json.dumps(summary), flush=True)
    return summary


def main():
    all_results: dict[str, dict] = {}

    for task_id in TASKS:
        print(json.dumps({"event": "TASK_BEGIN", "task_id": task_id}), flush=True)
        try:
            summary = run_task(task_id)
            all_results[task_id] = summary
        except Exception as exc:
            # run_task handles its own exceptions; this is a last-resort catch
            task_score = 0.1
            summary = {
                "event":      "END",
                "task_id":    task_id,
                "score":      task_score,
                "task_score": task_score,
                "error":      str(exc),
            }
            print(json.dumps(summary), flush=True)
            all_results[task_id] = summary

    # ── Final score summary ─────────────────────────────────────────────────
    scores = {tid: all_results.get(tid, {}).get("task_score", 0.1) for tid in TASKS}
    overall = round(sum(scores.values()) / len(TASKS), 4)

    print(json.dumps({"event": "FINAL", "scores": scores, "overall": overall}), flush=True)

    # Human-readable summary (for convenience)
    print(f"\n{'=' * 60}", flush=True)
    print("BASELINE SCORES", flush=True)
    print("=" * 60, flush=True)
    for tid in TASKS:
        print(f"  {tid}: {scores[tid]:.4f}", flush=True)
    print(f"  overall: {overall:.4f}", flush=True)


if __name__ == "__main__":
    main()
