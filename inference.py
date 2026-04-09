from __future__ import annotations

import argparse
import os
import re
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter, sleep

import requests as _requests
from openai import OpenAI

# ── Self-contained fallbacks ────────────────────────────────────────────────
# The hackathon validator runs inference.py in isolation (/tmp/workspace/)
# without client.py, models.py, or server/. These fallbacks let inference.py
# work standalone with just openai + requests.

try:
    from client import LeakHunterEnv
    from models import LeakHunterAction
except ImportError:

    @dataclass
    class LeakHunterAction:  # type: ignore[no-redef]
        action_type: str
        target_id: str
        method: str | None = None

        def model_dump(self):
            return {"action_type": self.action_type, "target_id": self.target_id, "method": self.method}

        def as_command(self):
            if self.action_type == "repair":
                return f"repair {self.target_id} {self.method}"
            return f"{self.action_type} {self.target_id}"

    class _Obs:
        def __init__(self, data: dict):
            self.text = data.get("text", "")
            self.done = data.get("done", False)
            self.summary_text = data.get("summary_text")
            self.sensor_readings = data.get("sensor_readings", [])
            self.alerts = data.get("alerts", [])
            self.reward_breakdown = data.get("reward_breakdown")

    class _StepResult:
        def __init__(self, observation, reward, done):
            self.observation = observation
            self.reward = reward
            self.done = done

    class _State:
        def __init__(self, data: dict):
            self.final_score = data.get("final_score")
            self.step_count = data.get("step_count", 0)
            self.episode_id = data.get("episode_id")
            self.last_message = data.get("last_message")
            self.forced_repair = data.get("forced_repair", False)

    class LeakHunterEnv:  # type: ignore[no-redef]
        def __init__(self, base_url="http://localhost:8000", timeout=30.0):
            self.base_url = base_url.rstrip("/")
            self.timeout = timeout
            self._session = _requests.Session()

        def reset(self, difficulty="easy", seed=None):
            payload: dict = {"difficulty": difficulty}
            if seed is not None:
                payload["seed"] = seed
            resp = self._session.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return _Obs(resp.json()["observation"])

        def step(self, action):
            resp = self._session.post(
                f"{self.base_url}/step", json={"action": action.model_dump()}, timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            return _StepResult(
                observation=_Obs(data["observation"]),
                reward=data.get("reward"),
                done=bool(data.get("done", False)),
            )

        def state(self):
            resp = self._session.get(f"{self.base_url}/state", timeout=self.timeout)
            resp.raise_for_status()
            return _State(resp.json())

        def close(self):
            self._session.close()

# Trace-related imports — only needed when --trace is used inside Docker.
try:
    import orjson
    from models import LeakHunterObservation, LeakHunterState, ResetObservation, RewardBreakdown
    from server.trace_models import AgentFrame, EpisodeTrace, TraceFrame, TraceMeta, TraceTerminal, TraceTopology
    _TRACE_AVAILABLE = True
except ImportError:
    _TRACE_AVAILABLE = False

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
BENCHMARK = "leakhunter"


# ── Mandatory stdout logging ([START], [STEP], [END]) ──────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

SYSTEM_PROMPT = """You are LeakHunter, a hydraulic field investigator.

Goal:
Find and fix a hidden pipe leak in a pressurized water distribution network under a strict action budget.

Action syntax:
- read_pressure Nxx
- read_flow Pxx
- install_sensor Nxx
- close_valve Pxx
- open_valve Pxx
- repair Pxx clamp_pipe
- repair Pxx replace_section
- repair Sx isolate_section

Reasoning rules:
1. Pressure drops near a leak are usually local, but medium difficulty may include a demand spike in a different zone and hard difficulty may include one node with a positive pressure sensor bias.
2. Baseline flow directions shown at reset are healthy-reference directions only. Current flow requires read_flow.
3. Installed sensors matter most before valve operations because they auto-report after topology changes.
4. Valve actions change the hydraulic state. Reads do not.
5. Use clamp_pipe only if you are highly confident of the exact pipe.
6. Use replace_section when you have narrowed the leak to a pipe corridor but still have one-pipe ambiguity.
7. Use isolate_section when you know the correct section but not the exact pipe.
8. Keep actions concise. Return exactly one action command and nothing else."""

READ_PRESSURE_RE = re.compile(
    r"\bread_pressure\s+([A-Z][A-Za-z0-9]*\d+)\b", re.I
)
READ_FLOW_RE = re.compile(r"\bread_flow\s+(P\d+)\b", re.I)
INSTALL_SENSOR_RE = re.compile(
    r"\binstall_sensor\s+([A-Z][A-Za-z0-9]*\d+)\b", re.I
)
CLOSE_RE = re.compile(r"\bclose_valve\s+(P\d+)\b", re.I)
OPEN_RE = re.compile(r"\bopen_valve\s+(P\d+)\b", re.I)
REPAIR_RE = re.compile(
    r"\brepair\s+((?:P\d+)|(?:S\d+))\s+(clamp_pipe|replace_section|isolate_section)\b",
    re.I,
)


@dataclass
class RunResult:
    difficulty: str
    seed: int
    final_score: float
    steps: int
    final_message: str


if _TRACE_AVAILABLE:
    class ClientTraceRecorder:
        def __init__(
            self,
            trace_dir: str = "traces",
            agent_model: str | None = None,
            agent_name: str | None = "llm",
        ) -> None:
            self._trace_dir = Path(trace_dir)
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            self._agent_model = agent_model
            self._agent_name = agent_name
            self._trace: EpisodeTrace | None = None
            self._trace_path: Path | None = None
            self._pending_agent_frame: AgentFrame | None = None
            self._finalized = False

        def record_reset(
            self, reset_obs: ResetObservation, state: LeakHunterState, difficulty: str, seed: int
        ) -> None:
            episode_id = state.episode_id or f"{difficulty}_{seed}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
            self._trace = EpisodeTrace(
                meta=TraceMeta(
                    episode_id=episode_id,
                    difficulty=difficulty,
                    seed=seed,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    agent_model=self._agent_model,
                    agent_name=self._agent_name,
                ),
                topology=TraceTopology(
                    nodes=list(reset_obs.nodes),
                    pipes=list(reset_obs.pipes),
                    sections=list(reset_obs.sections),
                    zones=list(reset_obs.zones),
                ),
                frames=[
                    TraceFrame(
                        step_index=0,
                        kind="reset",
                        action=None,
                        observation=reset_obs,
                        state=state,
                        public_readings=list(reset_obs.sensor_readings),
                        alerts=list(reset_obs.alerts),
                        hidden=None,
                        agent=None,
                    )
                ],
            )
            self._trace_path = self._trace_dir / f"{episode_id}.client.trace.json"
            self._finalized = False
            self._pending_agent_frame = None

        def set_agent_frame(self, agent_frame: AgentFrame) -> None:
            self._pending_agent_frame = agent_frame

        def record_step(
            self, action: LeakHunterAction, obs: LeakHunterObservation, state: LeakHunterState
        ) -> None:
            if self._trace is None:
                raise RuntimeError("ClientTraceRecorder.record_reset() must be called first")

            kind = "terminal" if obs.done else "step"
            self._trace.frames.append(
                TraceFrame(
                    step_index=state.step_count,
                    kind=kind,
                    action=action,
                    observation=obs,
                    state=state,
                    public_readings=list(obs.sensor_readings),
                    alerts=list(obs.alerts),
                    hidden=None,
                    agent=self._consume_agent_frame(),
                )
            )
            if obs.done:
                repair_target, repair_method = self._extract_repair_details(
                    state.last_message or ""
                )
                breakdown = obs.reward_breakdown or RewardBreakdown(
                    final_score=float(state.final_score or 0.0)
                )
                self._trace.terminal = TraceTerminal(
                    final_score=float(state.final_score or breakdown.final_score or 0.0),
                    forced_repair=state.forced_repair,
                    reward_breakdown=breakdown,
                    true_leak_pipe_id="unknown",
                    true_leak_section_id="unknown",
                    repair_target=repair_target,
                    repair_method=repair_method,
                )
                self.finalize()

        def finalize(self) -> None:
            if self._finalized or self._trace is None or self._trace_path is None:
                return
            with self._trace_path.open("wb") as f:
                f.write(orjson.dumps(self._trace.model_dump(mode="json"), option=orjson.OPT_INDENT_2))
                f.write(b"\n")
            self._finalized = True

        def _consume_agent_frame(self) -> AgentFrame | None:
            agent_frame = self._pending_agent_frame
            self._pending_agent_frame = None
            return agent_frame

        def _extract_repair_details(self, message: str) -> tuple[str | None, str | None]:
            match = re.search(
                r"\brepair=([A-Z][A-Za-z0-9]*)\s+(clamp_pipe|replace_section|isolate_section)\b",
                message,
            )
            if match is None:
                return (None, None)
            return (match.group(1), match.group(2))
else:
    ClientTraceRecorder = None  # type: ignore[assignment,misc]


def parse_action(text: str, default_target: str = "N01") -> LeakHunterAction:
    if m := REPAIR_RE.search(text):
        return LeakHunterAction(
            action_type="repair",
            target_id=m.group(1).upper(),
            method=m.group(2).lower(),
        )
    if m := READ_PRESSURE_RE.search(text):
        return LeakHunterAction(
            action_type="read_pressure", target_id=m.group(1).upper()
        )
    if m := READ_FLOW_RE.search(text):
        return LeakHunterAction(
            action_type="read_flow", target_id=m.group(1).upper()
        )
    if m := INSTALL_SENSOR_RE.search(text):
        return LeakHunterAction(
            action_type="install_sensor", target_id=m.group(1).upper()
        )
    if m := CLOSE_RE.search(text):
        return LeakHunterAction(
            action_type="close_valve", target_id=m.group(1).upper()
        )
    if m := OPEN_RE.search(text):
        return LeakHunterAction(
            action_type="open_valve", target_id=m.group(1).upper()
        )
    # keyword fallback
    low = text.lower()
    ids = re.findall(r"\b((?!Z\d)[A-Z][A-Za-z0-9]*\d+)\b", text.upper())
    chosen = ids[0] if ids else default_target
    is_pipe = chosen.startswith("P")
    is_section = chosen.startswith("S")
    is_node = not is_pipe and not is_section
    if "repair" in low or "isolate" in low or "clamp" in low or "replace" in low:
        if "clamp" in low:
            method = "clamp_pipe"
        elif "isolate" in low:
            method = "isolate_section"
            if is_pipe:
                # LLM said isolate but gave a pipe — use replace_section instead
                method = "replace_section"
        else:
            method = "isolate_section" if is_section else "replace_section"
        return LeakHunterAction(
            action_type="repair", target_id=chosen, method=method
        )
    if "sensor" in low and is_node:
        return LeakHunterAction(action_type="install_sensor", target_id=chosen)
    if "flow" in low and is_pipe:
        return LeakHunterAction(action_type="read_flow", target_id=chosen)
    if "close" in low and is_pipe:
        return LeakHunterAction(action_type="close_valve", target_id=chosen)
    if "open" in low and is_pipe:
        return LeakHunterAction(action_type="open_valve", target_id=chosen)
    # Default: if it's a pipe, read flow; if it's a node, read pressure
    if is_pipe:
        return LeakHunterAction(action_type="read_flow", target_id=chosen)
    try:
        return LeakHunterAction(action_type="read_pressure", target_id=chosen)
    except Exception:
        return LeakHunterAction(
            action_type="read_pressure", target_id=default_target
        )


def choose_default_target(reset_text: str) -> str:
    in_nodes = False
    for line in reset_text.splitlines():
        stripped = line.strip()
        if stripped == "Nodes:":
            in_nodes = True
            continue
        if not in_nodes:
            continue
        if stripped == "Pipes:":
            break
        m = re.match(r"^([A-Z][A-Za-z0-9]*\d+)\s+junction\b", stripped)
        if m:
            return m.group(1)
    return "N01"


def run_episode(
    client: LeakHunterEnv,
    llm: OpenAI,
    difficulty: str,
    seed: int,
    max_steps: int = 32,
    trace_recorder: object | None = None,
    model_name: str | None = None,
) -> RunResult:
    model_name = model_name or MODEL_NAME
    obs = client.reset(difficulty=difficulty, seed=seed)
    state = client.state()
    if trace_recorder is not None:
        trace_recorder.record_reset(obs, state, difficulty, seed)
    default_target = choose_default_target(obs.text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs.text},
    ]

    task_name = f"{difficulty}-seed{seed}"
    log_start(task=task_name, env=BENCHMARK, model=model_name)

    step_rewards: list[float] = []
    step_count = 0
    final_obs = None
    final_score = 0.0
    try:
        for step_num in range(1, max_steps + 1):
            started_at = perf_counter()
            for _retry in range(3):
                try:
                    response = llm.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=4096,
                    )
                    break
                except Exception as llm_err:
                    if _retry == 2:
                        raise
                    print(f"[WARN] LLM call failed (attempt {_retry+1}/3): {llm_err}", flush=True)
                    sleep(2 ** _retry)
            latency_ms = (perf_counter() - started_at) * 1000.0
            msg = response.choices[0].message
            assistant_text = msg.content or ""
            # Capture thinking/reasoning tokens from thinking models (e.g. Qwen 3.5)
            reasoning_text = getattr(msg, "reasoning", None) or ""
            if not reasoning_text and hasattr(msg, "model_extra") and msg.model_extra:
                reasoning_text = msg.model_extra.get("reasoning", "") or ""
            action = parse_action(assistant_text, default_target=default_target)
            if trace_recorder is not None:
                trace_recorder.set_agent_frame(
                    AgentFrame(
                        raw_output=assistant_text,
                        reasoning=reasoning_text or None,
                        parsed_action=action.as_command(),
                        model_name=model_name,
                        latency_ms=latency_ms,
                    )
                )
            step = client.step(action)
            final_obs = step.observation
            state = client.state()
            step_count = step_num
            reward = float(step.reward or 0.0)
            step_rewards.append(reward)

            log_step(
                step=step_num,
                action=action.as_command(),
                reward=reward,
                done=step.done,
                error=None,
            )

            if trace_recorder is not None:
                trace_recorder.record_step(action, final_obs, state)
            messages.append({"role": "assistant", "content": assistant_text})
            messages.append({"role": "user", "content": final_obs.text})
            # summary injection every time environment provides one
            if final_obs.summary_text:
                messages = [
                    messages[0],
                    {
                        "role": "system",
                        "content": f"Running factual summary:\n{final_obs.summary_text}",
                    },
                    *messages[-6:],
                ]
            if step.done:
                break

        final_score = min(max(float(state.final_score or 0.0), 0.0), 1.0)
    finally:
        # [END] MUST always be emitted after [START], even on exception
        success = final_score >= 0.35
        log_end(
            success=success,
            steps=step_count,
            score=final_score,
            rewards=step_rewards,
        )
        if trace_recorder is not None:
            trace_recorder.finalize()

    return RunResult(
        difficulty=difficulty,
        seed=seed,
        final_score=final_score,
        steps=state.step_count,
        final_message=state.last_message or "",
    )


def _wait_for_server(base_url: str, timeout: float = 120.0, interval: float = 2.0) -> None:
    """Poll the env server health endpoint until it responds or timeout is reached."""
    import urllib.request
    import urllib.error

    health_url = f"{base_url.rstrip('/')}/health"
    deadline = perf_counter() + timeout
    attempt = 0
    while perf_counter() < deadline:
        attempt += 1
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    print(f"[INFO] Server ready after {attempt} attempt(s)", flush=True)
                    return
        except Exception:
            pass
        sleep(interval)
    raise RuntimeError(f"Server at {base_url} did not become healthy within {timeout}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("LEAKHUNTER_URL", "http://localhost:8000"))
    parser.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33])
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--trace-dir", default="traces")
    args = parser.parse_args()

    # Log configuration for debugging
    print(f"[INFO] base_url={args.base_url} api_base={API_BASE_URL} model={MODEL_NAME}", flush=True)

    # Wait for the environment server to be ready
    _wait_for_server(args.base_url)

    api_key = API_KEY or "ollama"
    llm = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    env = LeakHunterEnv(base_url=args.base_url)

    rows: list[RunResult] = []
    for difficulty in ("easy", "medium", "hard"):
        for seed in args.seeds:
            trace_recorder = None
            if args.trace:
                if not _TRACE_AVAILABLE:
                    print("Warning: --trace requires the full leakhunter package; skipping.", flush=True)
                else:
                    trace_recorder = ClientTraceRecorder(
                        trace_dir=args.trace_dir,
                        agent_model=MODEL_NAME,
                        agent_name="llm",
                    )
            try:
                rows.append(
                    run_episode(
                        env,
                        llm,
                        difficulty,
                        seed,
                        trace_recorder=trace_recorder,
                    )
                )
            except Exception as exc:
                print(f"[ERROR] Episode {difficulty}/seed={seed} failed: {exc}", flush=True)
                traceback.print_exc()
                rows.append(RunResult(
                    difficulty=difficulty,
                    seed=seed,
                    final_score=0.0,
                    steps=0,
                    final_message=f"ERROR: {exc}",
                ))

    print("\nLeakHunter baseline results")
    print("=" * 60)
    for row in rows:
        print(
            f"{row.difficulty:>6} seed={row.seed:<3d} "
            f"score={row.final_score:.3f} steps={row.steps:<2d} "
            f"{row.final_message}"
        )
    avg = sum(r.final_score for r in rows) / max(len(rows), 1)
    print("=" * 60)
    print(f"Average score: {avg:.3f}")
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[FATAL] Unhandled exception in main: {exc}", flush=True)
        traceback.print_exc()
        raise SystemExit(1)
