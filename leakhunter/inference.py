from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass

from openai import OpenAI

from client import LeakHunterEnv
from models import LeakHunterAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4.1-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""

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
) -> RunResult:
    obs = client.reset(difficulty=difficulty, seed=seed)
    default_target = choose_default_target(obs.text)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs.text},
    ]
    final_obs = None
    for _ in range(max_steps):
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=512,
        )
        assistant_text = response.choices[0].message.content or ""
        action = parse_action(assistant_text, default_target=default_target)
        step = client.step(action)
        final_obs = step.observation
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
    state = client.state()
    return RunResult(
        difficulty=difficulty,
        seed=seed,
        final_score=float(state.final_score or 0.0),
        steps=state.step_count,
        final_message=state.last_message or "",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--seeds", type=int, nargs="*", default=[11, 22, 33])
    args = parser.parse_args()

    api_key = API_KEY or "ollama"
    llm = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    env = LeakHunterEnv(base_url=args.base_url)

    rows: list[RunResult] = []
    for difficulty in ("easy", "medium", "hard"):
        for seed in args.seeds:
            rows.append(run_episode(env, llm, difficulty, seed))

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
    main()
