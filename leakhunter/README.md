---
title: LeakHunter
emoji: "\U0001F4A7"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# LeakHunter

LeakHunter is an OpenEnv-compatible reinforcement-learning environment for hidden leak detection in a pressurized water distribution network.

The agent actively probes the system:
- querying pressures
- querying pipe flows
- installing persistent sensors
- closing and opening isolation valves
- committing to a final repair

That makes LeakHunter a real partially observable decision process with coupled action-observation dynamics.

## Action Space

| Action | Target | Cost | Sim time | State change |
|---|---:|---:|---:|---|
| `read_pressure Nxx` | node | 1 | 5 min | none |
| `read_flow Pxx` | pipe | 1 | 5 min | none |
| `install_sensor Nxx` | node | 3 | 20 min | persistent monitor |
| `close_valve Pxx` | valved pipe | 5 | 15 min | re-solve hydraulics |
| `open_valve Pxx` | closed valved pipe | 2 | 10 min | re-solve hydraulics |
| `repair ...` | pipe/section | 0 | 0 | terminal |

## Repair Methods

| Method | Best use | Max contribution |
|---|---|---:|
| `clamp_pipe` | exact pipe known | 0.45 |
| `replace_section` | pipe corridor within 2 hops known | 0.35 |
| `isolate_section` | correct reset-time section known | 0.25 |

## Difficulty Tiers

### Easy
- 10-node linear trunk with branches, 12 pipes, 2-3 sections, budget 20, no confounder

### Medium
- 24-node Y-branch with a loop, 30 pipes, 5-7 sections, budget 25, demand spike confounder

### Hard
- 30-node grid block with cross-connections + tank, 44 pipes, 7-10 sections, budget 28, sensor bias confounder

## Reward

Dense per-step rewards based on operational utility delta, plus terminal reward combining:
- Repair success (method-dependent)
- Localization precision
- Residual leak stopped
- Average service over episode
- Water conservation

Final score clipped to [0, 1].

## Local Development

```bash
cd leakhunter
uv sync --extra dev
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
curl http://localhost:8000/health
```

From the worktree root, the package-qualified app path also works:

```bash
python -m uvicorn leakhunter.server.app:app --host 0.0.0.0 --port 8000 --reload
```

## Docker

```bash
docker build -t leakhunter:latest -f Dockerfile .
docker run --rm -p 8000:8000 leakhunter:latest
```

## Deploy to HF Spaces

Install the optional OpenEnv tooling before pushing:

```bash
uv sync --extra openenv
```

```bash
openenv push --repo-id <username>/leakhunter
```

## Test Suite

```bash
cd leakhunter
python -m pytest tests/ -v --tb=short
python -m pytest tests/ -m performance -v --tb=short
```

From the parent of `leakhunter/`, this now works without setting `PYTHONPATH`:

```bash
python -m pytest leakhunter/tests/ -v --tb=short
```

`leakhunter/tests/conftest.py` inserts the `leakhunter/` directory onto `sys.path` during test collection, so the existing bare imports like `from models import ...` and `from server.environment import ...` resolve consistently. You do not need `PYTHONPATH=leakhunter` for pytest anymore.

## Baseline Scores

Current scripted baselines with Qwen 3.5 35B:

| Difficulty | Score |
|---|---:|
| Easy | 0.652 |
| Medium | 0.340 |
| Hard | 0.290 |
