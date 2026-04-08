from __future__ import annotations

import re

import pytest

from models import LeakHunterAction
from server.environment import LeakHunterEnvironment


def test_invalid_action_costs_one_budget():
    env = LeakHunterEnvironment()
    env.reset(difficulty="easy", seed=11)
    before = env.state.budget_remaining
    obs = env.step(
        LeakHunterAction(action_type="read_pressure", target_id="N99")
    )
    after = env.state.budget_remaining
    assert after == before - 1
    assert "ERROR" in obs.text


def test_budget_exhaustion_forces_repair():
    env = LeakHunterEnvironment()
    env.reset(difficulty="easy", seed=11)
    actions = [
        LeakHunterAction(action_type="close_valve", target_id="P03"),
        LeakHunterAction(action_type="open_valve", target_id="P03"),
        LeakHunterAction(action_type="close_valve", target_id="P05"),
        LeakHunterAction(action_type="open_valve", target_id="P05"),
        LeakHunterAction(action_type="install_sensor", target_id="N09"),
        LeakHunterAction(action_type="read_pressure", target_id="N08"),
        LeakHunterAction(action_type="read_flow", target_id="P09"),
        LeakHunterAction(action_type="read_pressure", target_id="N07"),
    ]
    obs = None
    for action in actions:
        obs = env.step(action)

    assert obs is not None
    assert obs.done
    assert env.state.forced_repair
    assert env.state.budget_remaining == 0


def test_double_close_does_not_crash():
    env = LeakHunterEnvironment()
    env.reset(difficulty="easy", seed=11)
    env.step(LeakHunterAction(action_type="close_valve", target_id="P03"))
    obs = env.step(
        LeakHunterAction(action_type="close_valve", target_id="P03")
    )
    assert "already closed" in obs.text


def test_all_valves_closed_no_crash():
    env = LeakHunterEnvironment()
    env.reset(difficulty="hard", seed=31)
    for pid, is_open in list(env.state.valve_states.items()):
        if is_open:
            obs = env.step(
                LeakHunterAction(action_type="close_valve", target_id=pid)
            )
            if obs.done:
                break
    assert env.state.done or env._ctx.current_result is not None


def test_solver_failure_recovery(monkeypatch):
    env = LeakHunterEnvironment()
    env.reset(difficulty="easy", seed=11)

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic solver crash")

    monkeypatch.setattr(env._ctx.simulator, "solve", boom)
    obs = env.step(
        LeakHunterAction(action_type="close_valve", target_id="P03")
    )
    assert obs is not None
    assert env._ctx is not None
    assert env._ctx.current_result.solver_ok is False
    assert env._ctx.current_result.error == "synthetic solver crash"


def test_summary_does_not_leak_post_revision_pressures():
    env = LeakHunterEnvironment()
    env.reset(difficulty="easy", seed=11)

    obs1 = env.step(
        LeakHunterAction(action_type="read_pressure", target_id="N08")
    )
    pre_valve_reading = obs1.sensor_readings[0]
    expected_delta = round(
        100.0
        * (pre_valve_reading.value - pre_valve_reading.baseline_value)
        / pre_valve_reading.baseline_value,
        1,
    )

    env.step(LeakHunterAction(action_type="install_sensor", target_id="N09"))
    env.step(LeakHunterAction(action_type="close_valve", target_id="P03"))
    obs4 = env.step(
        LeakHunterAction(action_type="read_pressure", target_id="N04")
    )

    assert obs4.summary_text is not None
    assert (0, "pressure:query", "N08") in env._ctx.measurement_cache

    match = re.search(r"\bN08 ([+-]\d+\.\d+)%", obs4.summary_text)
    assert match is not None, obs4.summary_text
    summary_delta = float(match.group(1))

    true_current = env._ctx.current_result.node_pressures_psi.get("N08", 0.0)
    baseline = env._ctx.baseline_nominal.node_pressures_psi.get("N08", 0.0)
    true_delta = (
        round(100.0 * (true_current - baseline) / baseline, 1)
        if baseline > 1e-9
        else 0.0
    )

    assert summary_delta == expected_delta
    assert summary_delta != true_delta
