from __future__ import annotations

import time

import pytest

from models import LeakHunterAction
from server.environment import LeakHunterEnvironment
from server.networks import build_episode_network
from server.simulator import HAVE_WNTR, WNTRHydraulicSimulator


def test_pressure_drop_near_leak():
    env = LeakHunterEnvironment()
    obs = env.reset(difficulty="easy", seed=11)
    baseline = {
        n.node_id: n.baseline_pressure_psi
        for n in obs.nodes
        if n.baseline_pressure_psi is not None
    }
    current = env._ctx.current_result.node_pressures_psi  # white-box test
    worst_drop = min(current[n] - baseline[n] for n in baseline if n in current)
    assert worst_drop < -2.0


def test_valve_closure_changes_hydraulics():
    env = LeakHunterEnvironment()
    env.reset(difficulty="easy", seed=11)
    before = dict(env._ctx.current_result.pipe_flows_m3s)
    env.step(LeakHunterAction(action_type="close_valve", target_id="P05"))
    after = dict(env._ctx.current_result.pipe_flows_m3s)
    changed = [pid for pid in before if abs(after[pid] - before[pid]) > 1e-5]
    assert changed


def test_baseline_ranges_bracket_nominal():
    env = LeakHunterEnvironment()
    obs = env.reset(difficulty="medium", seed=21)
    for n in obs.nodes:
        if (
            n.baseline_pressure_range_psi is None
            or n.baseline_pressure_psi is None
        ):
            continue
        lo, hi = n.baseline_pressure_range_psi
        assert lo <= n.baseline_pressure_psi <= hi


def test_noise_is_reproducible():
    env1 = LeakHunterEnvironment()
    env2 = LeakHunterEnvironment()
    env1.reset(difficulty="easy", seed=11)
    env2.reset(difficulty="easy", seed=11)
    a = LeakHunterAction(action_type="read_pressure", target_id="N08")
    r1 = env1.step(a).sensor_readings[0].value
    r2 = env2.step(a).sensor_readings[0].value
    assert r1 == r2


@pytest.mark.performance
@pytest.mark.skipif(not HAVE_WNTR, reason="WNTR not installed")
def test_hard_solve_time_under_budget():
    net = build_episode_network("hard", 31)
    sim = WNTRHydraulicSimulator(net, seed=31)
    t0 = time.perf_counter()
    for _ in range(30):
        sim.solve(
            demand_scale=1.0, include_leak=True, include_confounder=True
        )
    mean = (time.perf_counter() - t0) / 30.0
    assert mean < 0.35
