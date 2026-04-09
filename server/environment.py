from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
from math import cos, log, pi, sqrt
from typing import Callable, Optional

import numpy as np

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError:
    class Environment:
        """Stub when openenv is not installed."""
        SUPPORTS_CONCURRENT_SESSIONS = False
        def __init_subclass__(cls, **kwargs): pass

try:
    from ..models import (
        ActionRecord,
        LeakHunterAction,
        LeakHunterObservation,
        LeakHunterState,
        ResetObservation,
        RewardBreakdown,
        SensorReading,
    )
    from .networks import EpisodeNetwork, build_episode_network
    from .observations import (
        estimate_tokens,
        format_action_observation,
        format_reset_observation,
        format_running_summary,
    )
    from .rewards import (
        average_service_score,
        build_line_graph,
        compute_dense_reward,
        compute_localization_component,
        compute_repair_success_component,
        compute_water_conservation_component,
        residual_leak_component,
    )
    from .simulator import BaseHydraulicSimulator, HydraulicResult, make_simulator
except ImportError:
    from models import (
        ActionRecord,
        LeakHunterAction,
        LeakHunterObservation,
        LeakHunterState,
        ResetObservation,
        RewardBreakdown,
        SensorReading,
    )
    from server.networks import EpisodeNetwork, build_episode_network
    from server.observations import (
        estimate_tokens,
        format_action_observation,
        format_reset_observation,
        format_running_summary,
    )
    from server.rewards import (
        average_service_score,
        build_line_graph,
        compute_dense_reward,
        compute_localization_component,
        compute_repair_success_component,
        compute_water_conservation_component,
        residual_leak_component,
    )
    from server.simulator import BaseHydraulicSimulator, HydraulicResult, make_simulator


ACTION_COST = {
    "read_pressure": 1,
    "read_flow": 1,
    "install_sensor": 3,
    "close_valve": 5,
    "open_valve": 2,
    "repair": 0,
}

ACTION_MINUTES = {
    "read_pressure": 5,
    "read_flow": 5,
    "install_sensor": 20,
    "close_valve": 15,
    "open_valve": 10,
    "repair": 0,
}


@dataclass
class EpisodeContext:
    difficulty: str
    seed: int
    network: EpisodeNetwork
    simulator: BaseHydraulicSimulator
    baseline_nominal: HydraulicResult
    baseline_low: HydraulicResult
    baseline_high: HydraulicResult
    current_result: HydraulicResult
    budget_total: int
    budget_remaining: int
    elapsed_minutes: int = 0
    hydraulic_revision: int = 0
    step_index: int = 0
    done: bool = False
    forced_repair: bool = False
    queried_nodes: set[str] = field(default_factory=set)
    installed_sensors: set[str] = field(default_factory=set)
    valve_states: dict[str, bool] = field(default_factory=dict)
    measurement_cache: dict[tuple[int, str, str], SensorReading] = field(
        default_factory=dict
    )
    cumulative_dense_reward: float = 0.0
    return_so_far: float = 0.0
    service_integral_min: float = 0.0
    leak_loss_m3: float = 0.0
    action_log: list[ActionRecord] = field(default_factory=list)
    last_guess_target: str | None = None
    final_score: float | None = None
    last_message: str | None = None

    @property
    def pressure_ranges_psi(self) -> dict[str, tuple[float, float]]:
        out: dict[str, tuple[float, float]] = {}
        for node_id, nominal in self.baseline_nominal.node_pressures_psi.items():
            lo = min(
                self.baseline_low.node_pressures_psi.get(node_id, nominal),
                nominal,
                self.baseline_high.node_pressures_psi.get(node_id, nominal),
            )
            hi = max(
                self.baseline_low.node_pressures_psi.get(node_id, nominal),
                nominal,
                self.baseline_high.node_pressures_psi.get(node_id, nominal),
            )
            out[node_id] = (lo, hi)
        return out


class LeakHunterEnvironment(Environment):
    """
    OpenEnv environment for leak localization in a pressurized water network.

    The environment is single-session and deterministic given (difficulty, seed).
    Hydraulic state changes only after valve operations.
    """

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(
        self,
        episode_builder: Callable[[str, int], EpisodeNetwork] | None = None,
        simulator_factory: Callable[[EpisodeNetwork, int], BaseHydraulicSimulator]
        | None = None,
    ) -> None:
        super().__init__()
        self._episode_builder = episode_builder or build_episode_network
        self._simulator_factory = simulator_factory or make_simulator
        self._ctx: EpisodeContext | None = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        difficulty: str = "easy",
        **kwargs,
    ) -> ResetObservation:
        seed = int(seed if seed is not None else 0)
        network = self._episode_builder(difficulty, seed)
        simulator = self._simulator_factory(network, seed)

        baseline_low = simulator.solve(
            demand_scale=0.9, include_leak=False, include_confounder=False
        )
        baseline_nominal = simulator.solve(
            demand_scale=1.0, include_leak=False, include_confounder=False
        )
        baseline_high = simulator.solve(
            demand_scale=1.1, include_leak=False, include_confounder=False
        )
        current_result = simulator.solve(
            demand_scale=1.0, include_leak=True, include_confounder=True
        )

        ctx = EpisodeContext(
            difficulty=difficulty,
            seed=seed,
            network=network,
            simulator=simulator,
            baseline_nominal=baseline_nominal,
            baseline_low=baseline_low,
            baseline_high=baseline_high,
            current_result=current_result,
            budget_total=network.budget_total,
            budget_remaining=network.budget_total,
            valve_states={
                pid: pipe.initial_open
                for pid, pipe in network.pipes.items()
                if pipe.has_valve
            },
        )
        self._ctx = ctx
        reset_obs = self._build_reset_observation(episode_id=episode_id)
        return reset_obs

    def step(
        self,
        action: LeakHunterAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> LeakHunterObservation:
        if self._ctx is None:
            return self._make_error_observation(
                "Call reset() first.", reward=0.0, done=False
            )
        ctx = self._ctx
        if ctx.done:
            return self._make_error_observation(
                "Episode already finished.", reward=0.0, done=True
            )

        episode_return_before = ctx.return_so_far
        action_text = action.as_command()
        ctx.step_index += 1
        ctx.last_guess_target = action.target_id
        prev_result = ctx.current_result

        # Validate action
        invalid_reason = self._validate_action(action)
        if invalid_reason:
            self._apply_budget_only(cost=1)
            dense = compute_dense_reward(
                prev_result,
                prev_result,
                action_cost=1,
                budget_total=ctx.budget_total,
            )
            ctx.cumulative_dense_reward += dense
            if ctx.budget_remaining <= 0:
                return self._finish_forced_repair(
                    reason=f"Invalid action: {invalid_reason}",
                    action_text=action_text,
                    episode_return_before=episode_return_before,
                )
            ctx.return_so_far += dense
            message = f"ERROR: {invalid_reason}. Budget -1."
            return self._make_step_observation(
                action_text=action_text,
                message=message,
                readings=[],
                auto_reports=[],
                reward=dense,
                done=False,
                breakdown=RewardBreakdown(dense_component=dense),
            )

        # Explicit repair is terminal and consumes no time
        if action.action_type == "repair":
            return self._finish_repair(
                target_id=action.target_id,
                method=action.method,
                action_text=action_text,
                forced=False,
                action_prefix_message="Repair submitted.",
                episode_return_before=episode_return_before,
            )

        cost = ACTION_COST[action.action_type]
        minutes = ACTION_MINUTES[action.action_type]
        self._integrate_current_state(minutes)
        ctx.elapsed_minutes += minutes
        ctx.budget_remaining = max(0, ctx.budget_remaining - cost)

        readings: list[SensorReading] = []
        auto_reports: list[SensorReading] = []

        if action.action_type == "read_pressure":
            readings.append(self._read_pressure(action.target_id, source="query"))
            dense = compute_dense_reward(
                prev_result, prev_result, action_cost=cost, budget_total=ctx.budget_total
            )
        elif action.action_type == "read_flow":
            readings.append(self._read_flow(action.target_id))
            dense = compute_dense_reward(
                prev_result, prev_result, action_cost=cost, budget_total=ctx.budget_total
            )
        elif action.action_type == "install_sensor":
            ctx.installed_sensors.add(action.target_id)
            dense = compute_dense_reward(
                prev_result, prev_result, action_cost=cost, budget_total=ctx.budget_total
            )
        elif action.action_type == "close_valve":
            ctx.valve_states[action.target_id] = False
            ctx.hydraulic_revision += 1
            ctx.current_result = self._solve_current_hydraulics()
            auto_reports = self._auto_report_sensors()
            dense = compute_dense_reward(
                prev_result,
                ctx.current_result,
                action_cost=cost,
                budget_total=ctx.budget_total,
            )
        elif action.action_type == "open_valve":
            ctx.valve_states[action.target_id] = True
            ctx.hydraulic_revision += 1
            ctx.current_result = self._solve_current_hydraulics()
            auto_reports = self._auto_report_sensors()
            dense = compute_dense_reward(
                prev_result,
                ctx.current_result,
                action_cost=cost,
                budget_total=ctx.budget_total,
            )
        else:
            dense = 0.0

        ctx.cumulative_dense_reward += dense

        if ctx.budget_remaining <= 0:
            return self._finish_forced_repair(
                reason="Budget exhausted.",
                action_text=action_text,
                episode_return_before=episode_return_before,
                carried_readings=readings,
                carried_auto_reports=auto_reports,
            )

        ctx.return_so_far += dense
        return self._make_step_observation(
            action_text=action_text,
            message="OK",
            readings=readings,
            auto_reports=auto_reports,
            reward=dense,
            done=False,
            breakdown=RewardBreakdown(dense_component=dense),
        )

    @property
    def state(self) -> LeakHunterState:
        if self._ctx is None:
            return LeakHunterState()
        ctx = self._ctx
        return LeakHunterState(
            episode_id=None,
            step_count=ctx.step_index,
            difficulty=ctx.difficulty,
            seed=ctx.seed,
            budget_total=ctx.budget_total,
            budget_remaining=ctx.budget_remaining,
            elapsed_minutes=ctx.elapsed_minutes,
            hydraulic_revision=ctx.hydraulic_revision,
            installed_sensors=sorted(ctx.installed_sensors),
            valve_states=dict(ctx.valve_states),
            cumulative_dense_reward=ctx.cumulative_dense_reward,
            return_so_far=ctx.return_so_far,
            final_score=ctx.final_score,
            forced_repair=ctx.forced_repair,
            done=ctx.done,
            last_action=ctx.action_log[-1].action_text if ctx.action_log else None,
            last_message=ctx.last_message,
            action_log=list(ctx.action_log[-50:]),
        )

    # ---------- helpers ----------

    def _build_reset_observation(self, episode_id: str | None) -> ResetObservation:
        assert self._ctx is not None
        ctx = self._ctx
        obs = ResetObservation(
            done=False,
            reward=None,
            difficulty=ctx.difficulty,
            seed=ctx.seed,
            network_name=ctx.network.name,
            budget_total=ctx.budget_total,
            budget_remaining=ctx.budget_remaining,
            elapsed_minutes=0,
            hydraulic_revision=0,
            text="",
            last_action=None,
            nodes=ctx.network.public_nodes(
                ctx.baseline_nominal, ctx.pressure_ranges_psi
            ),
            pipes=ctx.network.public_pipes(ctx.baseline_nominal, ctx.valve_states),
            sections=ctx.network.public_sections(),
            zones=ctx.network.public_zones(),
            token_estimate=0,
            metadata={"episode_id": episode_id},
        )
        obs.text = format_reset_observation(obs)
        obs.token_estimate = estimate_tokens(obs.text)
        return obs

    def _validate_action(self, action: LeakHunterAction) -> str | None:
        assert self._ctx is not None
        ctx = self._ctx
        net = ctx.network

        if action.action_type in {"read_pressure", "install_sensor"}:
            if action.target_id not in net.exposed_node_ids:
                return f"Unknown node {action.target_id}"
            node = net.nodes[action.target_id]
            if node.kind != "junction":
                return (
                    f"Cannot target {node.kind} {action.target_id} — "
                    "only junctions allowed"
                )
            if (
                action.action_type == "install_sensor"
                and action.target_id in ctx.installed_sensors
            ):
                return f"Sensor already installed at {action.target_id}"
            return None

        if action.action_type in {"read_flow", "close_valve", "open_valve"}:
            if action.target_id not in net.pipes:
                return f"Unknown pipe {action.target_id}"
            pipe = net.pipes[action.target_id]
            if action.action_type in {"close_valve", "open_valve"} and not pipe.has_valve:
                return f"Pipe {action.target_id} has no controllable valve"
            if action.action_type == "close_valve" and not ctx.valve_states.get(
                action.target_id, pipe.initial_open
            ):
                return f"Valve on {action.target_id} is already closed"
            if action.action_type == "open_valve" and ctx.valve_states.get(
                action.target_id, pipe.initial_open
            ):
                return f"Valve on {action.target_id} is already open"
            return None

        if action.action_type == "repair":
            if (
                action.method == "isolate_section"
                and action.target_id not in net.section_ids
            ):
                return f"Unknown section {action.target_id}"
            if action.method in {"clamp_pipe", "replace_section"} and action.target_id not in net.pipes:
                return f"Unknown pipe {action.target_id}"
            return None

        return f"Unsupported action {action.action_type}"

    def _integrate_current_state(self, minutes: int) -> None:
        assert self._ctx is not None
        ctx = self._ctx
        if minutes <= 0:
            return
        ctx.service_integral_min += ctx.current_result.service_fraction * minutes
        ctx.leak_loss_m3 += ctx.current_result.leak_flow_m3s * minutes * 60.0

    def _apply_budget_only(self, cost: int) -> None:
        assert self._ctx is not None
        self._ctx.budget_remaining = max(0, self._ctx.budget_remaining - cost)

    def _solve_current_hydraulics(self) -> HydraulicResult:
        assert self._ctx is not None
        ctx = self._ctx
        try:
            return ctx.simulator.solve(
                demand_scale=1.0,
                include_leak=True,
                include_confounder=True,
                valve_overrides=ctx.valve_states,
            )
        except Exception as exc:
            return HydraulicResult.synthetic_failure(ctx.network, error=str(exc))

    def _deterministic_normal(self, channel: str, target_id: str) -> float:
        assert self._ctx is not None
        ctx = self._ctx
        raw = f"{ctx.seed}|{ctx.step_index}|{ctx.hydraulic_revision}|{channel}|{target_id}".encode()
        h = blake2b(raw, digest_size=16).digest()
        u1 = (int.from_bytes(h[:8], "big") + 1) / (2**64 + 2)
        u2 = (int.from_bytes(h[8:], "big") + 1) / (2**64 + 2)
        return sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)

    def _read_pressure(self, node_id: str, source: str) -> SensorReading:
        assert self._ctx is not None
        ctx = self._ctx
        key = (ctx.hydraulic_revision, f"pressure:{source}", node_id)
        if key in ctx.measurement_cache:
            ctx.queried_nodes.add(node_id)
            return ctx.measurement_cache[key]

        true_psi = ctx.current_result.node_pressures_psi.get(node_id, 0.0)
        bias = ctx.network.sensor_bias_psi_for(node_id)
        sigma = ctx.network.pressure_noise_sigma_psi
        noisy = true_psi + bias + sigma * self._deterministic_normal(
            f"pressure:{source}", node_id
        )

        baseline = ctx.baseline_nominal.node_pressures_psi.get(node_id, 0.0)
        delta_pct = (
            None
            if baseline <= 1e-9
            else 100.0 * (noisy - baseline) / baseline
        )

        if (not ctx.current_result.solver_ok) or noisy <= 5.0:
            reading = SensorReading(
                reading_type="pressure",
                target_id=node_id,
                source=source,
                value=None,
                units="PSI",
                delta_percent_from_baseline=delta_pct,
                baseline_value=baseline,
                status="solver_failed"
                if not ctx.current_result.solver_ok
                else "no_service",
                text=f"{node_id} pressure = NO SERVICE — below minimum pressure",
            )
        else:
            reading = SensorReading(
                reading_type="pressure",
                target_id=node_id,
                source=source,
                value=round(noisy, 2),
                units="PSI",
                delta_percent_from_baseline=None
                if delta_pct is None
                else round(delta_pct, 1),
                baseline_value=round(baseline, 2),
                status="ok",
                text=f"{node_id} pressure = {noisy:.2f} PSI ({delta_pct:+.1f}% vs baseline)"
                if delta_pct is not None
                else f"{node_id} pressure = {noisy:.2f} PSI",
            )

        ctx.measurement_cache[key] = reading
        ctx.queried_nodes.add(node_id)
        return reading

    def _read_flow(self, pipe_id: str) -> SensorReading:
        assert self._ctx is not None
        ctx = self._ctx
        key = (ctx.hydraulic_revision, "flow:query", pipe_id)
        if key in ctx.measurement_cache:
            return ctx.measurement_cache[key]

        true_q = ctx.current_result.pipe_flows_m3s.get(pipe_id, 0.0)
        baseline_q = ctx.baseline_nominal.pipe_flows_m3s.get(pipe_id, 0.0)

        if abs(true_q) < 1e-12:
            noisy = 0.0
        else:
            sigma = max(
                0.03 * max(abs(baseline_q), abs(true_q), 1e-4), 1e-4
            )
            noisy = true_q + sigma * self._deterministic_normal(
                "flow:query", pipe_id
            )

        delta_pct = (
            None
            if abs(baseline_q) <= 1e-9
            else 100.0 * (noisy - baseline_q) / abs(baseline_q)
        )

        reading = SensorReading(
            reading_type="flow",
            target_id=pipe_id,
            source="query",
            value=round(noisy, 5),
            units="m3/s",
            delta_percent_from_baseline=None
            if delta_pct is None
            else round(delta_pct, 1),
            baseline_value=round(baseline_q, 5),
            status="ok" if ctx.current_result.solver_ok else "solver_failed",
            text=ctx.network.describe_flow_reading(
                pipe_id, noisy, baseline_q, delta_pct
            ),
        )
        ctx.measurement_cache[key] = reading
        return reading

    def _auto_report_sensors(self) -> list[SensorReading]:
        assert self._ctx is not None
        return [
            self._read_pressure(node_id, source="auto")
            for node_id in sorted(self._ctx.installed_sensors)
        ]

    def _forced_guess(self) -> tuple[str, str]:
        assert self._ctx is not None
        ctx = self._ctx
        fallback_section = ctx.network.sections[0].section_id if ctx.network.sections else "S1"
        guess = ctx.last_guess_target
        if guess is None:
            return (fallback_section, "isolate_section")
        if guess.startswith("P") and guess in ctx.network.pipes:
            return (guess, "replace_section")
        if guess.startswith("S") and guess in ctx.network.section_ids:
            return (guess, "isolate_section")
        if not guess.startswith("P") and not guess.startswith("S"):
            sec_ids = ctx.network.node_to_section_ids.get(guess, [])
            if sec_ids:
                return (sec_ids[0], "isolate_section")
        return (fallback_section, "isolate_section")

    def _finish_forced_repair(
        self,
        reason: str,
        action_text: str,
        episode_return_before: float,
        carried_readings: list[SensorReading] | None = None,
        carried_auto_reports: list[SensorReading] | None = None,
    ) -> LeakHunterObservation:
        target_id, method = self._forced_guess()
        self._ctx.forced_repair = True
        return self._finish_repair(
            target_id=target_id,
            method=method,
            action_text=action_text,
            forced=True,
            action_prefix_message=reason,
            episode_return_before=episode_return_before,
            carried_readings=carried_readings or [],
            carried_auto_reports=carried_auto_reports or [],
        )

    def _finish_repair(
        self,
        target_id: str,
        method: str | None,
        action_text: str,
        forced: bool,
        action_prefix_message: str,
        episode_return_before: float,
        carried_readings: list[SensorReading] | None = None,
        carried_auto_reports: list[SensorReading] | None = None,
    ) -> LeakHunterObservation:
        assert self._ctx is not None
        ctx = self._ctx
        method = method or "isolate_section"

        line_graph = build_line_graph(ctx.network)

        repair_component, distance = compute_repair_success_component(
            network=ctx.network,
            line_graph=line_graph,
            target_id=target_id,
            method=method,
        )
        localization_component = compute_localization_component(
            network=ctx.network,
            line_graph=line_graph,
            target_id=target_id,
            method=method,
            distance=distance,
        )
        post_result = ctx.simulator.simulate_repair_outcome(
            valve_overrides=ctx.valve_states,
            target_id=target_id,
            method=method,
        )
        residual_component = residual_leak_component(
            initial_leak_flow_m3s=ctx.simulator.initial_leak_flow_m3s,
            post_repair_leak_flow_m3s=post_result.leak_flow_m3s,
        )
        service_component = average_service_score(
            service_integral_min=ctx.service_integral_min,
            elapsed_minutes=ctx.elapsed_minutes,
            current_service_fraction=ctx.current_result.service_fraction,
        )
        water_component = compute_water_conservation_component(
            cumulative_loss_m3=ctx.leak_loss_m3,
            loss_cap_m3=ctx.network.loss_cap_m3(ctx.simulator.initial_leak_flow_m3s),
        )

        terminal_component = (
            repair_component
            + localization_component
            + residual_component
            + service_component
            + water_component
        )
        final_score = float(
            np.clip(terminal_component + ctx.cumulative_dense_reward, 0.01, 0.99)
        )
        step_reward = final_score - episode_return_before

        breakdown = RewardBreakdown(
            dense_component=ctx.cumulative_dense_reward - episode_return_before,
            terminal_component=terminal_component,
            repair_component=repair_component,
            localization_component=localization_component,
            residual_leak_component=residual_component,
            service_component=service_component,
            water_component=water_component,
            final_score=final_score,
        )

        ctx.final_score = final_score
        ctx.return_so_far = final_score
        ctx.done = True

        message = (
            f"{action_prefix_message} "
            f"{'FORCED ' if forced else ''}repair={target_id} {method}. "
            f"Final score {final_score:.3f}."
        )

        obs = self._make_step_observation(
            action_text=action_text
            if not forced
            else f"{action_text} -> forced repair {target_id} {method}",
            message=message,
            readings=carried_readings or [],
            auto_reports=carried_auto_reports or [],
            reward=step_reward,
            done=True,
            breakdown=breakdown,
            terminal=True,
        )
        return obs

    def _make_step_observation(
        self,
        action_text: str,
        message: str,
        readings: list[SensorReading],
        auto_reports: list[SensorReading],
        reward: float,
        done: bool,
        breakdown: RewardBreakdown,
        terminal: bool = False,
    ) -> LeakHunterObservation:
        assert self._ctx is not None
        ctx = self._ctx

        summary = (
            format_running_summary(ctx)
            if (ctx.step_index % 4 == 0 or done)
            else None
        )
        alerts = (
            []
            if ctx.current_result.solver_ok
            else [
                f"Solver failure: {ctx.current_result.error or 'unknown hydraulic solver error'}"
            ]
        )
        text = format_action_observation(
            action_text=action_text,
            message=message,
            readings=readings,
            auto_reports=auto_reports,
            alerts=alerts,
            budget_remaining=ctx.budget_remaining,
            budget_total=ctx.budget_total,
            elapsed_minutes=ctx.elapsed_minutes,
            hydraulic_revision=ctx.hydraulic_revision,
            summary_text=summary,
            terminal=terminal,
        )

        obs = LeakHunterObservation(
            done=done,
            reward=reward,
            observation_type="terminal" if terminal else "step",
            text=text,
            difficulty=ctx.difficulty,
            budget_total=ctx.budget_total,
            budget_remaining=ctx.budget_remaining,
            elapsed_minutes=ctx.elapsed_minutes,
            hydraulic_revision=ctx.hydraulic_revision,
            last_action=action_text,
            sensor_readings=[*readings, *auto_reports],
            alerts=alerts,
            summary_text=summary,
            reward_breakdown=breakdown,
        )

        ctx.last_message = message
        ctx.action_log.append(
            ActionRecord(
                step_index=ctx.step_index,
                action_text=action_text,
                reward=reward,
                budget_remaining=ctx.budget_remaining,
                elapsed_minutes=ctx.elapsed_minutes,
                hydraulic_revision=ctx.hydraulic_revision,
                message=message,
            )
        )
        return obs

    def _make_error_observation(
        self, message: str, reward: float, done: bool
    ) -> LeakHunterObservation:
        difficulty = self._ctx.difficulty if self._ctx else "easy"
        budget_total = self._ctx.budget_total if self._ctx else 0
        budget_remaining = self._ctx.budget_remaining if self._ctx else 0
        elapsed = self._ctx.elapsed_minutes if self._ctx else 0
        rev = self._ctx.hydraulic_revision if self._ctx else 0
        return LeakHunterObservation(
            done=done,
            reward=reward,
            observation_type="error",
            text=message,
            difficulty=difficulty,
            budget_total=budget_total,
            budget_remaining=budget_remaining,
            elapsed_minutes=elapsed,
            hydraulic_revision=rev,
            alerts=[message],
        )
