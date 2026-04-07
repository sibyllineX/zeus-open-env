from __future__ import annotations

from typing import Iterable

try:
    from ..models import ResetObservation, SensorReading
except ImportError:
    from models import ResetObservation, SensorReading


def estimate_tokens(text: str) -> int:
    # 4 chars/token heuristic is sufficient for budget control here.
    return (len(text) + 3) // 4


def _reading_lines(readings: Iterable[SensorReading]) -> list[str]:
    return [f"- {r.text}" for r in readings]


def format_reset_observation(obs: ResetObservation) -> str:
    total_demand = max(sum(zz.nominal_demand_m3s for zz in obs.zones), 1e-9)
    zone_txt = " | ".join(
        f"{z.zone_id}: {','.join(z.node_ids)} "
        f"({100.0 * z.nominal_demand_m3s / total_demand:.1f}% nominal demand)"
        for z in obs.zones
    )

    node_lines = [
        f"{n.node_id} {n.kind} elev={n.elevation_m:.1f}m zone={n.zone_id} "
        f"baseP={n.baseline_pressure_psi:.1f}psi "
        f"range=[{n.baseline_pressure_range_psi[0]:.1f},{n.baseline_pressure_range_psi[1]:.1f}]"
        if n.baseline_pressure_psi is not None and n.baseline_pressure_range_psi is not None
        else f"{n.node_id} {n.kind} elev={n.elevation_m:.1f}m zone={n.zone_id}"
        for n in obs.nodes
    ]

    pipe_lines = [
        f"{p.pipe_id} {p.start_node}->{p.end_node} {p.diameter_mm:.0f}mm "
        f"valve={'yes' if p.has_valve else 'no'} state={'OPEN' if p.is_open else 'CLOSED'} "
        f"base_dir={p.baseline_direction}"
        for p in obs.pipes
    ]

    section_lines = [
        f"{s.section_id} pipes={','.join(s.pipe_ids)} "
        f"boundary={','.join(s.boundary_valve_pipe_ids)} "
        f"demand={100.0 * s.demand_fraction:.1f}%"
        for s in obs.sections
    ]

    parts = [
        f"LeakHunter {obs.difficulty.upper()} | budget {obs.budget_remaining}/{obs.budget_total} | elapsed 0 min",
        "Conventions: pressure in PSI; percentages are versus healthy baseline midpoint; "
        "baseline flow directions are shown below; current flow requires read_flow; "
        "installed sensors auto-report after valve operations.",
        f"Zones: {zone_txt}",
        "Nodes:",
        *node_lines,
        "Pipes:",
        *pipe_lines,
        "Sections:",
        *section_lines,
        "Repair methods: clamp_pipe = exact pipe best, replace_section = pipe within 2 hops, "
        "isolate_section = named reset-time section.",
    ]
    return "\n".join(parts)


def format_action_observation(
    action_text: str,
    message: str,
    readings: list[SensorReading],
    auto_reports: list[SensorReading],
    alerts: list[str],
    budget_remaining: int,
    budget_total: int,
    elapsed_minutes: int,
    hydraulic_revision: int,
    summary_text: str | None,
    terminal: bool = False,
) -> str:
    lines = [
        f"Action: {action_text}",
        f"Result: {message}",
    ]
    if readings:
        lines.append("Readings:")
        lines.extend(_reading_lines(readings))
    if auto_reports:
        lines.append("Auto-reports:")
        lines.extend(_reading_lines(auto_reports))
    if alerts:
        lines.append("Alerts:")
        lines.extend(f"- {alert}" for alert in alerts)
    lines.append(
        f"Status: budget {budget_remaining}/{budget_total} | "
        f"time {elapsed_minutes} min | hydraulic revision {hydraulic_revision}"
    )
    if summary_text:
        lines.append(summary_text)
    return "\n".join(lines)


def format_running_summary(ctx) -> str:
    recent = ctx.action_log[-4:]
    action_txt = "; ".join(a.action_text for a in recent) if recent else "none"
    valves = (
        ",".join(
            f"{pid}:{'OPEN' if is_open else 'CLOSED'}"
            for pid, is_open in sorted(ctx.valve_states.items())
        )
        or "none"
    )
    sensors = ",".join(sorted(ctx.installed_sensors)) or "none"
    anomalies = []
    for key, reading in ctx.measurement_cache.items():
        rev, channel, node_id = key
        if not channel.startswith("pressure:"):
            continue
        if reading.value is None or reading.baseline_value is None:
            continue
        if reading.baseline_value <= 1e-9:
            continue
        delta = (
            100.0
            * (reading.value - reading.baseline_value)
            / reading.baseline_value
        )
        anomalies.append((delta, node_id))
    anomalies.sort(key=lambda x: x[0])
    top = (
        ", ".join(f"{node} {delta:+.1f}%" for delta, node in anomalies[:3])
        if anomalies
        else "none"
    )
    return (
        f"SUMMARY | last actions: {action_txt} | valves: {valves} | "
        f"sensors: {sensors} | largest pressure deltas: {top}"
    )
