from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Protocol

import numpy as np

try:
    import wntr

    HAVE_WNTR = True
except Exception:
    wntr = None  # type: ignore[assignment]
    HAVE_WNTR = False

_FALLBACK_WARNING_EMITTED = False

try:
    from .networks import EpisodeNetwork
except ImportError:
    from server.networks import EpisodeNetwork

PSI_PER_METER = 1.42233


@dataclass
class HydraulicResult:
    node_pressures_psi: dict[str, float]
    node_pressures_m: dict[str, float]
    node_heads_m: dict[str, float]
    node_demands_m3s: dict[str, float]
    pipe_flows_m3s: dict[str, float]  # original pipe IDs only
    leak_flow_m3s: float
    delivered_demand_m3s: float
    nominal_demand_m3s: float
    service_fraction: float
    disrupted_demand_frac: float
    leak_frac: float
    solver_ok: bool
    error: str | None = None

    @classmethod
    def synthetic_failure(
        cls,
        network: EpisodeNetwork,
        error: str | None = None,
        initial_leak_flow_m3s: float | None = None,
    ) -> "HydraulicResult":
        exposed_nodes = network.exposed_node_ids
        exposed_pipes = network.pipes.keys()
        nominal = network.total_actual_nominal_demand_m3s(include_confounder=True)
        return cls(
            node_pressures_psi={nid: 0.0 for nid in exposed_nodes},
            node_pressures_m={nid: 0.0 for nid in exposed_nodes},
            node_heads_m={
                nid: network.nodes[nid].elevation_m for nid in exposed_nodes
            },
            node_demands_m3s={nid: 0.0 for nid in exposed_nodes},
            pipe_flows_m3s={pid: 0.0 for pid in exposed_pipes},
            leak_flow_m3s=(
                0.0
                if initial_leak_flow_m3s is None
                else initial_leak_flow_m3s
            ),
            delivered_demand_m3s=0.0,
            nominal_demand_m3s=nominal,
            service_fraction=0.0,
            disrupted_demand_frac=1.0,
            leak_frac=1.0,
            solver_ok=False,
            error=error,
        )


class BaseHydraulicSimulator(Protocol):
    network: EpisodeNetwork
    initial_leak_flow_m3s: float

    def solve(
        self,
        demand_scale: float,
        include_leak: bool,
        include_confounder: bool,
        valve_overrides: dict[str, bool] | None = None,
        leak_area_override_m2: float | None = None,
        extra_closed_pipes: set[str] | None = None,
    ) -> HydraulicResult: ...

    def simulate_repair_outcome(
        self,
        valve_overrides: dict[str, bool],
        target_id: str,
        method: str,
    ) -> HydraulicResult: ...


class WNTRHydraulicSimulator:
    def __init__(self, network: EpisodeNetwork, seed: int) -> None:
        self.network = network
        self.seed = seed
        self.initial_leak_flow_m3s = 0.0
        # Compute initial leak flow
        initial = self.solve(
            demand_scale=1.0,
            include_leak=True,
            include_confounder=True,
            valve_overrides=None,
        )
        self.initial_leak_flow_m3s = initial.leak_flow_m3s

    def solve(
        self,
        demand_scale: float,
        include_leak: bool,
        include_confounder: bool,
        valve_overrides: dict[str, bool] | None = None,
        leak_area_override_m2: float | None = None,
        extra_closed_pipes: set[str] | None = None,
    ) -> HydraulicResult:
        if not HAVE_WNTR:
            raise RuntimeError("WNTR is not installed")
        try:
            wn, leak_flow_segment = self._build_model(
                demand_scale=demand_scale,
                include_leak=include_leak,
                include_confounder=include_confounder,
                valve_overrides=valve_overrides,
                leak_area_override_m2=leak_area_override_m2,
                extra_closed_pipes=extra_closed_pipes or set(),
            )
            sim = wntr.sim.WNTRSimulator(wn)
            results = sim.run_sim()
            t = results.node["pressure"].index[-1]
            pressure_m = results.node["pressure"].loc[t].to_dict()
            head_m = results.node["head"].loc[t].to_dict()
            demand_m3s = results.node["demand"].loc[t].to_dict()
            flow_m3s_physical = results.link["flowrate"].loc[t].to_dict()

            exposed_pressures_m = {
                nid: float(pressure_m.get(nid, 0.0))
                for nid in self.network.exposed_node_ids
            }
            exposed_pressures_psi = {
                nid: PSI_PER_METER * p for nid, p in exposed_pressures_m.items()
            }
            exposed_heads = {
                nid: float(
                    head_m.get(nid, self.network.nodes[nid].elevation_m)
                )
                for nid in self.network.exposed_node_ids
            }
            exposed_demands = {
                nid: float(demand_m3s.get(nid, 0.0))
                for nid in self.network.exposed_node_ids
            }

            # Map physical pipe flows back to original pipe IDs
            original_pipe_flows = {}
            for pid in self.network.pipes:
                if pid == self.network.leak.original_pipe_id and include_leak:
                    # Use the upstream segment flow for the original pipe
                    original_pipe_flows[pid] = float(
                        flow_m3s_physical.get(f"{pid}__a", 0.0)
                    )
                else:
                    original_pipe_flows[pid] = float(
                        flow_m3s_physical.get(pid, 0.0)
                    )

            leak_node_id = self.network.leak.hidden_leak_node_id
            leak_flow = (
                float(demand_m3s.get(leak_node_id, 0.0)) if include_leak else 0.0
            )

            nominal = self.network.total_actual_nominal_demand_m3s(
                include_confounder=include_confounder,
                demand_scale=demand_scale,
            )

            delivered = 0.0
            unmet = 0.0
            for nid in self.network.customer_node_ids:
                actual = float(demand_m3s.get(nid, 0.0))
                nominal_i = self.network.actual_nominal_demand_for(
                    nid,
                    include_confounder=include_confounder,
                    demand_scale=demand_scale,
                )
                delivered += actual
                unmet += max(0.0, nominal_i - actual)

            service_fraction = (
                1.0
                if nominal <= 1e-12
                else max(0.0, min(1.0, delivered / nominal))
            )
            disrupted = (
                0.0 if nominal <= 1e-12 else max(0.0, min(1.0, unmet / nominal))
            )
            leak_frac = (
                0.0
                if (delivered + leak_flow) <= 1e-12
                else leak_flow / (delivered + leak_flow)
            )

            return HydraulicResult(
                node_pressures_psi=exposed_pressures_psi,
                node_pressures_m=exposed_pressures_m,
                node_heads_m=exposed_heads,
                node_demands_m3s=exposed_demands,
                pipe_flows_m3s=original_pipe_flows,
                leak_flow_m3s=leak_flow,
                delivered_demand_m3s=delivered,
                nominal_demand_m3s=nominal,
                service_fraction=service_fraction,
                disrupted_demand_frac=disrupted,
                leak_frac=leak_frac,
                solver_ok=True,
                error=None,
            )
        except Exception as exc:
            return HydraulicResult.synthetic_failure(
                self.network,
                error=str(exc),
                initial_leak_flow_m3s=self.initial_leak_flow_m3s,
            )

    def simulate_repair_outcome(
        self,
        valve_overrides: dict[str, bool],
        target_id: str,
        method: str,
    ) -> HydraulicResult:
        leak_pipe = self.network.leak.original_pipe_id

        if method == "clamp_pipe":
            if target_id == leak_pipe:
                return self.solve(
                    demand_scale=1.0,
                    include_leak=True,
                    include_confounder=True,
                    valve_overrides=valve_overrides,
                    leak_area_override_m2=0.0,
                )
            if target_id in self.network.adjacent_pipe_ids(leak_pipe):
                return self.solve(
                    demand_scale=1.0,
                    include_leak=True,
                    include_confounder=True,
                    valve_overrides=valve_overrides,
                    leak_area_override_m2=0.5 * self.network.leak.area_m2,
                )
            return self.solve(
                demand_scale=1.0,
                include_leak=True,
                include_confounder=True,
                valve_overrides=valve_overrides,
            )

        if method == "replace_section":
            neighborhood = self.network.pipe_ball(target_id, radius=2)
            if leak_pipe in neighborhood:
                return self.solve(
                    demand_scale=1.0,
                    include_leak=True,
                    include_confounder=True,
                    valve_overrides=valve_overrides,
                    leak_area_override_m2=0.0,
                )
            return self.solve(
                demand_scale=1.0,
                include_leak=True,
                include_confounder=True,
                valve_overrides=valve_overrides,
            )

        if method == "isolate_section":
            sec = self.network.section_map[target_id]
            forced_closed = set(sec.boundary_valve_pipe_ids)
            return self.solve(
                demand_scale=1.0,
                include_leak=True,
                include_confounder=True,
                valve_overrides=valve_overrides,
                extra_closed_pipes=forced_closed,
            )

        return self.solve(
            demand_scale=1.0,
            include_leak=True,
            include_confounder=True,
            valve_overrides=valve_overrides,
        )

    def _build_model(
        self,
        demand_scale: float,
        include_leak: bool,
        include_confounder: bool,
        valve_overrides: dict[str, bool] | None,
        leak_area_override_m2: float | None,
        extra_closed_pipes: set[str],
    ):
        wn = wntr.network.WaterNetworkModel()
        wn.options.hydraulic.headloss = "H-W"
        wn.options.hydraulic.demand_model = "PDD"
        wn.options.hydraulic.minimum_pressure = 3.516
        wn.options.hydraulic.required_pressure = 21.097
        wn.options.hydraulic.pressure_exponent = 0.5
        wn.options.time.duration = 60
        wn.options.time.hydraulic_timestep = 60
        wn.options.time.report_timestep = 60

        # Add nodes
        for node in self.network.nodes.values():
            if node.kind == "reservoir":
                wn.add_reservoir(
                    node.node_id,
                    base_head=node.base_head_m,
                    coordinates=(node.x, node.y),
                )
            elif node.kind == "tank":
                wn.add_tank(
                    node.node_id,
                    elevation=node.elevation_m,
                    init_level=node.tank_init_level_m,
                    min_level=node.tank_min_level_m,
                    max_level=node.tank_max_level_m,
                    diameter=node.tank_diameter_m,
                    coordinates=(node.x, node.y),
                )
            else:
                base_demand = self.network.actual_nominal_demand_for(
                    node.node_id,
                    include_confounder=include_confounder,
                    demand_scale=demand_scale,
                )
                wn.add_junction(
                    node.node_id,
                    base_demand=base_demand,
                    elevation=node.elevation_m,
                    coordinates=(node.x, node.y),
                )

        # Add pipes
        valve_state_map = valve_overrides or {}
        for pipe in self.network.pipes.values():
            is_open = valve_state_map.get(pipe.pipe_id, pipe.initial_open)
            if pipe.pipe_id in extra_closed_pipes:
                is_open = False

            if (
                include_leak
                and pipe.pipe_id == self.network.leak.original_pipe_id
            ):
                split = self.network.leak.split_fraction
                leak_node = self.network.leak.hidden_leak_node_id
                x = pipe.start_x + split * (pipe.end_x - pipe.start_x)
                y = pipe.start_y + split * (pipe.end_y - pipe.start_y)
                elev = self.network.interpolate_elevation(
                    pipe.start_node, pipe.end_node, split
                )
                wn.add_junction(
                    leak_node,
                    base_demand=0.0,
                    elevation=elev,
                    coordinates=(x, y),
                )
                wn.add_pipe(
                    f"{pipe.pipe_id}__a",
                    pipe.start_node,
                    leak_node,
                    length=pipe.length_m * split,
                    diameter=pipe.diameter_m,
                    roughness=pipe.roughness_c,
                    minor_loss=0.0,
                    initial_status="OPEN" if is_open else "CLOSED",
                )
                wn.add_pipe(
                    f"{pipe.pipe_id}__b",
                    leak_node,
                    pipe.end_node,
                    length=pipe.length_m * (1.0 - split),
                    diameter=pipe.diameter_m,
                    roughness=pipe.roughness_c,
                    minor_loss=0.0,
                    initial_status="OPEN" if is_open else "CLOSED",
                )
            else:
                wn.add_pipe(
                    pipe.pipe_id,
                    pipe.start_node,
                    pipe.end_node,
                    length=pipe.length_m,
                    diameter=pipe.diameter_m,
                    roughness=pipe.roughness_c,
                    minor_loss=0.0,
                    initial_status="OPEN" if is_open else "CLOSED",
                )

        # Add leak
        if include_leak:
            leak_node = wn.get_node(self.network.leak.hidden_leak_node_id)
            area = (
                self.network.leak.area_m2
                if leak_area_override_m2 is None
                else leak_area_override_m2
            )
            leak_node.add_leak(
                wn,
                area=area,
                discharge_coeff=self.network.leak.discharge_coeff,
                start_time=0,
                end_time=61,
            )

        return wn, self.network.leak.original_pipe_id


class SparseFallbackSimulator:
    """Deterministic approximate fallback. Use only if WNTR unavailable or too slow."""

    def __init__(self, network: EpisodeNetwork, seed: int) -> None:
        self.network = network
        self.seed = seed
        self.initial_leak_flow_m3s = max(
            1e-4,
            0.05 * network.total_actual_nominal_demand_m3s(include_confounder=True),
        )

    def solve(
        self,
        demand_scale: float,
        include_leak: bool,
        include_confounder: bool,
        valve_overrides: dict[str, bool] | None = None,
        leak_area_override_m2: float | None = None,
        extra_closed_pipes: set[str] | None = None,
    ) -> HydraulicResult:
        return HydraulicResult.synthetic_failure(
            self.network,
            error=(
                "WNTR unavailable; SparseFallbackSimulator is a stub and cannot "
                "compute hydraulic results"
            ),
            initial_leak_flow_m3s=self.initial_leak_flow_m3s,
        )

    def simulate_repair_outcome(
        self,
        valve_overrides: dict[str, bool],
        target_id: str,
        method: str,
    ) -> HydraulicResult:
        return self.solve(
            demand_scale=1.0,
            include_leak=True,
            include_confounder=True,
            valve_overrides=valve_overrides,
        )


def make_simulator(
    network: EpisodeNetwork, seed: int
) -> BaseHydraulicSimulator:
    global _FALLBACK_WARNING_EMITTED

    if HAVE_WNTR:
        return WNTRHydraulicSimulator(network, seed)  # type: ignore[return-value]

    if not _FALLBACK_WARNING_EMITTED:
        warnings.warn(
            "WNTR is unavailable; LeakHunter is using SparseFallbackSimulator. "
            "This fallback is a stub and will return solver-failure results "
            "until `wntr` is installed.",
            RuntimeWarning,
            stacklevel=2,
        )
        _FALLBACK_WARNING_EMITTED = True

    return SparseFallbackSimulator(network, seed)  # type: ignore[return-value]
