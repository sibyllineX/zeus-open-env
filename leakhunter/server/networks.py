from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import networkx as nx
import numpy as np

try:
    from ..models import NodeInfo, PipeInfo, SectionInfo, ZoneInfo
except ImportError:
    from models import NodeInfo, PipeInfo, SectionInfo, ZoneInfo


@dataclass(frozen=True)
class NodeDef:
    node_id: str
    kind: Literal["junction", "reservoir", "tank"]
    elevation_m: float
    zone_id: str
    x: float
    y: float
    base_demand_m3s: float = 0.0
    base_head_m: float | None = None
    tank_init_level_m: float | None = None
    tank_min_level_m: float | None = None
    tank_max_level_m: float | None = None
    tank_diameter_m: float | None = None


@dataclass(frozen=True)
class PipeDef:
    pipe_id: str
    start_node: str
    end_node: str
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    length_m: float
    diameter_m: float
    roughness_c: float
    has_valve: bool
    initial_open: bool


@dataclass(frozen=True)
class LeakDef:
    original_pipe_id: str
    area_m2: float
    split_fraction: float
    discharge_coeff: float
    hidden_leak_node_id: str


@dataclass
class SectionDef:
    section_id: str
    pipe_ids: list[str]
    boundary_valve_pipe_ids: list[str]
    demand_fraction: float = 0.0


@dataclass(frozen=True)
class ConfounderDef:
    confounder_type: Literal["none", "demand_spike", "sensor_bias"]
    zone_id: str | None = None
    node_id: str | None = None
    multiplier: float | None = None
    bias_psi: float | None = None


@dataclass
class EpisodeNetwork:
    name: str
    difficulty: str
    seed: int
    nodes: dict[str, NodeDef]
    pipes: dict[str, PipeDef]
    zones: dict[str, list[str]]
    sections: list[SectionDef]
    leak: LeakDef
    confounder: ConfounderDef
    budget_total: int
    pressure_noise_sigma_psi: float
    flow_noise_rel_sigma: float

    @property
    def exposed_node_ids(self) -> list[str]:
        return [nid for nid, n in self.nodes.items() if not nid.startswith("LEAK_")]

    @property
    def customer_node_ids(self) -> list[str]:
        return [
            nid
            for nid, n in self.nodes.items()
            if n.kind == "junction" and not nid.startswith("LEAK_")
        ]

    @property
    def section_ids(self) -> set[str]:
        return {s.section_id for s in self.sections}

    @property
    def section_map(self) -> dict[str, SectionDef]:
        return {s.section_id: s for s in self.sections}

    @property
    def node_to_section_ids(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for sec in self.sections:
            for pid in sec.pipe_ids:
                p = self.pipes[pid]
                out.setdefault(p.start_node, []).append(sec.section_id)
                out.setdefault(p.end_node, []).append(sec.section_id)
        return {k: sorted(set(v)) for k, v in out.items()}

    def sensor_bias_psi_for(self, node_id: str) -> float:
        if self.confounder.confounder_type == "sensor_bias" and self.confounder.node_id == node_id:
            return float(self.confounder.bias_psi or 0.0)
        return 0.0

    def interpolate_elevation(self, a: str, b: str, frac: float) -> float:
        za = self.nodes[a].elevation_m
        zb = self.nodes[b].elevation_m
        return za + frac * (zb - za)

    def actual_nominal_demand_for(
        self, node_id: str, include_confounder: bool, demand_scale: float = 1.0
    ) -> float:
        node = self.nodes[node_id]
        if node.kind != "junction":
            return 0.0
        base = node.base_demand_m3s * demand_scale
        if (
            include_confounder
            and self.confounder.confounder_type == "demand_spike"
            and self.confounder.zone_id == node.zone_id
        ):
            base *= float(self.confounder.multiplier or 1.0)
        return base

    def total_actual_nominal_demand_m3s(
        self, include_confounder: bool, demand_scale: float = 1.0
    ) -> float:
        return sum(
            self.actual_nominal_demand_for(nid, include_confounder, demand_scale)
            for nid in self.customer_node_ids
        )

    def loss_cap_m3(self, initial_leak_flow_m3s: float) -> float:
        cap_minutes = {"easy": 180, "medium": 240, "hard": 300}[self.difficulty]
        return initial_leak_flow_m3s * cap_minutes * 60.0

    def adjacent_pipe_ids(self, pipe_id: str) -> set[str]:
        p = self.pipes[pipe_id]
        out = set()
        for qid, q in self.pipes.items():
            if qid == pipe_id:
                continue
            if len({p.start_node, p.end_node}.intersection({q.start_node, q.end_node})) > 0:
                out.add(qid)
        return out

    def pipe_ball(self, center_pipe_id: str, radius: int) -> set[str]:
        g = nx.Graph()
        for pid in self.pipes:
            g.add_node(pid)
        for a, pa in self.pipes.items():
            for b, pb in self.pipes.items():
                if a >= b:
                    continue
                if len({pa.start_node, pa.end_node}.intersection({pb.start_node, pb.end_node})) > 0:
                    g.add_edge(a, b)
        lengths = nx.single_source_shortest_path_length(g, center_pipe_id, cutoff=radius)
        return set(lengths.keys())

    def describe_flow_reading(
        self, pipe_id: str, q: float, baseline_q: float, delta_pct: float | None
    ) -> str:
        p = self.pipes[pipe_id]
        if abs(q) < 1e-9:
            direction = "zero flow"
        elif q >= 0:
            direction = f"{p.start_node}->{p.end_node}"
        else:
            direction = f"{p.end_node}->{p.start_node}"
        base_dir = (
            "start_to_end"
            if baseline_q > 0
            else "end_to_start" if baseline_q < 0 else "closed"
        )
        if delta_pct is None:
            return f"{pipe_id} flow = {q:.5f} m3/s ({direction}; no healthy baseline)"
        return f"{pipe_id} flow = {q:.5f} m3/s ({direction}; {delta_pct:+.1f}% vs baseline {base_dir})"

    def public_nodes(
        self, baseline_nominal, pressure_ranges: dict[str, tuple[float, float]]
    ) -> list[NodeInfo]:
        out = []
        for nid in self.exposed_node_ids:
            n = self.nodes[nid]
            out.append(
                NodeInfo(
                    node_id=nid,
                    kind=n.kind,
                    elevation_m=n.elevation_m,
                    zone_id=n.zone_id,
                    coordinates_xy=(n.x, n.y),
                    base_demand_m3s=n.base_demand_m3s,
                    baseline_pressure_psi=baseline_nominal.node_pressures_psi.get(nid),
                    baseline_pressure_range_psi=pressure_ranges.get(nid),
                )
            )
        return out

    def public_pipes(
        self, baseline_nominal, valve_states: dict[str, bool]
    ) -> list[PipeInfo]:
        out = []
        for pid, p in self.pipes.items():
            q = baseline_nominal.pipe_flows_m3s.get(pid, 0.0)
            direction = (
                "start_to_end"
                if q > 0
                else "end_to_start"
                if q < 0
                else ("closed" if not p.initial_open else "zero")
            )
            out.append(
                PipeInfo(
                    pipe_id=pid,
                    start_node=p.start_node,
                    end_node=p.end_node,
                    length_m=p.length_m,
                    diameter_mm=1000.0 * p.diameter_m,
                    roughness_c=p.roughness_c,
                    has_valve=p.has_valve,
                    is_open=valve_states.get(pid, p.initial_open),
                    baseline_flow_m3s=q,
                    baseline_direction=direction,
                )
            )
        return out

    def public_sections(self) -> list[SectionInfo]:
        return [
            SectionInfo(
                section_id=s.section_id,
                pipe_ids=s.pipe_ids,
                boundary_valve_pipe_ids=s.boundary_valve_pipe_ids,
                demand_fraction=s.demand_fraction,
            )
            for s in self.sections
        ]

    def public_zones(self) -> list[ZoneInfo]:
        return [
            ZoneInfo(
                zone_id=zid,
                node_ids=nodes,
                nominal_demand_m3s=sum(
                    self.nodes[n].base_demand_m3s
                    for n in nodes
                    if self.nodes[n].kind == "junction"
                ),
            )
            for zid, nodes in self.zones.items()
        ]


def build_episode_network(difficulty: str, seed: int) -> EpisodeNetwork:
    if difficulty not in ("easy", "medium", "hard"):
        raise ValueError(
            f"Unknown difficulty '{difficulty}'. Expected one of: easy, medium, hard"
        )

    rng = np.random.default_rng(seed)

    if difficulty == "easy":
        nodes, pipes, zones = _build_easy(rng)
        budget_total = 20
        pressure_sigma = 0.5
        leak_area = float(rng.uniform(0.002, 0.005))
        confounder = ConfounderDef("none")
    elif difficulty == "medium":
        nodes, pipes, zones = _build_medium(rng)
        budget_total = 25
        pressure_sigma = 2.0
        leak_area = float(rng.uniform(0.001, 0.003))
    else:
        nodes, pipes, zones = _build_hard(rng)
        budget_total = 28
        pressure_sigma = 3.5
        leak_area = float(rng.uniform(0.0005, 0.002))

    leak_pipe_id = _sample_leak_pipe(rng, pipes)
    leak_pipe = pipes[leak_pipe_id]

    if difficulty == "easy":
        confounder = ConfounderDef("none")
    elif difficulty == "medium":
        leak_zone = nodes[leak_pipe.start_node].zone_id
        confounder = _sample_medium_confounder(rng, zones, leak_zone)
    else:
        leak_nodes = {leak_pipe.start_node, leak_pipe.end_node}
        confounder = _sample_hard_confounder(rng, nodes, pipes, leak_nodes)

    leak = LeakDef(
        original_pipe_id=leak_pipe_id,
        area_m2=leak_area,
        split_fraction=float(rng.uniform(0.25, 0.75)),
        discharge_coeff=0.75,
        hidden_leak_node_id=f"LEAK_{leak_pipe_id}",
    )

    sections = compute_sections(nodes, pipes)

    return EpisodeNetwork(
        name=f"leakhunter_{difficulty}",
        difficulty=difficulty,
        seed=seed,
        nodes=nodes,
        pipes=pipes,
        zones=zones,
        sections=sections,
        leak=leak,
        confounder=confounder,
        budget_total=budget_total,
        pressure_noise_sigma_psi=pressure_sigma,
        flow_noise_rel_sigma=0.03,
    )


def _node(
    node_id, kind, elev, zone, x, y, demand_lps=0.0, head=None, tank=None
) -> NodeDef:
    tank = tank or {}
    return NodeDef(
        node_id=node_id,
        kind=kind,
        elevation_m=elev,
        zone_id=zone,
        x=x,
        y=y,
        base_demand_m3s=demand_lps / 1000.0,
        base_head_m=head,
        tank_init_level_m=tank.get("init"),
        tank_min_level_m=tank.get("min"),
        tank_max_level_m=tank.get("max"),
        tank_diameter_m=tank.get("diam"),
    )


def _pipe(pipe_id, a, b, nodes, rng, diameter_mm, valve=False, initial_open=True) -> PipeDef:
    na, nb = nodes[a], nodes[b]
    dx = nb.x - na.x
    dy = nb.y - na.y
    nominal_len = 300.0 * max(abs(dx) + abs(dy), 1.0)
    length_m = nominal_len * float(rng.uniform(0.9, 1.1))
    diameter_m = (diameter_mm / 1000.0) * float(rng.uniform(0.95, 1.05))
    roughness = float(rng.uniform(115.0, 130.0))
    return PipeDef(
        pipe_id=pipe_id,
        start_node=a,
        end_node=b,
        start_x=na.x,
        start_y=na.y,
        end_x=nb.x,
        end_y=nb.y,
        length_m=length_m,
        diameter_m=diameter_m,
        roughness_c=roughness,
        has_valve=valve,
        initial_open=initial_open,
    )


def _build_easy(rng):
    nodes = {
        "R0": _node("R0", "reservoir", 0.0, "SRC", -1, 0, head=145.0),
        "N01": _node("N01", "junction", 101.0, "Z1", 0, 0, demand_lps=2.5 * rng.uniform(0.9, 1.1)),
        "N02": _node("N02", "junction", 102.0, "Z1", 1, 0, demand_lps=2.8 * rng.uniform(0.9, 1.1)),
        "N03": _node("N03", "junction", 103.0, "Z1", 2, 0, demand_lps=2.2 * rng.uniform(0.9, 1.1)),
        "N04": _node("N04", "junction", 104.0, "Z2", 3, 0, demand_lps=2.6 * rng.uniform(0.9, 1.1)),
        "N05": _node("N05", "junction", 105.0, "Z3", 4, 0, demand_lps=2.0 * rng.uniform(0.9, 1.1)),
        "N06": _node("N06", "junction", 103.5, "Z1", 1, 1, demand_lps=1.8 * rng.uniform(0.9, 1.1)),
        "N07": _node("N07", "junction", 104.5, "Z1", 2, 1, demand_lps=1.7 * rng.uniform(0.9, 1.1)),
        "N08": _node("N08", "junction", 104.8, "Z2", 3, -1, demand_lps=2.1 * rng.uniform(0.9, 1.1)),
        "N09": _node("N09", "junction", 106.0, "Z2", 4, -1, demand_lps=2.0 * rng.uniform(0.9, 1.1)),
        "N10": _node("N10", "junction", 106.2, "Z3", 5, 0, demand_lps=1.9 * rng.uniform(0.9, 1.1)),
    }
    pipes = {
        "P01": _pipe("P01", "R0", "N01", nodes, rng, 350, valve=False, initial_open=True),
        "P02": _pipe("P02", "N01", "N02", nodes, rng, 300, valve=False, initial_open=True),
        "P03": _pipe("P03", "N02", "N03", nodes, rng, 300, valve=True, initial_open=True),
        "P04": _pipe("P04", "N03", "N04", nodes, rng, 250, valve=False, initial_open=True),
        "P05": _pipe("P05", "N04", "N05", nodes, rng, 250, valve=True, initial_open=True),
        "P06": _pipe("P06", "N02", "N06", nodes, rng, 200, valve=False, initial_open=True),
        "P07": _pipe("P07", "N06", "N07", nodes, rng, 180, valve=False, initial_open=True),
        "P08": _pipe("P08", "N04", "N08", nodes, rng, 180, valve=False, initial_open=True),
        "P09": _pipe("P09", "N08", "N09", nodes, rng, 180, valve=False, initial_open=True),
        "P10": _pipe("P10", "N05", "N10", nodes, rng, 180, valve=False, initial_open=True),
        "P11": _pipe("P11", "N07", "N09", nodes, rng, 150, valve=True, initial_open=False),
        "P12": _pipe("P12", "N09", "N10", nodes, rng, 150, valve=True, initial_open=False),
    }
    zones = {
        "Z1": ["N01", "N02", "N03", "N06", "N07"],
        "Z2": ["N04", "N08", "N09"],
        "Z3": ["N05", "N10"],
    }
    return nodes, pipes, zones


def _build_medium(rng):
    nodes = {"R0": _node("R0", "reservoir", 0.0, "SRC", -1, 0, head=150.0)}
    for i in range(1, 7):
        nodes[f"T0{i}"] = _node(
            f"T0{i}",
            "junction",
            101.0 + 0.8 * i,
            "Z1",
            i - 1,
            0,
            demand_lps=(2.6 + 0.2 * i) * rng.uniform(0.9, 1.1),
        )
    upper_coords = [(2.5, 1), (3, 2), (4, 2), (5, 2), (6, 2), (7, 2)]
    lower_coords = [(2.5, -1), (3, -2), (4, -2), (5, -2), (6, -2), (7, -2)]
    for i, (x, y) in enumerate(upper_coords, start=1):
        nodes[f"U0{i}"] = _node(
            f"U0{i}",
            "junction",
            103.0 + 0.5 * i,
            "Z2",
            x,
            y,
            demand_lps=(2.0 + 0.2 * i) * rng.uniform(0.9, 1.1),
        )
    for i, (x, y) in enumerate(lower_coords, start=1):
        nodes[f"L0{i}"] = _node(
            f"L0{i}",
            "junction",
            103.0 + 0.5 * i,
            "Z3",
            x,
            y,
            demand_lps=(2.0 + 0.2 * i) * rng.uniform(0.9, 1.1),
        )
    nodes["A1"] = _node("A1", "junction", 105.0, "Z1", 3.5, 0.8, demand_lps=1.7 * rng.uniform(0.9, 1.1))
    nodes["A2"] = _node("A2", "junction", 105.6, "Z1", 4.2, 1.1, demand_lps=1.5 * rng.uniform(0.9, 1.1))
    nodes["B1"] = _node("B1", "junction", 106.0, "Z2", 5.8, 3.0, demand_lps=1.6 * rng.uniform(0.9, 1.1))
    nodes["B2"] = _node("B2", "junction", 106.4, "Z2", 6.6, 3.2, demand_lps=1.4 * rng.uniform(0.9, 1.1))
    nodes["C1"] = _node("C1", "junction", 106.0, "Z3", 5.8, -3.0, demand_lps=1.6 * rng.uniform(0.9, 1.1))
    nodes["C2"] = _node("C2", "junction", 106.4, "Z3", 6.6, -3.2, demand_lps=1.4 * rng.uniform(0.9, 1.1))

    spec = [
        ("P01", "R0", "T01", 350, False, True),
        ("P02", "T01", "T02", 320, False, True),
        ("P03", "T02", "T03", 300, True, True),
        ("P04", "T03", "T04", 280, False, True),
        ("P05", "T04", "T05", 280, True, True),
        ("P06", "T05", "T06", 250, False, True),
        ("P07", "T03", "U01", 250, False, True),
        ("P08", "U01", "U02", 220, False, True),
        ("P09", "U02", "U03", 220, False, True),
        ("P10", "U03", "U04", 220, True, True),
        ("P11", "U04", "U05", 220, False, True),
        ("P12", "U05", "U06", 200, False, True),
        ("P13", "T03", "L01", 250, False, True),
        ("P14", "L01", "L02", 220, False, True),
        ("P15", "L02", "L03", 220, False, True),
        ("P16", "L03", "L04", 220, True, True),
        ("P17", "L04", "L05", 220, False, True),
        ("P18", "L05", "L06", 200, False, True),
        ("P19", "U02", "L02", 200, True, True),
        ("P20", "T05", "A1", 180, False, True),
        ("P21", "A1", "A2", 160, False, True),
        ("P22", "U04", "B1", 180, False, True),
        ("P23", "B1", "B2", 160, False, True),
        ("P24", "L04", "C1", 180, False, True),
        ("P25", "C1", "C2", 160, False, True),
        ("P26", "U05", "T06", 150, True, False),
        ("P27", "L05", "T06", 150, True, False),
        ("P28", "A2", "U05", 150, True, False),
        ("P29", "C2", "L05", 150, True, False),
        ("P30", "B2", "T06", 150, True, False),
    ]
    pipes = {
        pid: _pipe(pid, a, b, nodes, rng, dia, valve=valve, initial_open=opn)
        for pid, a, b, dia, valve, opn in spec
    }
    zones = {
        "Z1": [f"T0{i}" for i in range(1, 7)] + ["A1", "A2"],
        "Z2": [f"U0{i}" for i in range(1, 7)] + ["B1", "B2"],
        "Z3": [f"L0{i}" for i in range(1, 7)] + ["C1", "C2"],
    }
    return nodes, pipes, zones


def _build_hard(rng):
    nodes = {
        "R0": _node("R0", "reservoir", 0.0, "SRC", -1, -2, head=155.0),
        "TK0": _node(
            "TK0",
            "tank",
            118.0,
            "SRC",
            6,
            0,
            tank={"init": 28.0, "min": 0.0, "max": 35.0, "diam": 20.0},
        ),
    }
    for r in range(5):
        for c in range(6):
            zid = "Z1" if c < 2 else "Z2" if c < 4 else "Z3"
            nodes[f"G{r}{c}"] = _node(
                f"G{r}{c}",
                "junction",
                100.0 + 0.8 * r + 0.6 * c,
                zid,
                c,
                -r,
                demand_lps=(1.2 + 0.2 * ((r + c) % 4)) * rng.uniform(0.9, 1.1),
            )

    pipes = {}
    pid_num = 1

    def add(a, b, dia, valve=False, open_=True):
        nonlocal pid_num
        pid = f"P{pid_num:02d}"
        pipes[pid] = _pipe(pid, a, b, nodes, rng, dia, valve=valve, initial_open=open_)
        pid_num += 1

    # source/tank
    add("R0", "G20", 350, False, True)
    add("TK0", "G05", 300, False, True)

    # horizontals
    valve_edges = {
        ("G02", "G03"),
        ("G13", "G14"),
        ("G20", "G21"),
        ("G23", "G24"),
        ("G11", "G12"),
        ("G22", "G23"),
        ("G32", "G33"),
        ("G33", "G34"),
        ("G41", "G42"),
        ("G43", "G44"),
    }
    for r in range(5):
        for c in range(5):
            a, b = f"G{r}{c}", f"G{r}{c + 1}"
            add(a, b, 200 if c < 2 else 180, valve=((a, b) in valve_edges), open_=True)

    # verticals
    removed_verticals = {
        ("G00", "G10"),
        ("G01", "G11"),
        ("G04", "G14"),
        ("G12", "G22"),
        ("G20", "G30"),
        ("G35", "G45"),
        ("G24", "G34"),
    }
    valve_verticals = {
        ("G03", "G13"),
        ("G05", "G15"),
        ("G11", "G21"),
        ("G13", "G23"),
        ("G22", "G32"),
        ("G25", "G35"),
        ("G24", "G34"),
        ("G30", "G40"),
        ("G31", "G41"),
        ("G32", "G42"),
    }
    for r in range(4):
        for c in range(6):
            a, b = f"G{r}{c}", f"G{r + 1}{c}"
            if (a, b) in removed_verticals:
                continue
            add(a, b, 180, valve=((a, b) in valve_verticals), open_=True)

    zones = {
        "Z1": [f"G{r}{c}" for r in range(5) for c in range(2)],
        "Z2": [f"G{r}{c}" for r in range(5) for c in range(2, 4)],
        "Z3": [f"G{r}{c}" for r in range(5) for c in range(4, 6)],
    }
    return nodes, pipes, zones


def _sample_leak_pipe(rng, pipes):
    eligible = [
        pid
        for pid, p in pipes.items()
        if (not p.has_valve)
        and p.initial_open
        and not p.start_node.startswith("R")
        and not p.end_node.startswith("R")
        and not p.start_node.startswith("TK")
        and not p.end_node.startswith("TK")
    ]
    return str(rng.choice(sorted(eligible)))


def _sample_medium_confounder(rng, zones, leak_zone):
    other = sorted(zid for zid in zones if zid not in {"SRC", leak_zone})
    zone_id = str(rng.choice(other))
    return ConfounderDef(
        "demand_spike", zone_id=zone_id, multiplier=float(rng.uniform(1.20, 1.45))
    )


def _sample_hard_confounder(rng, nodes, pipes, leak_nodes):
    excluded = set(leak_nodes)
    for pipe in pipes.values():
        if pipe.start_node in leak_nodes:
            excluded.add(pipe.end_node)
        if pipe.end_node in leak_nodes:
            excluded.add(pipe.start_node)
    candidates = sorted(
        nid for nid in nodes if nid.startswith("G") and nid not in excluded
    )
    node_id = str(rng.choice(sorted(candidates)))
    return ConfounderDef(
        "sensor_bias", node_id=node_id, bias_psi=float(rng.uniform(3.0, 5.0))
    )


def compute_sections(
    nodes: dict[str, NodeDef], pipes: dict[str, PipeDef]
) -> list[SectionDef]:
    adj: dict[str, list[tuple[str, str]]] = {}
    for pid, pipe in pipes.items():
        if pipe.initial_open and not pipe.has_valve:
            adj.setdefault(pipe.start_node, []).append((pipe.end_node, pid))
            adj.setdefault(pipe.end_node, []).append((pipe.start_node, pid))

    seen: set[str] = set()
    sections: list[SectionDef] = []
    for node_id in adj:
        if node_id in seen:
            continue
        stack = [node_id]
        seen.add(node_id)
        comp_nodes: set[str] = set()
        comp_pipes: set[str] = set()
        while stack:
            u = stack.pop()
            comp_nodes.add(u)
            for v, pid in adj.get(u, []):
                comp_pipes.add(pid)
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        boundary = []
        for pid, pipe in pipes.items():
            if not pipe.has_valve:
                continue
            if pipe.start_node in comp_nodes or pipe.end_node in comp_nodes:
                boundary.append(pid)
        sections.append(
            SectionDef(
                section_id=f"S{len(sections) + 1}",
                pipe_ids=sorted(comp_pipes),
                boundary_valve_pipe_ids=sorted(set(boundary)),
            )
        )

    total = sum(n.base_demand_m3s for n in nodes.values() if n.kind == "junction")
    demand_by_sec: dict[str, float] = {s.section_id: 0.0 for s in sections}
    node_to_sec: dict[str, set[str]] = {}
    for s in sections:
        for pid in s.pipe_ids:
            p = pipes[pid]
            node_to_sec.setdefault(p.start_node, set()).add(s.section_id)
            node_to_sec.setdefault(p.end_node, set()).add(s.section_id)
    for nid, sec_ids in node_to_sec.items():
        if nodes[nid].kind != "junction":
            continue
        share = nodes[nid].base_demand_m3s / max(len(sec_ids), 1)
        for sid in sec_ids:
            demand_by_sec[sid] += share
    for s in sections:
        s.demand_fraction = demand_by_sec[s.section_id] / max(total, 1e-9)
    return sections
