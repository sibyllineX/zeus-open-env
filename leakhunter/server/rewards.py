from __future__ import annotations

from math import floor

import networkx as nx
import numpy as np

try:
    from .networks import EpisodeNetwork
except ImportError:
    from server.networks import EpisodeNetwork


def compute_utility(result) -> float:
    return 1.0 - 0.65 * result.leak_frac - 0.35 * result.disrupted_demand_frac


def compute_dense_reward(
    prev_result, curr_result, action_cost: int, budget_total: int
) -> float:
    delta_u = compute_utility(curr_result) - compute_utility(prev_result)
    return float(
        np.clip(0.25 * delta_u, -0.05, 0.05) - 0.01 * (action_cost / budget_total)
    )


def build_line_graph(network: EpisodeNetwork) -> nx.Graph:
    g = nx.Graph()
    for pipe_id in network.pipes:
        g.add_node(pipe_id)
    pipe_items = list(network.pipes.items())
    for i, (pid_a, pa) in enumerate(pipe_items):
        for pid_b, pb in pipe_items[i + 1 :]:
            if (
                len(
                    {pa.start_node, pa.end_node}.intersection(
                        {pb.start_node, pb.end_node}
                    )
                )
                > 0
            ):
                g.add_edge(pid_a, pid_b)
    return g


def graph_diameter_or_default(g: nx.Graph) -> int:
    if g.number_of_nodes() == 0:
        return 3
    if nx.is_connected(g):
        return nx.diameter(g)
    return max(nx.diameter(g.subgraph(c)) for c in nx.connected_components(g))


def pipe_distance(g: nx.Graph, a: str, b: str) -> int:
    try:
        return int(nx.shortest_path_length(g, a, b))
    except Exception:
        return 999


def compute_repair_success_component(
    network: EpisodeNetwork,
    line_graph: nx.Graph,
    target_id: str,
    method: str,
) -> tuple[float, int]:
    leak_pipe = network.leak.original_pipe_id

    if method == "clamp_pipe":
        d = pipe_distance(line_graph, target_id, leak_pipe)
        if d == 0:
            return 0.45, d
        if d == 1:
            return 0.45 * 0.5, d
        return 0.0, d

    if method == "replace_section":
        d = pipe_distance(line_graph, target_id, leak_pipe)
        if d <= 2:
            return 0.35 * (1.0 - d / 3.0), d
        return 0.0, d

    if method == "isolate_section":
        section = network.section_map[target_id]
        ds = [pipe_distance(line_graph, p, leak_pipe) for p in section.pipe_ids]
        min_d = min(ds) if ds else 999
        if leak_pipe in section.pipe_ids:
            return 0.25, 2 + min_d
        return 0.0, 2 + min_d

    return 0.0, 999


def compute_localization_component(
    network: EpisodeNetwork,
    line_graph: nx.Graph,
    target_id: str,
    method: str,
    distance: int,
) -> float:
    diameter = graph_diameter_or_default(line_graph)
    k = max(3, floor(diameter / 2))
    return 0.20 * max(0.0, 1.0 - distance / max(k, 1))


def residual_leak_component(
    initial_leak_flow_m3s: float, post_repair_leak_flow_m3s: float
) -> float:
    if initial_leak_flow_m3s <= 1e-12:
        return 0.10
    stopped_frac = max(
        0.0, 1.0 - post_repair_leak_flow_m3s / initial_leak_flow_m3s
    )
    return 0.10 * stopped_frac


def average_service_score(
    service_integral_min: float,
    elapsed_minutes: int,
    current_service_fraction: float,
) -> float:
    if elapsed_minutes <= 0:
        return 0.10 * current_service_fraction
    return 0.10 * max(0.0, min(1.0, service_integral_min / elapsed_minutes))


def compute_water_conservation_component(
    cumulative_loss_m3: float, loss_cap_m3: float
) -> float:
    if loss_cap_m3 <= 1e-12:
        return 0.10
    return 0.10 * max(0.0, 1.0 - cumulative_loss_m3 / loss_cap_m3)
