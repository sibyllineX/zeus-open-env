from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, model_validator

try:
    from openenv.core.env_server.types import Action, Observation, State
    from openenv.core.client_types import StepResult
except ImportError:  # fallback for older installs
    from pydantic import BaseModel as Action
    from pydantic import BaseModel as Observation
    from pydantic import BaseModel as State

    from typing import Generic, TypeVar

    T = TypeVar("T")

    class StepResult(BaseModel, Generic[T]):
        observation: Any = None
        reward: float | None = None
        done: bool = False


Difficulty = Literal["easy", "medium", "hard"]
ActionType = Literal[
    "read_pressure",
    "read_flow",
    "install_sensor",
    "close_valve",
    "open_valve",
    "repair",
]
RepairMethod = Literal["clamp_pipe", "replace_section", "isolate_section"]
ReadingType = Literal["pressure", "flow"]
ReadingSource = Literal["query", "auto"]
ObservationType = Literal["reset", "step", "error", "terminal"]


class NodeInfo(BaseModel):
    """Public description of an exposed network node."""

    node_id: str
    kind: Literal["junction", "reservoir", "tank"]
    elevation_m: float
    zone_id: str
    coordinates_xy: tuple[float, float]
    base_demand_m3s: float = 0.0
    baseline_pressure_psi: float | None = None
    baseline_pressure_range_psi: tuple[float, float] | None = None


class PipeInfo(BaseModel):
    """Public description of an original pipe, never a split artifact."""

    pipe_id: str
    start_node: str
    end_node: str
    length_m: float
    diameter_mm: float
    roughness_c: float
    has_valve: bool
    is_open: bool
    baseline_flow_m3s: float | None = None
    baseline_direction: Literal["start_to_end", "end_to_start", "closed", "zero"] = "zero"


class SectionInfo(BaseModel):
    """Reset-time section grouping used for isolate_section repair."""

    section_id: str
    pipe_ids: list[str]
    boundary_valve_pipe_ids: list[str]
    demand_fraction: float


class ZoneInfo(BaseModel):
    """Loose geographic/service zone grouping used in reset observation."""

    zone_id: str
    node_ids: list[str]
    nominal_demand_m3s: float


class SensorReading(BaseModel):
    """One pressure or flow reading returned either by query or auto-report."""

    reading_type: ReadingType
    target_id: str
    source: ReadingSource
    value: float | None = None
    units: str
    delta_percent_from_baseline: float | None = None
    baseline_value: float | None = None
    status: Literal["ok", "no_service", "solver_failed"] = "ok"
    text: str


class RewardBreakdown(BaseModel):
    """Fully expanded reward terms for logging/tests/UI."""

    dense_component: float = 0.0
    terminal_component: float = 0.0
    repair_component: float = 0.0
    localization_component: float = 0.0
    residual_leak_component: float = 0.0
    service_component: float = 0.0
    water_component: float = 0.0
    final_score: float | None = None


class ActionRecord(BaseModel):
    """Compact action log entry exposed via state()."""

    step_index: int
    action_text: str
    reward: float
    budget_remaining: int
    elapsed_minutes: int
    hydraulic_revision: int
    message: str


class LeakHunterAction(Action):
    """
    Single action model for all 6 actions.

    target_id conventions:
    - Nxx for read_pressure / install_sensor
    - Pxx for read_flow / close_valve / open_valve
    - Pxx or Sx for repair depending on method
    """

    action_type: ActionType
    target_id: str = Field(min_length=1)
    method: RepairMethod | None = None

    @model_validator(mode="after")
    def validate_by_type(self) -> "LeakHunterAction":
        if self.action_type in {"read_pressure", "install_sensor"}:
            if self.target_id.startswith("P"):
                raise ValueError("Node actions require a node target, not a pipe (Pxx)")
            if self.method is not None:
                raise ValueError("method must be omitted for non-repair actions")
        elif self.action_type in {"read_flow", "close_valve", "open_valve"}:
            if not self.target_id.startswith("P"):
                raise ValueError("Pipe actions require target_id like P07")
            if self.method is not None:
                raise ValueError("method must be omitted for non-repair actions")
        elif self.action_type == "repair":
            if self.method is None:
                raise ValueError("repair action requires a method")
            if self.method in {"clamp_pipe", "replace_section"} and not self.target_id.startswith("P"):
                raise ValueError("clamp_pipe/replace_section require a pipe target like P07")
            if self.method == "isolate_section" and not self.target_id.startswith("S"):
                raise ValueError("isolate_section requires a section target like S2")
        return self

    def as_command(self) -> str:
        if self.action_type == "repair":
            return f"repair {self.target_id} {self.method}"
        return f"{self.action_type} {self.target_id}"


class LeakHunterObservation(Observation):
    """Per-step observation returned by the server."""

    observation_type: ObservationType = "step"
    text: str = ""
    difficulty: Difficulty = "easy"
    budget_total: int = 0
    budget_remaining: int = 0
    elapsed_minutes: int = 0
    hydraulic_revision: int = 0
    last_action: str | None = None
    sensor_readings: list[SensorReading] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)
    summary_text: str | None = None
    reward_breakdown: RewardBreakdown | None = None
    done: bool = False
    reward: float | None = None


class ResetObservation(LeakHunterObservation):
    """Initial observation containing topology, sections, and baseline info."""

    observation_type: Literal["reset"] = "reset"
    seed: int = 0
    network_name: str = ""
    nodes: list[NodeInfo] = Field(default_factory=list)
    pipes: list[PipeInfo] = Field(default_factory=list)
    sections: list[SectionInfo] = Field(default_factory=list)
    zones: list[ZoneInfo] = Field(default_factory=list)
    token_estimate: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class LeakHunterState(State):
    """Internal episode metadata exposed by GET /state."""

    episode_id: str | None = None
    step_count: int = 0
    difficulty: Difficulty | None = None
    seed: int | None = None
    budget_total: int = 0
    budget_remaining: int = 0
    elapsed_minutes: int = 0
    hydraulic_revision: int = 0
    installed_sensors: list[str] = Field(default_factory=list)
    valve_states: dict[str, bool] = Field(default_factory=dict)  # True=open
    cumulative_dense_reward: float = 0.0
    return_so_far: float = 0.0
    final_score: float | None = None
    forced_repair: bool = False
    done: bool = False
    last_action: str | None = None
    last_message: str | None = None
    action_log: list[ActionRecord] = Field(default_factory=list)


AnyObservation = Union[LeakHunterObservation, ResetObservation]
LeakHunterStepResult = StepResult
