from __future__ import annotations

from typing import Any

import requests

try:
    from openenv.core.client_types import StepResult
except ImportError:

    class StepResult:
        def __init__(self, observation=None, reward=None, done=False):
            self.observation = observation
            self.reward = reward
            self.done = done

try:
    from openenv.core.client import EnvClient
except Exception:

    class EnvClient:
        def __init__(self, base_url: str, timeout: float = 30.0) -> None:
            self.base_url = base_url.rstrip("/")
            self.timeout = timeout


try:
    from .models import (
        LeakHunterAction,
        LeakHunterObservation,
        LeakHunterState,
        ResetObservation,
    )
except ImportError:
    from models import (
        LeakHunterAction,
        LeakHunterObservation,
        LeakHunterState,
        ResetObservation,
    )


class LeakHunterEnv(EnvClient):
    """Thin typed HTTP client for LeakHunter."""

    def __init__(
        self, base_url: str = "http://localhost:8000", timeout: float = 30.0
    ) -> None:
        super().__init__(base_url=base_url, timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def reset(
        self, difficulty: str = "easy", seed: int | None = None
    ) -> ResetObservation:
        payload: dict[str, Any] = {"difficulty": difficulty}
        if seed is not None:
            payload["seed"] = seed
        resp = self._session.post(
            f"{self.base_url}/reset", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        return ResetObservation.model_validate(data["observation"])

    def step(self, action: LeakHunterAction) -> StepResult:
        payload = {"action": action.model_dump()}
        resp = self._session.post(
            f"{self.base_url}/step", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()
        obs = LeakHunterObservation.model_validate(data["observation"])
        return StepResult(
            observation=obs,
            reward=data.get("reward"),
            done=bool(data.get("done", False)),
        )

    def state(self) -> LeakHunterState:
        resp = self._session.get(
            f"{self.base_url}/state", timeout=self.timeout
        )
        resp.raise_for_status()
        return LeakHunterState.model_validate(resp.json())

    def close(self) -> None:
        self._session.close()
