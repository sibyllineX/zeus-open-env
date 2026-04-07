from __future__ import annotations

import pytest

from models import LeakHunterAction
from server.environment import LeakHunterEnvironment
from server.grader import classify_score, grade_state


def resolve_scripted_actions(env, actions):
    assert env._ctx is not None
    leak_pipe_id = env._ctx.network.leak.original_pipe_id
    leak_section_id = next(
        (
            section.section_id
            for section in env._ctx.network.sections
            if leak_pipe_id in section.pipe_ids
        ),
        env._ctx.network.sections[0].section_id if env._ctx.network.sections else "S1",
    )
    resolved_actions = []
    for action in actions:
        if action.action_type != "repair":
            resolved_actions.append(action)
            continue
        target_id = (
            leak_section_id if action.method == "isolate_section" else leak_pipe_id
        )
        resolved_actions.append(
            LeakHunterAction(
                action_type="repair",
                target_id=target_id,
                method=action.method,
            )
        )
    return resolved_actions


def run_actions(env, difficulty, seed, actions):
    env.reset(difficulty=difficulty, seed=seed)
    resolved_actions = resolve_scripted_actions(env, actions)
    obs = None
    for action in resolved_actions:
        obs = env.step(action)
        if obs.done:
            break
    return obs, env.state, resolved_actions


TRAJECTORIES = [
    pytest.param(
        "easy",
        11,
        [
            LeakHunterAction(action_type="install_sensor", target_id="N09"),
            LeakHunterAction(action_type="read_pressure", target_id="N08"),
            LeakHunterAction(action_type="read_flow", target_id="P09"),
            LeakHunterAction(action_type="close_valve", target_id="P05"),
            LeakHunterAction(action_type="repair", target_id="P08", method="clamp_pipe"),
        ],
        id="easy_expert",
    ),
    pytest.param(
        "easy",
        12,
        [
            LeakHunterAction(action_type="read_pressure", target_id="N07"),
            LeakHunterAction(action_type="read_pressure", target_id="N09"),
            LeakHunterAction(action_type="repair", target_id="S2", method="isolate_section"),
        ],
        id="easy_decent",
    ),
    pytest.param(
        "easy",
        13,
        [
            LeakHunterAction(action_type="read_pressure", target_id="N01"),
            LeakHunterAction(action_type="repair", target_id="P02", method="clamp_pipe"),
        ],
        id="easy_bad",
    ),
    pytest.param(
        "medium",
        21,
        [
            LeakHunterAction(action_type="install_sensor", target_id="U04"),
            LeakHunterAction(action_type="install_sensor", target_id="L04"),
            LeakHunterAction(action_type="read_pressure", target_id="U05"),
            LeakHunterAction(action_type="read_pressure", target_id="L05"),
            LeakHunterAction(action_type="close_valve", target_id="P10"),
            LeakHunterAction(action_type="repair", target_id="P11", method="replace_section"),
        ],
        id="medium_good",
    ),
    pytest.param(
        "medium",
        22,
        [
            LeakHunterAction(action_type="read_flow", target_id="P19"),
            LeakHunterAction(action_type="repair", target_id="S3", method="isolate_section"),
        ],
        id="medium_decent",
    ),
    pytest.param(
        "medium",
        23,
        [
            LeakHunterAction(action_type="read_pressure", target_id="T01"),
            LeakHunterAction(action_type="repair", target_id="P02", method="clamp_pipe"),
        ],
        id="medium_bad",
    ),
    pytest.param(
        "hard",
        31,
        [
            LeakHunterAction(action_type="install_sensor", target_id="G24"),
            LeakHunterAction(action_type="install_sensor", target_id="G34"),
            LeakHunterAction(action_type="read_pressure", target_id="G23"),
            LeakHunterAction(action_type="read_flow", target_id="P18"),
            LeakHunterAction(action_type="close_valve", target_id="P09"),
            LeakHunterAction(action_type="repair", target_id="S1", method="isolate_section"),
        ],
        id="hard_decent",
    ),
    pytest.param(
        "hard",
        32,
        [
            LeakHunterAction(action_type="read_pressure", target_id="G00"),
            LeakHunterAction(action_type="read_pressure", target_id="G45"),
            LeakHunterAction(action_type="repair", target_id="P03", method="replace_section"),
        ],
        id="hard_bad",
    ),
]


@pytest.mark.parametrize("difficulty,seed,actions", TRAJECTORIES)
def test_scripted_trajectories(difficulty, seed, actions):
    env = LeakHunterEnvironment()
    obs, state, resolved_actions = run_actions(env, difficulty, seed, actions)
    score = grade_state(state)
    band = classify_score(difficulty, score)
    repair_action = resolved_actions[-1]

    assert obs is not None
    assert obs.done
    assert state.done
    assert state.final_score is not None
    assert 0.0 <= score <= 1.0
    assert band in {"expert", "good"}, f"Expected expert/good, got {band} (score={score:.3f})"
    assert state.budget_remaining >= 0
    assert state.budget_remaining <= state.budget_total
    assert state.forced_repair is False
    assert state.last_message is not None
    assert "Final score" in state.last_message
    assert repair_action.target_id in state.last_message
