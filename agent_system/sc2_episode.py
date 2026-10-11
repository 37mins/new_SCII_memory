"""Gymnasium termination fields for SC2 environment transitions."""

from itertools import count


def interaction_steps(max_steps):
    """Iterate until SC2 ends when max_steps is zero; positive values are explicit caps."""
    return range(max_steps) if max_steps > 0 else count()


def interaction_limit_reached(max_steps, decision_count):
    return max_steps > 0 and decision_count >= max_steps


def outcome_fields(result):
    if result not in ("Victory", "Defeat", "Tie"):
        raise ValueError(f"Unknown SC2 game result: {result!r}")
    return {
        "terminated": True,
        "truncated": False,
        "game_result": result,
        "termination_reason": "game_result",
        # Preserve the frozen r6 reward rule; this change only fixes end semantics.
        "outcome_reward": {"Victory": 20.0, "Defeat": -50.0, "Tie": -50.0}[result],
    }


def truncation_fields(reason="interaction_limit"):
    return {
        "terminated": False,
        "truncated": True,
        "game_result": None,
        "termination_reason": reason,
        "outcome_reward": 0.0,
    }


def step_info(data, action_valid):
    info = {
        "action_is_effective": action_valid,
        "action_is_valid": action_valid,
        "won": data["success"],
        "attack_ct": data["attack_ct"],
    }
    info.update(
        {
            key: data.get(key, default)
            for key, default in (
                ("terminated", False),
                ("truncated", False),
                ("game_result", None),
                ("termination_reason", None),
                ("outcome_reward", 0.0),
            )
        }
    )
    if "policy_action_applied" in data:
        info.update({key: data[key] for key in (
            "policy_action_applied", "policy_action_id", "last_consumed_policy_action_id",
        )})
    if "telemetry" in data:
        info["telemetry"] = data["telemetry"]
    return info


def gymnasium_transition(data, action_valid):
    info = step_info(data, action_valid)
    terminated = bool(info["terminated"])
    truncated = bool(info["truncated"])
    if bool(data["done"]) != (terminated or truncated):
        raise RuntimeError("SC2 transition ended without terminated/truncated attribution")
    return data["obs"], data["reward"], terminated, truncated, info


def apply_external_truncation(terminated, truncated, info, limit_reached):
    """Apply the RL wrapper's interaction limit without replacing a real result."""
    if terminated or truncated or not limit_reached:
        return terminated, truncated, info
    return False, True, {**info, **truncation_fields()}


def policy_transitions(trajectory):
    """Exclude terminal polls, preserving outcomes on the last consumed action."""
    selected = []
    for row in trajectory:
        if not row["active_masks"]:
            continue
        # Environments without the SC2 consumption handshake keep their contract.
        if row.get("policy_action_applied", True):
            selected.append(dict(row))
        else:
            if not row["environment_done"]:
                raise RuntimeError("Unconsumed nonterminal policy response")
            if selected:
                selected[-1]["rewards"] += row["rewards"]
    return selected
