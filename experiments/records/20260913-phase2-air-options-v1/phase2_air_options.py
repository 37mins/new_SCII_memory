"""Opt-in stage-two option exposure. No scheduling, parsing or execution changes."""
import copy
import functools
import os


INSTRUCTION = (
    "Choose Marine reinforcements or the listed air-tech options according to "
    "the current observation. Factory, Starport and Vikings are optional; "
    "use only currently available actions and respect their remaining quotas."
)


def extend_phase_two(strategist, obs, phase):
    if phase["_phase_index"] != 1:
        return phase
    result = copy.deepcopy(phase)
    terminal = strategist.strategy[-1]["bo todo"]
    # Reuse the existing late-stage macro reconciliation, without adding to the
    # persistent template and accidentally doubling its cumulative Starport target.
    options = strategist._reconcile_completed_macro(
        len(strategist.strategy) - 1, obs,
        {name: terminal[name] for name in ("starport", "vikingfighter")},
    )
    result["bo todo"].update(options)
    result["instruction"] = INSTRUCTION
    return result


def install():
    """Explicit experimental hook only; no default entry imports this module."""
    if (os.environ.get("SC2_STRATEGY_PROFILE") != "marine_pressure"
            or os.environ.get("SC2_ENABLE_AIR_OPTIONS") != "1"):
        raise ValueError("Phase-two candidate requires frozen marine_pressure + air options")
    # envSC2 installs the Arena import path; patch the class Commander actually
    # uses, not a second instance imported under a fully qualified alias.
    from agents.commander import SC2Strategist
    original = SC2Strategist.generate_strategy

    @functools.wraps(original)
    def generate(strategist, obs):
        return extend_phase_two(strategist, obs, original(strategist, obs))

    SC2Strategist.generate_strategy = generate
