# Reward the validated answer, not parser repair

The terminal-policy control assigns `correct_format=True` at the end of
`SC2Planner.generate_plan()`, even after an invalid answer is truncated, filtered
or completed by rules. `game_end()` then adds 0.5 to its terminal reward. This
candidate changes only the source of that verdict, not the amount or timing.

`StrategicSC2EnvironmentManagerV2.step()` already validates the complete answer
against the current BO candidates and response limits. It now passes that
existing verdict with the unchanged action string in a small envelope:
`{"text": answer, "plan_format_valid": bool}`. The verdict also requires an
evaluable BO context, successful projection and no final-stage tool violation.
SC2Env forwards it as socket metadata. Only when Planner consumes the ACTION
does it update its historical `correct_format` field. Parsing and rule repair
cannot overwrite that field. An ACTION ignored during terminal acknowledgement
does not change it. Direct text callers without a verdict can still execute,
but cannot earn a format bonus; deploy these three files together.

Despite the historical name, this is the existing **full plan-contract** verdict:
syntax, current candidate membership, counts and response limits. It does not
prove that a unit was built, that the plan was strategically good, or that every
bullet was invalid. No new validator or attack/production rule is introduced.
The original `-0.1` invalid-action training treatment is unchanged. Nonterminal
state shaping, attack bonuses, terminal win/loss reward and loss-state shaping
are also unchanged. Only a terminal bonus incorrectly attached to an invalid
last consumed answer is removed. Earlier invalid answers are not retrospectively
penalized by this change.

This follows the response-based scoring interface documented by upstream
[verl reward functions](https://github.com/verl-project/verl/blob/main/docs/preparation/reward_function.rst):
score the generated response and award format credit conditionally. That public
example does not prescribe our SC2 protocol, the 0.5 coefficient or the socket
metadata; those are existing project choices and a small integration repair.

Regression tests execute real validator, Planner and transport methods with
fake process boundaries. They cover invalid names, invalid suffixes beyond a
valid prefix, excess bullets, successive valid/invalid replies, missing verdicts
and unchanged rule-repaired BO. Tests do not establish a win-rate improvement.
The approved live gate is described in
[the experiment record](../../experiments/records/20260905-r6-format-reward-v1/README.md).
