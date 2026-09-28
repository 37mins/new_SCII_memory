# C thinking-tag boundary repair — 2026-09-20

C197 runtime33abbc3 stopped during sample batch64 when one response contained a
literal `<think>` or `</think>` tag. User authorized repair and continuation.
This revision is based directly on33abbc3, not the current development HEAD.

When existing `env.forbid_thinking_output` is true, reject the **whole** offending
action as blank/invalid at the environment boundary. Preserve the raw response in
the prompt/response audit and upstream rollout tokens/log probabilities; do not
strip tags and accept a suffix, resample, invent a policy sample or add a penalty.
Existing invalid-answer BO repair remains in effect: this is not a no-op and its
rule-generated actions must not be attributed to a valid model answer. No format
bonus is awarded. Other slots and valid replies are unchanged. Natural-terminal,
truncation, empty-policy, finite-metric and storage checks remain intact.

This changes malformed-output failure handling and is a protocol revision, not
evidence of better learning. Plot the resumed branch separately afterstep56 and
retain failed57–63. Reward coefficients, prompts, seeds, temperature, optimizer,
precision and Executor rules are unchanged. CPU regression is not GPU acceptance.
