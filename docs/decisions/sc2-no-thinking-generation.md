# SC2 no-thinking generation v1

2026-09-20 user authorization: leave the active C197 experiment untouched;
Level7 and all following queued levels must use explicit no-thinking generation.

The original SpB template ignores `enable_thinking=False`. This opt-in protocol
adds `<think>\n\n</think>\n` after its verified assistant header, in the INPUT only.
HF generation blocks ASCII `<think` and `</think` prefixes, case-insensitively,
including alternative multi-token spellings and boundaries adjacent to tokens
omitted by `skip_special_tokens=True`. Do not just remove tags after generation.
This controls explicit thinking-format output, not unverifiable internal cognition.

Required overrides, passed together in short training, resumed training and both
evaluations:

```
+data.sc2_no_thinking=True
+actor_rollout_ref.rollout.no_thinking_tokenizer_path=/assets/spb
+actor_rollout_ref.actor.no_thinking_tokenizer_path=/assets/spb
+actor_rollout_ref.ref.no_thinking_tokenizer_path=/assets/spb
```

Default is off; existing C's frozen source and all runs are unchanged. Tokenizer
loading is offline and read-only. Only the verified Qwen2 fast ByteLevel tokenizer
with SpB tag mappings151667/151668 is accepted; unsupported modes fail explicitly.
Actor/ref require padded, unfused, non-sequence-parallel logits. Both apply the
identical prefix-dependent support mask before normalized log probability and
entropy, including current/old actor probabilities and reference KL. Finite-min
masked logits avoid NaNs in the existing entropy function. The existing
temperature/top-p protocol is unchanged; this does not resolve or claim exact
equivalence of top-p sampled and untruncated teacher-forced distributions.

Each worker prints `SC2_NO_THINKING` with the support-mask hash. Generated response
tokens are not rewritten, retried or salvaged. The prior invalid-action boundary
remains a fallback safety check. No added reward, optimizer change, seed change,
extra game budget, or additional initial evaluation. Empty-thinking input prefix
and decoding constraints are an explicit protocol difference from C Level6, so
future Level7–10 must not be labeled as changing difficulty alone.

Verification must cover default-off behavior, real tokenizer rendering/decoding,
single/split/case-varied tags, prompt/response separation, batch/padding behavior,
HF generation with synthetic CPU logits strongly favoring forbidden output,
actor/reference teacher-forced probabilities, finite entropy/gradients and the
four real stage configurations. CPU checks do not establish SC2/GPU success.
