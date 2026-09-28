# Per-call state reward

The validated natural-termination runtime (`8ac1e19`) still adds
`min(0.5, round(0.25 * state_score, 3))` on every nonterminal Planner call.
Consequently, twenty calls with the same score of 2 produce +10 reward even
without progress. The score is an existing heuristic, not a learned value.

`SC2_STEP_SHAPING_MODE=legacy` retains that formula. The optional `off` mode
skips only this addition in `SC2Planner.get_action()` and logs
`Step shaping reward off: 0.000`. Incoming rewards, model answers and the
entire `game_end()` method are unchanged. Victory/defeat rewards, terminal
state shaping, format rewards, invalid-action penalties, termination, Planner
query, parser, Executor, micro and GiGPO/PPO remain frozen. This is not a claim
that every nonterminal reward or every unproductive loop has been removed.

The launcher rejects unknown modes before creating a run, forwards the chosen
mode to every Ray worker and records it in the effective configuration.
The original A/B kept `legacy` as its default. Integrated Fixes v1 now defaults
to `off`; `legacy` requires an explicit override. Historical frozen runs are
unchanged. The implementation is the same one already tested in that A/B.

The public basis is [GiGPO (2025), Appendix D.1](https://arxiv.org/html/2505.10978v2#A4.SS1):
its ALFWorld/WebShop experiments use task outcomes and invalid-action penalties
with gamma 0.95, without requiring an SC2-style per-call state bonus. This
supports the feasibility of the ablation, not a guarantee that it improves SC2.
We do not copy their numeric rewards into our environment.

The earlier local delta candidate is not imported. The current collector sums
episode rewards without discount, while `compute_step_discounted_returns()`
uses gamma 0.95 for the step component. Therefore a discounted potential-shaping
formula alone would not justify a policy-invariance claim for the combined
GiGPO objective. This A/B avoids changing either return definition or inventing
a replacement reward formula. It also removes potentially useful dense feedback;
the net effect must be measured.
