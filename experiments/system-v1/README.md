# Historical System-v1 reference results

These measurements come from completed historical logs, not from running the
new integration package. The 2026-08-31 audit checked 96 metric rows, zero
process exit, and complete checkpoint markers/model shards for each listed run.
The exact Qwen release record remains at
[Qwen r6](../qwen3-4b-level5-r6/manifest.json).

| Run | Wins / 768 | First → last 16 success | First → last 16 reward |
|---|---:|---:|---:|
| Qwen canonical r6, seed0 | 625 (81.38%) | 73.44% → 89.84% | 6.22 → 14.83 |
| SpB canonical r7, seed0 | 652 (84.90%) | 82.03% → 85.94% | 9.39 → 11.61 |
| SpB Rush r4, seed1 | 667 (86.85%) | 84.38% → 86.72% | 13.87 → 15.16 |
| SpB Rush r8, seed2 | 674 (87.76%) | 85.16% → 80.47% | 14.82 → 11.20 |

The same-name trainer valid-action mean was 29.74% for Qwen r6 and 23.02%
for SpB r7; the response-budget-hit mean was 59.76% and 94.59%, respectively.
Qwen's pre-clip gradient mean/max was 350.95/2851.23, all finite. These known
limitations remain in the frozen baseline.

SpB Rush seed2 had 94.22% trainer-valid actions and 2.42% budget hits, while
seed1 had 4.70% and 93.69%. Seed1's final 16 steps had zero trainer-valid
ratio. Planner, validator and BO hashes match across those two historical
runs. This is not sufficient to establish stable learning or attribute the
difference solely to the environment seed.

All percentages above are means of per-update trainer metrics except the
explicit integer win counts. The valid-action metric is not the strict
whole-plan pass rate. These are on-policy training rollouts, not held-out
policy evaluation, and different BO/seed arms are not controlled architecture
comparisons. The small Level 6 Rush smoke (7/8) is recorded separately in the
old experiment branch; it does not establish 96-step Level 6 reliability.

The machine-readable identifiers and artifact locations are in
[references.json](references.json). The raw logs, original configs, checkpoint
and frozen runtime archives remain at those locations.
