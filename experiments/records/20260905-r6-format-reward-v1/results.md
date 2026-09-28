# Format reward: r1 completion

Frozen source `46ce2ccde96b153e008731ca78d95168699dd0d4`, September 5,
13:45:01–14:14:35 UTC. Exit 0, 1/1 PPO update, finite metrics and all four-rank
checkpoint shards complete. Artifact hashes and raw metrics are in
[r1-completion.json](r1-completion.json).

Eight natural games produced six wins and two defeats (75%; Wilson 95% interval
40.9–92.9%). There were 56 consumed policy replies, 18 fully valid, and three
excluded terminal replies. Consumption IDs and environment reward conservation
passed the same r2 gate, with the expected source explicitly changed to r1.
Five games consumed policy decisions; three games (two wins, one defeat) did not.

All five last consumed answers were invalid; the three other games had no
consumed answer. Console evidence contains eight terminal format bonuses, all
zero. This verifies removal of false terminal format credit in this batch.
The valid-last-answer +0.5 branch was not exercised live; existing CPU tests
cover it. It would be incorrect to call all 56 answers invalid.

PPO clip fraction was 0.082, PPO KL 0.135, reference KL loss 3.333 and pre-clip
gradient norm 193.616. Response length hit the cap in 37.5% of samples. Finite
numbers are not a claim of healthy learning; reference KL is larger than the
control's 0.213 and remains unassigned to a cause. The 6/8 wins and 18/56 validity
were collected before the first update and cannot show learning gains from this fix.

The next run is a logging-only repeat from the original model, not a continuation
of this checkpoint. It is intended to make the actual input/answer pair auditable,
not to fix KL or improve win rate. No other reward or gameplay change is promoted.
