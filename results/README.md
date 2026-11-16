# Results

Output from the two runs that need accounts, so you can look at them without rerunning.

`credentialed_rollout/` is one episode of Qwen3.6-35B-A3B on the `books` task on Snowflake. It
got a reward of 0.73: all 15 raw tables loaded and 2 of the 3 target tables were exact. It hit
the 40 turn limit I had at the time. The file has the full transcript and the replay logs.

`rl_step/` is one training step on Qwen3.5-4B with 4 rollouts on `books`. Rewards were 0.15,
0.20, 0.00 and 0.17. It has the metrics, every turn of the rollouts, and the config.
