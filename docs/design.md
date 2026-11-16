# ELT-Bench RL environment on Tinker

This is an RL environment for [ELT-Bench](https://github.com/uiuc-kang-lab/ELT-Bench) built on
tinker-cookbook. The model gets an ELT task, works on it with tools in a sandbox, and is rewarded
by rerunning its pipeline from scratch and checking the warehouse.

## Results

1. Integration test without credentials (`uv run pytest`): a scripted reference solution gets
   1.0 on the `books` task and the scripted cheaters get 0.
2. Credentialed rollout: Qwen3.6-35B-A3B (not trained) got 0.73 on `books` on a real Snowflake
   account. The reference solution gets 1.0 there.
3. One Tinker RL step on Qwen3.5-4B. The group of 4 rollouts got 0.15, 0.20, 0.00 and 0.17, and
   the step cost about $0.89.

## 1. The task

Each ELT-Bench task has a `config.yaml` with the source connections, a `data_model.yaml` that
describes the target tables, and docs for the source tables. The agent has to load data from
Postgres, MongoDB, S3, a REST API and flat files into a warehouse, then write dbt models that
build the target tables. The expected row counts, the table keys and the answer tables are
hidden. There are 100 tasks with 1 to 9 target tables each.

Every rollout gets its own Docker sandbox and its own warehouse namespace, with a login that can
only see that namespace. The tools are `bash`, `read_file`, `write_file` (only under
`/workspace/elt`), `run_sql` (read-only) and `submit`. The episode ends on `submit`, when the
turn or token budget runs out (50 turns), or after three turns in a row with no tool call.

ELT-Bench uses Airbyte and Terraform for loading. I replaced that with `elt-load`, a small loader
the agent configures with a `load.yaml`. Airbyte takes minutes per sync and needs an 8 GB
install, which is too slow for RL.

## 2. Reward

```
reward = 0.2 * (fraction of raw tables with the right row count)
       + 0.8 * average over target tables of (1 if exact, else 0.5 * fraction of non-key columns right)
```

- The reward never looks at the agent's own warehouse. When the episode ends I copy its `elt/`
  folder into a fresh sandbox pointed at an empty namespace, rerun `elt-load` and `dbt run`, and
  grade that. So tables made by hand or tools changed inside the sandbox don't count.
- A pass/fail reward would almost always be 0 for a small model on a 30 to 50 turn task, so
  loading and individual columns get partial credit. Both are still checked on the rerun.
- I didn't reward things like "dbt run exited 0", because an empty project also exits 0.
- Key columns get no partial credit (copying ids is free) but must match for the other columns
  to count.
- The table comparison is ELT-Bench's own code, so a 1.0 is also an official pass.

## 3. Implementation

The main pieces are a task loader, the grader, a Docker sandbox (the cookbook only has Modal and
SandboxFusion), the tools, and the env, which follows the cookbook's `harbor_rl` recipe.
Everything warehouse-specific is behind a `Destination` class, so Postgres (local, the default)
and Snowflake are one file each. Training uses `build_agent_tool_env` and `rl.train.main` with
LoRA, async `stable_reinforce` and retries for sandbox failures.

I hit two problems with the cookbook during real rollouts. The runner can stop an episode before
the env does, which skipped the reward, so I grade those episodes in `compute_group_rewards`. A
turn without a tool call (usually one cut off by the token limit) ended the whole episode, so now
it gets a nudge instead.

Other changes from ELT-Bench: flat files are served locally instead of from Google Drive,
LocalStack is pinned to 4.9 because later versions need a paid token, and the source logins are
read-only.

## 4. Reward hacking

- Reading the answers: they are never in the sandbox, and the sandbox has no internet (Snowflake
  goes through a proxy that only allows Snowflake).
- Claiming success or building tables by hand: only the rerun is graded.
- Faking tools in the sandbox: the rerun uses a clean container.
- Pasting the answers into a model as literals: a model that reads no source table or has more
  than 40 literals scores 0.
- Changing the sources to hurt the other rollouts in the group: the Postgres login is read-only.
  MongoDB and S3 can't be locked, so I fingerprint them before and after each group and zero the
  group if anything changed.
- Copying id columns or adding extra rows: keys get no credit and row counts are checked.

`tests/test_reward_hacking.py` has a cheater for most of these. What's still open: the literal
check is a heuristic, so values split across files or encoded would get past it. Rerunning on
changed source data would catch that, but needs a reference solution per task. Also, the source
superuser password is ELT-Bench's public default, so writes with it are detected but not blocked.

## 5. Speed

The environment isn't the bottleneck. With a scripted policy, one `books` episode takes about 7 s
of environment time on local Postgres and about 2 minutes on Snowflake (mostly loading). Real
episodes spend 2 to 15 minutes sampling from the model, and in the RL step sampling took 135 s
against 9 s for training. `scripts/benchmark_env.py` measures this.

Things that help: `elt-load` instead of Airbyte, transform-only episodes
(`env.start_loaded=true`) that clone already-loaded raw tables instead of loading them again,
async training, clipping tool output to 8k characters, and Tinker's prompt cache (94% of prefill
tokens were cached).
