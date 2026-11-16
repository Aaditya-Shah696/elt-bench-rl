# elt-bench-rl

An RL environment for [ELT-Bench](https://github.com/uiuc-kang-lab/ELT-Bench) on Tinker, built
with tinker-cookbook. The design is in [docs/design.md](docs/design.md).

## Setup

You need Docker, [uv](https://docs.astral.sh/uv/), `psql` and the AWS CLI.

Clone ELT-Bench next to this repo and download its data:

```bash
git clone https://github.com/uiuc-kang-lab/ELT-Bench.git ../ELT-Bench
cd ../ELT-Bench/setup
uvx gdown 'https://drive.google.com/uc?id=1-Gv5g_Yg_YrR-NxH4s2tSEK3VhJQc2Q0'
uvx gdown 'https://drive.google.com/uc?id=11vQqNEWXoPG6sjKytAn7TtFLDMiQa17I'
unzip -q data_db.zip -d . && unzip -q data_api.zip -d ../elt-docker/rest_api
cd .. && uvx --from huggingface_hub hf download tttjjj/elt_bench --repo-type dataset \
    --include "gt_snowflake/**" --local-dir ./ground_truth
cd ../elt-bench-rl
```

Then start everything:

```bash
uv sync
scripts/services.sh up
uv run python scripts/load_sources.py --only books
docker build -t elt-rl-agent docker/agent
```

For the model runs, log in with `uv run tinker auth login`. For Snowflake, run
`uv run python scripts/snowflake_setup.py --account <your account>` and follow what it prints.

## Running

Rollout on Snowflake:

```bash
uv run python -m elt_rl.rollout task=books env.destination=snowflake model_name=Qwen/Qwen3.6-35B-A3B
```

One RL step:

```bash
uv run python -m elt_rl.train max_steps=1
```

Output from my runs is in [results/](results/).
