#!/bin/bash

# TODO hdm: created a copy for testing so I can easily comment out certain bits while easily
# referencing the actual version. Will delete later.

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# Run as:
# bash dev/cpu_demo_run.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# This is also why I hide this script away in dev/

# all the setup stuff
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# wipe the report
# TODO hdm: maybe check if we need to/can update to write to run specific dir each time instead of overwriting?
python -m nanochat.report reset

# train tokenizer on ~1B characters
# Downloading dataset will take a while.
python -m nanochat.dataset -n 4
python -m scripts.tok_train --max_chars=1000000000
python -m scripts.tok_eval

# train a very small 4 layer model on the CPU
# each optimization step processes a single sequence of 1024 tokens
# we only run 50 steps of optimization (bump this to get better results)
# hdm: changed from num_iterations=50 to auto-computed based on chinchilla scaling, which comes out
# to ~700k steps. This ran in ~24 hours on m1 mac. Increased core_metric_every a lot because this
# was taking up the bulk of training time with karpathy's default.
python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --total_batch_size=1024 \
    --eval_tokens=4096 \
    --core_metric_every=20000 \
    --core_metric_max_per_task=12 \
    --journal_freq=10000
python -m scripts.base_loss --device_batch_size=1 --split_tokens=4096
python -m scripts.base_eval --max-per-task=16

# midtraining
# TODO hdm: remove num_iterations before full run
python -m scripts.mid_train \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --eval_every=50 \
    --eval_tokens=4096 \
    --total_batch_size=1024
# eval results will be terrible, this is just to execute the code paths.
# note that we lower the execution memory limit to 1MB to avoid warnings on smaller systems
python -m scripts.chat_eval --source=mid --max-new-tokens=128 --max-problems=20

# SFT
# hdm: removed num_iterations and eval_steps
python -m scripts.chat_sft \
    --device_batch_size=1 \
    --target_examples_per_step=4 \
    --eval_metrics_max_problems=16

# Chat CLI
python -m scripts.chat_cli -p "Why is the sky blue?"

# Chat Web
python -m scripts.chat_web

python -m nanochat.report generate
