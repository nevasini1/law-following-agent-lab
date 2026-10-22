#!/usr/bin/env bash
# Remainder of the TEST sweep (trimmed plan; finished runs are served from the inference cache).
set -euo pipefail
cd "$(dirname "$0")/.."
PF=8
python experiments/01_agent_eval.py --per-family $PF --models qwen2.5-1.5b --conditions comply bare pressure rag
python experiments/01_agent_eval.py --per-family $PF --models qwen3-4b --conditions comply pressure
python experiments/01_agent_eval.py --per-family $PF --models llama3.2-3b --conditions comply
echo SWEEP_DONE
