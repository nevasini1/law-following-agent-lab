#!/usr/bin/env bash
# Full TEST sweep: 7 open-weight variants under the "comply" prompt; prompt ablations on two of them.
set -euo pipefail
cd "$(dirname "$0")/.."
PF=8
python experiments/01_agent_eval.py --per-family $PF --models qwen2.5-0.5b gemma3-1b --conditions comply
python experiments/01_agent_eval.py --per-family $PF --models qwen2.5-1.5b --conditions comply bare pressure rag
python experiments/01_agent_eval.py --per-family $PF --models qwen3-4b --conditions comply pressure bare
python experiments/01_agent_eval.py --per-family $PF --models llama3.2-3b gemma3-4b phi4-mini --conditions comply
echo SWEEP_DONE
