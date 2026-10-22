#!/usr/bin/env bash
# Wait for the TEST sweep to finish, then run the REASON probes.
cd "$(dirname "$0")/.."
until grep -q SWEEP_DONE results/logs/agent_eval.log 2>/dev/null; do sleep 60; done
python experiments/02_reasoning.py --models qwen2.5-1.5b qwen3-4b > results/logs/reasoning.log 2>&1
echo REASON_DONE >> results/logs/reasoning.log
