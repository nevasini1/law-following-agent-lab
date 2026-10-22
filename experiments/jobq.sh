#!/usr/bin/env bash
# Sequential CPU job queue: runs results/queue/*.job (bash snippets) one at a time, in name order,
# after the TEST sweep finishes. Drop new .job files in at any time.
cd "$(dirname "$0")/.."
until grep -q SWEEP_DONE results/logs/agent_eval.log 2>/dev/null; do sleep 30; done
while true; do
  next=$(ls results/queue/*.job 2>/dev/null | sort | head -1)
  if [ -z "$next" ]; then sleep 30; continue; fi
  name=$(basename "$next" .job)
  mv "$next" "results/queue/$name.running"
  echo "[$(date +%H:%M:%S)] start $name" >> results/logs/jobq.log
  bash "results/queue/$name.running" > "results/logs/$name.log" 2>&1
  rc=$?  # capture before $(date) below resets $?
  echo "[$(date +%H:%M:%S)] end $name exit=$rc" >> results/logs/jobq.log
  mv "results/queue/$name.running" "results/queue/$name.done"
done
