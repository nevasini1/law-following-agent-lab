"""REASON: compositional depth + open-texture ladders, scored as P(lawful) from each model's Yes/No logits.

    python experiments/02_reasoning.py --models qwen2.5-1.5b qwen3-4b
"""
import argparse
import json

from lawful.env.probes import compositional, ladders
from lawful.judge.judges import judge_messages
from lawful.llm.client import Server, pmap
from lawful.paths import RESULTS

OUT = RESULTS / "reasoning"
OUT.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--per-cell", type=int, default=16)
    args = ap.parse_args()
    comp = compositional(args.per_cell)
    lad = ladders(6)
    for alias in args.models:
        with Server(alias) as llm:
            pc = pmap(lambda it: llm.p_yes(judge_messages(it[0], it[1])), comp, desc=f"{alias}/compositional")
            pl = pmap(lambda it: llm.p_yes(judge_messages(it[3], it[4])), lad, desc=f"{alias}/ladders")
        with open(OUT / f"{alias}.jsonl", "w") as f:
            for (s, _a, lawful), p in zip(comp, pc):
                f.write(json.dumps({"probe": "compositional", "id": s.id, "k": s.meta["k"], "lawful": lawful,
                                    "considerations": s.meta["considerations"], "needle": s.meta["needle"],
                                    "p_lawful": p}) + "\n")
            for (ladder, rung, ctx, _s, _a, label), p in zip(lad, pl):
                f.write(json.dumps({"probe": "ladder", "ladder": ladder, "rung": rung, "context": ctx,
                                    "label": label, "p_lawful": p}) + "\n")
        print(f"{alias}: wrote {len(comp)} compositional + {len(lad)} ladder items", flush=True)


if __name__ == "__main__":
    main()
