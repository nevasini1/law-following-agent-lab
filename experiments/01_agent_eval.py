"""TEST: run open-weight agents through generated scenarios and score every decision with the legal oracle.

    python experiments/01_agent_eval.py --models qwen2.5-0.5b qwen3-4b --conditions comply bare
"""
import argparse
import json

from lawful.agents import retrieval
from lawful.agents.llm_agent import act, render_case
from lawful.env.scenarios import TARGET_PROVISIONS, generate
from lawful.law.oracle import score
from lawful.llm.client import Server, pmap
from lawful.paths import RESULTS, SCENARIOS

OUT = RESULTS / "agent_eval"
OUT.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--conditions", nargs="+", default=["comply"])
    ap.add_argument("--per-family", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scenarios = generate(args.per_family, args.seed)
    with open(SCENARIOS / "main.jsonl", "w") as f:
        for s in scenarios:
            f.write(json.dumps(s.to_dict()) + "\n")
    retrieved = {s.id: retrieval.render(retrieval.rank_hybrid(render_case(s))[:3]) for s in scenarios}

    for alias in args.models:
        with Server(alias) as llm:
            for cond in args.conditions:
                outs = pmap(lambda s, cond=cond: act(llm, s, cond, retrieved[s.id]), scenarios, desc=f"{alias}/{cond}")
                with open(OUT / f"{alias}__{cond}.jsonl", "w") as f:
                    for s, o in zip(scenarios, outs):
                        sc = score(s, o["action"])
                        sc["target_violation"] = bool(set(sc["provisions"]) & TARGET_PROVISIONS[s.family]
                                                      and sc["violation"])
                        f.write(json.dumps({"model": alias, "condition": cond, "scenario": s.id,
                                            "family": s.family, "raw": o["raw"],
                                            "action": o["action"].to_dict() if o["action"] else None,
                                            **sc}) + "\n")
                n = len(outs)
                rows = [json.loads(l) for l in open(OUT / f"{alias}__{cond}.jsonl")]
                print(f"{alias:14s} {cond:9s} n={n} violation={sum(r['violation'] for r in rows)/n:.2f} "
                      f"success={sum(r['success'] for r in rows)/n:.2f} "
                      f"format={sum(r['format_ok'] for r in rows)/n:.2f}", flush=True)


if __name__ == "__main__":
    main()
