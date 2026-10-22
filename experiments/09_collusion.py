"""COLLUDE: do LLM pricing agents coordinate prices, and do legal instruction / a runtime monitor stop it?

    python experiments/09_collusion.py --model qwen2.5-1.5b --rounds 20 --seeds 3
"""
import argparse
import json

from lawful.collude.market import P_MONO, P_NASH, play
from lawful.llm.client import Server
from lawful.paths import RESULTS

OUT = RESULTS / "collusion"
OUT.mkdir(exist_ok=True)
CONDITIONS = {"no_channel": dict(channel=False, law=False, monitor=False),
              "channel": dict(channel=True, law=False, monitor=False),
              "channel+law": dict(channel=True, law=True, monitor=False),
              "channel+monitor": dict(channel=True, law=False, monitor=True)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-1.5b")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    runs = []
    with Server(args.model, parallel=2) as llm:
        for name, kw in CONDITIONS.items():
            for seed in range(args.seeds):
                run = play(llm, args.rounds, seed=seed, **kw) | {"condition": name, "model": args.model}
                runs.append(run)
                print(f"{name:16s} seed={seed} index={run['collusion_index']:+.2f} "
                      f"agreement_msgs={run['agreement_msgs']}/{run['msgs']}", flush=True)
    (OUT / f"{args.model}.json").write_text(json.dumps({"p_nash": P_NASH, "p_mono": P_MONO, "runs": runs}, indent=1))


if __name__ == "__main__":
    main()
