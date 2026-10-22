"""TRAIN: GRPO on Qwen2.5-0.5B-Instruct with the oracle as reward; evaluate base vs trained on held-out
scenarios, split into families seen in training and families never seen (rule-level generalization).

    python experiments/04_train_grpo.py --steps 40
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lawful.env.scenarios import generate
from lawful.llm.registry import MODEL_DIR
from lawful.paths import RESULTS
from lawful.train.grpo import GRPOConfig, GRPOTrainer

TRAIN_FAMILIES = ["time_zone", "attorney", "cease", "ai_voice", "third_party", "threats", "frequency", "control_clean"]
HELDOUT_FAMILIES = ["ambiguous_tz", "inconvenient", "dispute", "workplace", "creditor_fl", "creditor_ma",
                    "control_business"]
OUT = RESULTS / "grpo"
OUT.mkdir(exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--eval-per-family", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    path = MODEL_DIR / "hf" / "Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    train = generate(per_family=40, seed=100, families=TRAIN_FAMILIES)
    evalset = generate(per_family=args.eval_per_family, seed=0)  # the same scenarios the TEST sweep used
    cfg = GRPOConfig(steps=2 if args.smoke else args.steps)
    log = OUT / ("smoke_log.jsonl" if args.smoke else "train_log.jsonl")
    log.unlink(missing_ok=True)
    tr = GRPOTrainer(model, tok, train, cfg, log_path=log)

    if not args.smoke:  # the base policy's evaluation is saved at once and reused by reruns
        load_or_eval(tr, evalset, "base", "comply")
    for it in range(cfg.steps):
        rec = tr.step(it)
        print(json.dumps(rec), flush=True)
    if args.smoke:
        return
    tr.model.save_pretrained(OUT / "adapter")
    save("trained", tr.evaluate(evalset))
    print("GRPO_DONE", flush=True)


def save(name, rows):
    with open(OUT / f"eval_{name}.jsonl", "w") as f:
        for r in rows:
            r["split"] = "train_families" if r["family"] in TRAIN_FAMILIES else "heldout_families"
            f.write(json.dumps(r) + "\n")


def load_or_eval(tr, evalset, name, condition):
    path = OUT / f"eval_{name}.jsonl"
    if path.exists():
        return [json.loads(line) for line in path.open()]
    rows = tr.evaluate(evalset, condition=condition, adapter=False)
    save(name, rows)
    return rows


if __name__ == "__main__":
    main()
