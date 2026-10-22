"""PROBE control: how much of what the probes find is already in the prompt's words?

Rebuilds the exact P1 and P2 prompts of 07_probes.py and fits a bag-of-words classifier (TF-IDF + logistic
regression) with the same labels, scenario groups, grouped 5-fold CV and rule-family transfer splits. If words
alone separate the classes as well as the hidden states do, the probe is reading the text, not latent legal
knowledge. No model is needed; it runs in seconds.

    python experiments/07b_probe_text_baseline.py
"""
import importlib.util
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from lawful.agents.llm_agent import messages
from lawful.env.world import Scenario
from lawful.judge.judges import judge_messages
from lawful.law.oracle import evaluate
from lawful.paths import ROOT, SCENARIOS

spec = importlib.util.spec_from_file_location("probes_exp", ROOT / "experiments" / "07_probes.py")
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)


def text(chat):
    return "\n".join(m["content"] for m in chat)


def bow():
    return make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                         LogisticRegression(C=1.0, max_iter=3000))


def grouped_auroc(docs, y, groups):
    pred = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(docs, y, groups):
        pred[te] = bow().fit([docs[i] for i in tr], y[tr]).predict_proba([docs[i] for i in te])[:, 1]
    return float(roc_auc_score(y, pred))


def main():
    items = exp.p1_items()
    docs = [text(judge_messages(s, a)) for s, a, _ in items]
    y = np.array([lab for _, _, lab in items])
    groups = np.array([s.id for s, _, _ in items])
    fam = np.array([s.family for s, _, _ in items])
    fams, rng, transfers = sorted(set(fam)), np.random.default_rng(0), []
    for _ in range(10):  # the same family splits as 07_probes.py
        tr = np.isin(fam, list(rng.choice(fams, len(fams) // 2, replace=False)))
        te = ~tr
        clf = bow().fit([docs[i] for i in np.flatnonzero(tr)], y[tr])
        p = clf.predict_proba([docs[i] for i in np.flatnonzero(te)])[:, 1]
        transfers.append(float(roc_auc_score(y[te], p)) if len(set(y[te])) > 1 else float("nan"))
    out = {"p1": {"n": len(y), "auroc": grouped_auroc(docs, y, groups), "transfer_auroc_mean": float(np.nanmean(transfers)),
                  "transfer_auroc": transfers}}
    scen = [Scenario.from_dict(json.loads(line)) for line in open(SCENARIOS / "main.jsonl")]
    risky = np.array([int(evaluate(s, exp.naive_action(s)).violation) for s in scen])
    docs2 = [text(messages(s, "comply")) for s in scen]
    out["p2"] = {"n": len(scen), "auroc": grouped_auroc(docs2, risky, np.array([s.id for s in scen]))}
    (exp.OUT / "text_baseline.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {a: b for a, b in v.items() if a != "transfer_auroc"} for k, v in out.items()}), flush=True)


if __name__ == "__main__":
    main()
