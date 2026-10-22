"""Aggregate every experiment into results/summary/*.json (numbers only; figures are made separately).

All agent decisions are re-scored here with the *current* oracle from their saved raw outputs, so a fix to a
detector propagates to every table without re-running inference.
"""
import json
import math
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from lawful.agents.llm_agent import parse_action
from lawful.env.scenarios import TARGET_PROVISIONS
from lawful.env.world import Scenario
from lawful.judge.dawid_skene import dawid_skene
from lawful.judge.ppi import coverage_study
from lawful.law.oracle import score
from lawful.paths import RESULTS, SCENARIOS
from lawful.stats import cohen_kappa, ece, mcnemar, wilson

SUM = RESULTS / "summary"
SUM.mkdir(exist_ok=True)


def _w(k, n):
    p, lo, hi = wilson(k, n)
    return {"k": int(k), "n": int(n), "p": p, "lo": lo, "hi": hi}


def agent_eval():
    scen = {s["id"]: Scenario.from_dict(s) for s in map(json.loads, open(SCENARIOS / "main.jsonl"))}
    runs = {}
    for path in sorted((RESULTS / "agent_eval").glob("*.jsonl")):
        rows = []
        for r in map(json.loads, open(path)):
            s = scen[r["scenario"]]
            a, _ = parse_action(r["raw"])
            m = score(s, a)
            m["target_violation"] = bool(m["violation"] and set(m["provisions"]) & TARGET_PROVISIONS[s.family])
            rows.append({"scenario": s.id, "family": s.family, "tool": a.tool if a else None, **m})
        runs[path.stem] = rows
    out = {"runs": {}, "families": {}, "ablations": {}}
    for name, rows in runs.items():
        model, cond = name.split("__")
        scored = [r for r in rows if not r["contested"]]
        lawful_exists = [r for r in scored if r["family"] != "cease"]
        out["runs"][name] = {
            "model": model, "condition": cond, "n": len(rows), "n_scored": len(scored),
            "violation": _w(sum(r["violation"] for r in scored), len(scored)),
            "success": _w(sum(r["success"] for r in scored), len(scored)),
            "disclosure_violation": _w(sum(r["disclosure_violation"] for r in rows), len(rows)),
            "format": _w(sum(r["format_ok"] for r in rows), len(rows)),
            "refusal": _w(sum((r["tool"] in ("no_action", "escalate")) for r in lawful_exists), len(lawful_exists)),
            "controls_success": _w(sum(r["success"] for r in scored if r["family"].startswith("control")),
                                   sum(r["family"].startswith("control") for r in scored)),
            "tools": Counter(r["tool"] for r in rows),
        }
        fam = defaultdict(list)
        for r in scored:
            fam[r["family"]].append(r)
        out["families"][name] = {f: {"violation": _w(sum(r["violation"] for r in rs), len(rs)),
                                     "success": _w(sum(r["success"] for r in rs), len(rs))} for f, rs in fam.items()}
    for model in sorted({n.split("__")[0] for n in runs}):
        base = runs.get(f"{model}__comply")
        for cond in ("bare", "pressure", "rag"):
            other = runs.get(f"{model}__{cond}")
            if not base or not other:
                continue
            keep = [i for i, (a, b) in enumerate(zip(base, other)) if not a["contested"] and not b["contested"]]
            va = [base[i]["violation"] for i in keep]
            vb = [other[i]["violation"] for i in keep]
            out["ablations"][f"{model}:{cond}"] = {"comply": float(np.mean(va)), cond: float(np.mean(vb)),
                                                   "delta": float(np.mean(vb) - np.mean(va)),
                                                   "mcnemar_p": mcnemar(va, vb), "n": len(keep)}
    (SUM / "agent_eval.json").write_text(json.dumps(out, indent=1, default=dict))
    return out


def reasoning():
    out = {}
    for path in sorted((RESULTS / "reasoning").glob("*.jsonl")):
        rows = list(map(json.loads, open(path)))
        comp = [r for r in rows if r["probe"] == "compositional" and not math.isnan(r["p_lawful"])]
        byk = {}
        for k in sorted({r["k"] for r in comp}):
            rs = [r for r in comp if r["k"] == k]
            y = np.array([r["lawful"] for r in rs])
            p = np.array([r["p_lawful"] for r in rs])
            byk[k] = {"n": len(rs), "auroc": float(roc_auc_score(y, p)),
                      "balanced_acc": float(((p > 0.5) == y)[y].mean() / 2 + ((p <= 0.5) == ~y)[~y].mean() / 2),
                      "acc_lawful": float((p > 0.5)[y].mean()), "acc_unlawful": float((p <= 0.5)[~y].mean())}
        lad = [r for r in rows if r["probe"] == "ladder"]
        ladders = {}
        for name in sorted({r["ladder"] for r in lad}):
            rs = [r for r in lad if r["ladder"] == name]
            rungs = sorted({r["rung"] for r in rs})
            pv = {g: np.array([1 - r["p_lawful"] for r in rs if r["rung"] == g]) for g in rungs}
            means = [float(np.nanmean(pv[g])) for g in rungs]
            labels = [next(r["label"] for r in rs if r["rung"] == g) for g in rungs]
            # monotonicity: stronger facts (earlier rungs) should never get lower P(violation) than weaker ones,
            # checked within each context (irrelevant facts held fixed)
            ctxs = sorted({r["context"] for r in rs})
            viol = tot = 0
            for c in ctxs:
                seq = [1 - next(r["p_lawful"] for r in rs if r["rung"] == g and r["context"] == c) for g in rungs]
                for i in range(len(seq)):
                    for j in range(i + 1, len(seq)):
                        tot += 1
                        viol += seq[j] > seq[i] + 0.05
            within = float(np.nanmean([np.nanstd(pv[g]) for g in rungs]))
            between = float(np.nanstd(means))
            H = lambda p: -(p * np.log2(p + 1e-9) + (1 - p) * np.log2(1 - p + 1e-9))
            core = [H(np.clip(pv[g], 1e-6, 1 - 1e-6)).mean() for g, l in zip(rungs, labels) if l in ("clear", "yes", "none", "no")]
            pen = [H(np.clip(pv[g], 1e-6, 1 - 1e-6)).mean() for g, l in zip(rungs, labels) if l == "contested"]
            anchored = [(m, l) for m, l in zip(means, labels) if l in ("clear", "yes")]
            ladders[name] = {"rung_means": means, "labels": labels, "monotonicity_violation_rate": viol / max(tot, 1),
                             "within_rung_sd": within, "between_rung_sd": between,
                             "signal_to_noise": between / within if within > 0 else float("inf"),
                             "entropy_core": float(np.mean(core)) if core else None,
                             "entropy_penumbra": float(np.mean(pen)) if pen else None,
                             "anchor_hit_rate": float(np.mean([m > 0.5 for m, _ in anchored])) if anchored else None}
        out[path.stem] = {"compositional": byk, "ladders": ladders}
    (SUM / "reasoning.json").write_text(json.dumps(out, indent=1))
    return out


def judges():
    path = RESULTS / "judges" / "results.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    pool = res["pool"]
    y = np.array([p["violation"] for p in pool]).astype(int)
    kind = np.array([p["kind"] for p in pool])
    out = {"n": len(pool), "violation_rate": float(y.mean()), "judges": {}, "probes": {}}
    scores = {name: np.array([0.5 if (v is None or (isinstance(v, float) and math.isnan(v))) else v for v in vals])
              for name, vals in res["judges"].items()}
    scores["keyword"] = np.array([p["keyword"] for p in pool])
    for name, p in scores.items():
        pred = (p > 0.5).astype(int)
        out["judges"][name] = {
            "auroc": float(roc_auc_score(y, p)), "accuracy": float((pred == y).mean()), "kappa": cohen_kappa(pred, y),
            "ece": ece(p, y), "recall_state": float(pred[(y == 1) & (kind == "state")].mean()),
            "recall_content": float(pred[(y == 1) & (kind == "content")].mean()),
            "false_positive_rate": float(pred[y == 0].mean()), "mean_p": float(p.mean())}
    # Dawid-Skene over binarized judges, no gold
    names = [n for n in scores if n != "keyword"] + ["keyword"]
    L = np.stack([(scores[n] > 0.5).astype(int) for n in names], 1)
    T, conf, prior = dawid_skene(L)
    est_acc = {n: float(conf[j, 0, 0] * prior[0] + conf[j, 1, 1] * prior[1]) for j, n in enumerate(names)}
    true_acc = {n: float(((scores[n] > 0.5).astype(int) == y).mean()) for n in names}
    out["dawid_skene"] = {"estimated_accuracy": est_acc, "true_accuracy": true_acc,
                          "aggregate_accuracy": float(((T[:, 1] > 0.5).astype(int) == y).mean()),
                          "majority_accuracy": float(((L.mean(1) > 0.5).astype(int) == y).mean()),
                          "estimated_prevalence": float(prior[1])}
    # PPI with the best LLM judge
    best = max((n for n in scores if n not in ("keyword",)), key=lambda n: out["judges"][n]["auroc"])
    out["ppi"] = {"judge": best, "studies": [coverage_study(y, (scores[best] > 0.5).astype(float), n, reps=2000)
                                             for n in (20, 40, 80)]}
    # runs before probe_judge was recorded probed the last LLM judge (results/queue/50_judges: --probe-judge qwen3-4b)
    out["probe_judge"] = res.get("probe_judge") or [n for n in res["judges"] if n != "qwen3guard-0.6b"][-1]
    for key, d in res["probes"].items():
        base = scores[out["probe_judge"]][d["idx"]]
        pert = np.array([0.5 if v is None or math.isnan(v) else v for v in d["p"]])
        out["probes"][key] = {"n": len(pert), "mean_p_before": float(base.mean()), "mean_p_after": float(pert.mean()),
                              "flip_rate": float(((base > 0.5) != (pert > 0.5)).mean()),
                              "flip_to_compliant": float(((base > 0.5) & (pert <= 0.5)).mean())}
    (SUM / "judges.json").write_text(json.dumps(out, indent=1))
    return out


def court_labels():
    path = RESULTS / "court_labels" / "results.json"
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    curve = defaultdict(list)
    for r in res["ecthr_curve"]:
        curve[(r["judge"], r["n_train"])].append(r["auroc"])
    out = {"curve": [{"judge": j, "n_train": n, "auroc_mean": float(np.mean(v)), "auroc_sd": float(np.std(v))}
                     for (j, n), v in sorted(curve.items())], "fods_trained": res.get("fods_trained"),
           "meta": res.get("ecthr_meta"), "llm": {}}
    for alias, rows in res.get("ecthr_llm", {}).items():
        yy = np.array([r["y"] for r in rows])
        pp = np.array([0.5 if math.isnan(r["p"]) else r["p"] for r in rows])
        idx = [r["idx"] for r in rows]
        same = {}
        for name in ("tfidf+lr", "bge-small+lr"):
            f = RESULTS / "court_labels" / f"ecthr_test_p_{name}.npy"
            if f.exists():
                same[name] = float(roc_auc_score(yy, np.load(f)[idx]))
        out["llm"][alias] = {"n": len(rows), "auroc": float(roc_auc_score(yy, pp)), "trained_on_same_subset": same,
                             "fods": {k: v for k, v in res.get("fods_llm", {}).get(alias, {}).items() if k != "pairs"}}
    (SUM / "court_labels.json").write_text(json.dumps(out, indent=1))
    return out


def grpo():
    d = RESULTS / "grpo"
    if not (d / "eval_trained.jsonl").exists():
        return None
    log = list(map(json.loads, open(d / "train_log.jsonl")))
    out = {"train_log": log, "eval": {}}
    for name in ("base", "trained", "base_pressure", "trained_pressure"):
        if not (d / f"eval_{name}.jsonl").exists():
            continue
        rows = list(map(json.loads, open(d / f"eval_{name}.jsonl")))
        out["eval"][name] = {}
        for split in ("train_families", "heldout_families", "all"):
            rs = [r for r in rows if split == "all" or r["split"] == split]
            sc = [r for r in rs if not r.get("contested")]
            out["eval"][name][split] = {
                "violation": _w(sum(r.get("violation", False) for r in sc), len(sc)),
                "violation_valid": _w(sum(r.get("violation", False) for r in sc if r.get("format_ok")),
                                      sum(bool(r.get("format_ok")) for r in sc)),
                "success": _w(sum(r.get("success", False) for r in sc), len(sc)),
                "format": _w(sum(r.get("format_ok", False) for r in rs), len(rs)),
                "disclosure_violation": _w(sum(r.get("disclosure_violation", False) for r in rs), len(rs)),
                "reward": float(np.mean([r["reward"] for r in rs]))}
        ctrl = [r for r in rows if r["family"].startswith("control")]
        out["eval"][name]["controls_success"] = _w(sum(r.get("success", False) for r in ctrl), len(ctrl))
        out["eval"][name]["tools"] = Counter(r.get("tool", "unparsed") for r in rows)
        out["eval"][name]["lawful_calls"] = sum(r.get("tool") == "call" and r.get("success", False) for r in rows)
        fv = {}
        for r in rows:
            if not r.get("contested"):
                k, n = fv.get(r["family"], (0, 0))
                fv[r["family"]] = (k + bool(r.get("violation")), n + 1)
        out["eval"][name]["family_violations"] = fv
    b = list(map(json.loads, open(d / "eval_base.jsonl")))
    t = list(map(json.loads, open(d / "eval_trained.jsonl")))
    out["mcnemar_violation"] = mcnemar([r.get("violation", False) for r in b], [r.get("violation", False) for r in t])
    out["mcnemar_success"] = mcnemar([r.get("success", False) for r in b], [r.get("success", False) for r in t])
    (SUM / "grpo.json").write_text(json.dumps(out, indent=1, default=dict))
    return out


def probes():
    path = RESULTS / "probes" / "results.json"
    if not path.exists():
        return None
    out = json.loads(path.read_text())
    tb = RESULTS / "probes" / "text_baseline.json"
    if tb.exists():  # experiments/07b_probe_text_baseline.py: the same prompts, folds and labels, words only
        out["text_baseline"] = json.loads(tb.read_text())
    (SUM / "probes.json").write_text(json.dumps(out))
    return out


def copy_if(name):
    path = RESULTS / name / "results.json"
    if path.exists():
        (SUM / f"{name}.json").write_text(path.read_text())


def main():
    for fn in (agent_eval, reasoning, judges, court_labels, grpo, probes):
        try:
            r = fn()
            print(f"{fn.__name__}: {'ok' if r is not None else 'no results yet'}")
        except FileNotFoundError as e:
            print(f"{fn.__name__}: missing input ({e.filename})")
    copy_if("control")


if __name__ == "__main__":
    main()
