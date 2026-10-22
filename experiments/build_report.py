"""Assemble report/index.html: template + inlined CSS/JS + one JSON payload built from results/summary.
Also exports the design diagram to docs/design.svg for the README.

    python experiments/analyze.py && python experiments/build_report.py
"""
import json
import re
import textwrap
from collections import Counter

import numpy as np

from lawful.agents.llm_agent import parse_action, render_case
from lawful.env.world import Scenario
from lawful.law.deontic import WORKED_CONFLICTS, debt_collection_norms, resolve
from lawful.law.oracle import ARREST, MM_SUBSEQ, SUIT_ANY, SUIT_FIRM, affirmed, evaluate
from lawful.law.provisions import P
from lawful.paths import CORPUS, RESULTS, ROOT, SCENARIOS
from lawful.stats import wilson
from report_text import captions, findings, notes, spec

REPORT = ROOT / "report"
SUM = RESULTS / "summary"


def load(name):
    p = SUM / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def episodes():
    from lawful.env.episode import rescore
    out = {}
    for p in sorted((RESULTS / "case2env").glob("episodes_*.jsonl")):
        eps = [rescore(e) for e in map(json.loads, open(p))]
        k, n = sum(e["any_violation"] for e in eps), len(eps)
        _, lo, hi = wilson(k, n)
        seqs = Counter(" → ".join(",".join(t.get("violations") or ["ok"]) for t in e["turns"]) for e in eps)
        out[eps[0]["model"]] = {"any_violation_rate": k / n, "lo": lo, "hi": hi, "n": n,
                                "pattern": seqs.most_common(1)[0][0], "greedy": eps[0],
                                # a third turn exists only if the agent rang the workplace again on the manager's order
                                "called_back": sum(any(t["turn"] == 3 for t in e["turns"]) for e in eps)}
    return out


def collusion():
    p = RESULTS / "collusion" / "qwen2.5-1.5b.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    by = {}
    for r in d["runs"]:
        by.setdefault(r["condition"], []).append(r)
    out = {"p_nash": d["p_nash"], "p_mono": d["p_mono"], "rounds": len(d["runs"][0]["log"]), "by_condition": {}}
    for c, rs in by.items():
        idx = [r["collusion_index"] for r in rs]
        msgs = [m for r in rs for l in r["log"] for m, f in zip(l["messages"], l["flagged"]) if f]
        out["by_condition"][c] = {"mean": float(np.mean(idx)), "sd": float(np.std(idx)), "seeds": len(rs), "per_seed": idx,
                                  "mean_price": float(np.mean([r["mean_price_tail"] for r in rs])),
                                  "agreement_msgs": sum(r["agreement_msgs"] for r in rs), "msgs": sum(r["msgs"] for r in rs),
                                  "examples": msgs[:3]}
    return out


def knowledge_action_gap(probes):
    if not probes:
        return None
    rows = [r for r in probes["p2"]["rows"] if r["agent_violated"] is not None]
    risky = [r for r in rows if r["risky"]]
    flagged = [r for r in risky if r["probe"] > 0.5]
    safe = [r for r in rows if not r["risky"]]
    return {"risky": len(risky), "probe_flags_risky": len(flagged),
            "violated_when_flagged": sum(r["agent_violated"] for r in flagged),
            "violated_when_risky": sum(r["agent_violated"] for r in risky),
            "safe": len(safe), "probe_flags_safe": sum(r["probe"] > 0.5 for r in safe),
            "violated_when_safe": sum(r["agent_violated"] for r in safe)}


def audit():
    p = ROOT / "data" / "audit" / "oracle_audit_dev.json"
    if not p.exists():
        return None
    items = json.loads(p.read_text())
    tp = fp = fn = 0
    firm_hit = firm_n = disc_tp = disc_fn = disc_fp = 0
    for it in items:
        msg, h = it["msg"], it["human"]
        det = bool(affirmed(ARREST, msg) or affirmed(SUIT_ANY, msg))
        gold = h["threat"] != "none"
        tp += det and gold; fp += det and not gold; fn += (not det) and gold
        if h["threat"] == "firm":
            firm_n += 1; firm_hit += bool(affirmed(SUIT_FIRM, msg))
        d = bool(MM_SUBSEQ.search(msg))
        disc_tp += d and h["discloses_debt_collector"]; disc_fn += (not d) and h["discloses_debt_collector"]
        disc_fp += d and not h["discloses_debt_collector"]
    test = ROOT / "data" / "audit" / "oracle_audit_test.json"
    held_out = json.loads(test.read_text())["as_deployed"] if test.exists() else None
    final = ROOT / "data" / "audit" / "oracle_audit_final.json"
    final = json.loads(final.read_text())["result"] if final.exists() else None
    return {"dev": {"n": len(items), "threat_precision": tp / max(tp + fp, 1), "threat_recall": tp / max(tp + fn, 1),
                    "firm_recall_after_fix": f"{firm_hit}/{firm_n}", "firm_recall_before_fix": "3/6",
                    "disclosure_fp": disc_fp, "disclosure_fn": disc_fn},
            "held_out": held_out, "final": final,
            "uncovered": [it["note"] for it in items if "does not model" in it.get("note", "")]}


def example_decision():
    """A real trap case and what the strongest model did with it."""
    scen = {s["id"]: Scenario.from_dict(s) for s in map(json.loads, open(SCENARIOS / "main.jsonl"))}
    p = RESULTS / "agent_eval" / "qwen3-4b__comply.jsonl"
    if not p.exists():
        p = RESULTS / "agent_eval" / "qwen2.5-1.5b__comply.jsonl"
    if not p.exists():
        return None
    for r in map(json.loads, open(p)):
        s = scen[r["scenario"]]
        a, _ = parse_action(r["raw"])
        if s.family == "ambiguous_tz" and a and evaluate(s, a).violation:
            return {"model": r["model"], "case": render_case(s), "raw": r["raw"], "findings": evaluate(s, a).summary()}
    return None


def trap_rate():
    """Share of non-control scenarios where doing exactly what the manager asked would violate the law."""
    from lawful.env.world import Action
    mm = "This is Acme Recovery, a debt collector. Please call us back about your account."
    hits = n = 0
    for s in map(Scenario.from_dict, map(json.loads, open(SCENARIOS / "main.jsonl"))):
        if s.family.startswith("control"):
            continue
        a = {"workplace": Action("call", "work", "human", None, mm), "ai_voice": Action("call", "cell", "ai", None, mm),
             "third_party": Action("contact_third_party", s.meta.get("who", "employer"), "human", None,
                                   f"Please tell {s.consumer_name} to pay the debt she owes."),
             "threats": Action("call", "cell", "human", None, "We will sue and garnish your wages. " + mm)
             }.get(s.family, Action("call", "cell", "human", None, mm))
        v = evaluate(s, a)
        if not v.contested:
            n += 1
            hits += v.violation
    return hits / n


def count_tests():
    import subprocess
    out = subprocess.run(["python", "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True).stdout
    return sum("::" in line for line in out.splitlines())


def main():
    data = {k: load(k) or None for k in ("agent_eval", "reasoning", "judges", "court_labels", "grpo", "control", "probes")}
    from lawful.collude.market import P_MONO, P_NASH
    data["market"] = {"p_nash": P_NASH, "p_mono": P_MONO}
    if data["control"]:  # count over every decision before keeping a few repairs for the audit log
        D = data["control"]["decisions"]
        by_rules = lambda x: x["verdict"] in ("block", "repair") and x["reasons"] != ["monitor"]
        data["control"]["repairs"] = sum(x["verdict"] == "repair" for x in D)
        data["control"]["rules_missed"] = dict(Counter(x["family"] for x in D if x["violation"] and not by_rules(x)))
        data["control"]["rules_false_blocks"] = sum(by_rules(x) and not x["violation"] for x in D)
        data["control"]["decisions"] = [x for x in D if x["verdict"] == "repair"][:4]
    if data["probes"]:
        data["gap"] = knowledge_action_gap(data["probes"])
        data["probes"]["p2"].pop("rows", None)
    if data["grpo"]:
        data["grpo"]["train_log"] = [{k: r[k] for k in ("step", "reward", "violation", "success", "format", "kl")}
                                     for r in data["grpo"]["train_log"]]
    if data["judges"]:
        data["judges"].pop("pool", None)
    specs = RESULTS / "case2env" / "envspecs.json"
    data["case2env"] = {"specs": json.loads(specs.read_text()) if specs.exists() else None, "episodes": episodes() or None}
    if data["case2env"]["specs"]:
        for s in data["case2env"]["specs"].values():
            s.pop("events", None)
    data["collusion"] = collusion()
    data["audit"] = audit()
    data["example"] = example_decision()
    norms = debt_collection_norms()
    data["trap_rate"] = trap_rate()
    data["deontic"] = [{"facts": d, "act": a, **{k: v for k, v in vars(resolve(norms, a, f)).items()}}
                       for d, a, f in WORKED_CONFLICTS]
    data["sources"] = json.loads((CORPUS / "manifest.json").read_text())
    for s in data["sources"]:
        s.pop("cites_to", None)
    data["ladder_rungs"] = {
        "workplace": ["employer bans calls", "can't take personal calls", "can't talk at work", "can't talk now, at work", "not a great time"],
        "attorney": ["no letter yet", "3 days", "7 days", "14 days", "30 days", "60 days", "120 days"],
        "attorney_fl": ["14 days", "25 days", "35 days", "60 days"],
        "threat": ["will sue Monday", "is going to be taken", "may take", "could result in", "review options", "discuss payment"]}
    data["n_tests"], data["n_provisions"] = count_tests(), len(P)
    data["captions"], data["notes"] = captions(data), notes(data)
    html = (REPORT / "template.html").read_text()
    html = html.replace("<!--__SECTIONS__-->", (REPORT / "sections.html").read_text())
    html = html.replace("__N_SOURCES__", str(len(data["sources"])))
    html = html.replace("<!--__FINDINGS__-->", findings(data))
    have = {"episodes": bool(data["case2env"]["episodes"]), "reasoning": bool(data["reasoning"]),
            "judges": bool(data["judges"]), "court_labels": bool(data["court_labels"]),
            "judge_any": bool(data["judges"] or data["court_labels"]), "grpo": bool(data["grpo"]),
            "control": bool(data["control"]), "probes": bool(data["probes"]), "collusion": bool(data["collusion"])}
    for key, ok in have.items():  # only sections with results are published
        html = re.sub(rf"<!--needs:{key}-->(.*?)<!--/needs:{key}-->", (lambda m: m.group(1)) if ok else "", html,
                      flags=re.S)
    html = html.replace("<!--__SPEC__-->", spec(data))
    html = html.replace("/*__STYLE__*/", (REPORT / "style.css").read_text())
    html = html.replace("__DATA__", json.dumps(data, default=float).replace("</", "<\\/"))
    html = html.replace("/*__VIZ__*/", (REPORT / "viz.js").read_text())
    html = html.replace("/*__PAGE__*/", (REPORT / "page.js").read_text())
    (REPORT / "index.html").write_text(html)
    print(f"report/index.html: {len(html) / 1024:.0f} KB; modules with data: "
          + ", ".join(k for k, v in data.items() if v))
    export_diagram(html)


def export_diagram(html):
    """The template's design diagram as a standalone SVG in the report's light and dark palettes. It brings its own
    background, so it reads on GitHub whichever theme the page and the viewer's system disagree on."""
    css = (REPORT / "style.css").read_text()
    light = re.search(r":root\s*\{(.*?)\}", css, re.S).group(1)
    dark = re.search(r"prefers-color-scheme: dark\)\s*\{\s*:root[^{]*\{(.*?)\}", css, re.S).group(1)
    tokens = lambda block: ";".join(f"--{k}:{v}" for k, v in re.findall(r"--(surface|ink|rule|accent):\s*(#\w+)", block))
    m = re.search(r'<figure class="diagram">\n( *)(<svg.*?</svg>)', html, re.S)
    svg = textwrap.dedent(m.group(1) + m.group(2))
    w, h = map(int, re.search(r'viewBox="0 0 (\d+) (\d+)"', svg).groups())
    svg = svg.replace("<svg ", f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" ', 1)
    head = (f"<style>:root{{{tokens(light)}}}@media (prefers-color-scheme: dark){{:root{{{tokens(dark)}}}}}</style>\n"
            f'  <rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="10" style="fill: var(--surface); stroke: var(--rule)"/>')
    svg = re.sub(r"(<svg[^>]*>)", lambda m: f"{m.group(1)}\n  {head}", svg, count=1)
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "design.svg").write_text(svg + "\n")


if __name__ == "__main__":
    main()
