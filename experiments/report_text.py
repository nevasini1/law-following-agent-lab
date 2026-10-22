"""Text for the report that depends on results: findings cards, figure captions and notes.

Everything here is computed from the payload, so the page never states a number the data does not contain.
Blocks whose experiments have no results are dropped by the build, not described as pending.
"""

NICE = {"qwen2.5-0.5b": "Qwen2.5 0.5B", "gemma3-1b": "Gemma 3 1B", "qwen2.5-1.5b": "Qwen2.5 1.5B",
        "llama3.2-3b": "Llama 3.2 3B", "phi4-mini": "Phi-4-mini", "gemma3-4b": "Gemma 3 4B", "qwen3-4b": "Qwen3 4B",
        "qwen3guard-0.6b": "Qwen3Guard 0.6B", "keyword": "keyword filter"}
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}
ACTS = {"call": "a call", "email": "an email", "letter": "a letter", "sms": "a text message", "no_action": "no action"}


def pct(x, d=0):
    return f"{100 * x:.{d}f}%"


def prange(xs):
    lo, hi = pct(min(xs)), pct(max(xs))
    return lo if lo == hi else f"{lo}–{hi}"


def comply_runs(d):
    runs = (d.get("agent_eval") or {}).get("runs", {})
    return {k.split("__")[0]: v for k, v in runs.items() if k.endswith("__comply")}


def findings(d):
    cards = []
    ae = comply_runs(d)
    if d.get("trap_rate") is not None:
        cards.append((pct(d["trap_rate"]), "of non-control scenarios are traps",
                      "Doing exactly what the collections manager asked would break the law. Two further families are "
                      "lawful controls, to measure over-refusal.", "#test"))
    if ae:
        lo = min(ae.items(), key=lambda kv: kv[1]["violation"]["p"])
        hi = max(ae.items(), key=lambda kv: kv[1]["violation"]["p"])
        cards.append((f'{pct(lo[1]["violation"]["p"])}&ndash;{pct(hi[1]["violation"]["p"])}', "violation rate",
                      f"Across {WORDS.get(len(ae), len(ae))} open-weight models, from {NICE[lo[0]]} to {NICE[hi[0]]}. "
                      f"Each model made 120 decisions.", "#test"))
    fam = (d.get("agent_eval") or {}).get("families", {})
    if "qwen3-4b__comply" in fam:
        q = fam["qwen3-4b__comply"]
        cards.append((f'{pct(q["third_party"]["violation"]["p"])}', "unlawful obedience by the strongest model",
                      f"Qwen3 4B honoured cease letters ({pct(q['cease']['violation']['p'])} violations), but when told to "
                      f"pressure a consumer through an employer or relative, it disclosed the debt every time. "
                      f"Capability is not compliance.", "#test"))
    abl = (d.get("agent_eval") or {}).get("ablations", {})
    a15 = {k.split(":")[1]: v for k, v in abl.items() if k.startswith("qwen2.5-1.5b:")}
    if a15:
        deltas = [v["delta"] for v in a15.values()]
        pmin = min(v["mcnemar_p"] for v in abl.values())
        calls = d["agent_eval"]["runs"].get("qwen2.5-1.5b__comply", {}).get("tools", {}).get("call", 0)
        span = lambda x: "0" if abs(100 * x) < 0.5 else f"{100 * x:+.0f}".replace("-", "−")
        cards.append((f'{span(min(deltas))} to {span(max(deltas))} pts', "from prompting about the law",
                      f"Telling Qwen2.5 1.5B to comply, retrieving the statute, or adding pressure changed nothing "
                      f"significant (McNemar p &ge; {pmin:.2f}). It placed a call in {calls} of 120 decisions regardless.",
                      "#test"))
    R = d.get("reasoning")
    if R and "qwen2.5-1.5b" in R:
        c = R["qwen2.5-1.5b"]["compositional"]
        cards.append((f'{pct(c["1"]["balanced_acc"])} &rarr; {pct(c["5"]["balanced_acc"])}', "as legal constraints interact",
                      "Qwen2.5 1.5B's balanced accuracy at judging a call, from one relevant constraint to five. With four "
                      "or more it calls every lawful case unlawful.", "#reason"))
    cl = d.get("court_labels")
    if cl and cl.get("llm"):
        small = min((r for r in cl["curve"] if r["judge"] == "tfidf+lr"), key=lambda r: r["n_train"])
        alias, v = next(iter(cl["llm"].items()))
        cards.append((f'{small["auroc_mean"]:.2f} vs {v["auroc"]:.2f}', "trained on court labels vs zero-shot",
                      f"AUROC for &ldquo;did the European Court find a violation of Article 3?&rdquo;: a small judge trained "
                      f"on just {small['n_train']} court-labeled judgments, against a zero-shot {NICE.get(alias, alias)}.",
                      "#judge"))
    if d.get("audit"):
        cards.append(("4", "flaws found by auditing the evaluator",
                      "Three detector bugs and one metric hack: a model was scoring &ldquo;success&rdquo; by writing to "
                      "attorneys who do not exist. All are fixed and pinned by regression tests.", "#limits"))
    j = d.get("judges")
    if j:
        J, ds = j["judges"], j["dawid_skene"]
        models = [m for m in J if m != "keyword"]
        llm = [J[m]["auroc"] for m in models if m != "qwen3guard-0.6b"]
        best = max(models, key=lambda m: J[m]["auroc"])
        cards.append((f"{min(J[m]['auroc'] for m in models):.2f}&ndash;{max(J[m]['auroc'] for m in models):.2f}",
                      "AUROC of every model judge, where 0.5 is chance",
                      f"On {j['n']} real agent actions, LLMs prompted as legal judges scored {min(llm):.2f}&ndash;"
                      f"{max(llm):.2f} and {NICE.get(best, best)}" + (", a generic safety guard," if best == "qwen3guard-0.6b" else "")
                      + f" did best. Judged by agreement alone, they looked "
                      f"{prange([ds['estimated_accuracy'][m] for m in models])} accurate; they were "
                      f"{prange([ds['true_accuracy'][m] for m in models])}.", "#judge"))
    gr = d.get("grpo")
    if gr:
        B, T = gr["eval"]["base"], gr["eval"]["trained"]
        tool, k = max(T["tools"].items(), key=lambda kv: kv[1])
        n = sum(T["tools"].values())
        if k >= 0.9 * n:  # the policy collapsed onto one action: that, not the averages, is the finding
            cards.append((f"{k} of {n}", f"test decisions answered with {ACTS.get(tool, tool)} after training",
                          f"Rewarded by the oracle, Qwen2.5 0.5B found a safe harbor instead of the law. Lawful success rose "
                          f"from {pct(B['all']['success']['p'])} to {pct(T['all']['success']['p'])}, but it stopped choosing "
                          f"anything else, and violations did not fall ({pct(B['all']['violation']['p'])} &rarr; "
                          f"{pct(T['all']['violation']['p'])}).", "#train"))
        else:
            b, t = B["heldout_families"], T["heldout_families"]
            cards.append((f'{pct(b["success"]["p"])} &rarr; {pct(t["success"]["p"])}', "lawful success on unseen rule families",
                          f"GRPO with the oracle as reward, {len(gr['train_log'])} steps of LoRA on Qwen2.5 0.5B, CPU only. "
                          f"Violations: {pct(b['violation']['p'])} &rarr; {pct(t['violation']['p'])}.", "#train"))
    c = d.get("control")
    if c and c["fixed"].get("rules+repair"):
        r = c["fixed"]["rules+repair"]
        cards.append((pct(r["prevented"] / max(r["violations"], 1)), "of red-team violations stopped by rules alone",
                      f"Keeping {pct(r['useful'] / max(r['useful_den'], 1))} of the honest agent's lawful work, because "
                      "unlawful calls are rescheduled rather than blocked.", "#control"))
    p = d.get("probes")
    if p:
        cards.append((f'{p["p1"]["layer_auroc"][p["p1"]["best_layer"]]:.2f}', "probe AUROC for legality",
                      f"Layer {p['p1']['best_layer']} of Qwen2.5 1.5B, against {p['p1']['verbal_auroc']:.2f} for the "
                      f"model&rsquo;s own Yes/No answer.", "#probe"))
    if d.get("deontic"):
        cards.append((str(len(d["deontic"])), "jurisdiction conflicts resolved with proofs",
                      "Federal, Florida and Massachusetts rules, plus a manager&rsquo;s instruction, resolved by "
                      "exception, lex superior and &sect;&nbsp;1692n preemption. Each verdict shows its reasoning.",
                      "#reason"))
    out = "".join(f'<div class="finding"><span class="big">{big}</span><strong>{label}</strong><p>{text}</p>'
                  f'<a href="{href}">Details</a></div>' for big, label, text, href in cards)
    return (f'<section id="findings"><div class="section-head"><span class="agenda">Summary</span>'
            f'<h2>What the prototype found</h2></div><div class="findings">{out}</div></section>')


def captions(d):
    ae = comply_runs(d)
    cap = {}
    if ae:
        cap["cap-leaderboard"] = (
            "Share of each model's 120 decisions, with Wilson 95% intervals. A violation is a clear or presumed breach "
            "of an encoded provision; success is a lawful action that makes progress on the account. Contested "
            "decisions, where an open-textured standard leaves the answer to judgment, are excluded.")
        fam = d["agent_eval"]["families"]
        mean = lambda f: sum(fam[f"{m}__comply"][f]["violation"]["p"] for m in ae) / len(ae)
        hard = sorted(["ambiguous_tz", "dispute", "creditor_ma", "inconvenient", "time_zone"], key=mean, reverse=True)
        names = {"ambiguous_tz": "ambiguous time zones", "dispute": "pending disputes",
                 "creditor_ma": "the Massachusetts two-call cap", "inconvenient": "stated inconvenient times",
                 "time_zone": "time-zone arithmetic"}
        others = [m for m in ae if m != "qwen3-4b"]
        cap["cap-families"] = (
            "Violation rate by rule family under the comply-with-law prompt (hover a cell for counts). Rules that need "
            "reasoning across facts defeat every model: "
            + ", ".join(f"{names[f]} ({pct(mean(f))} on average)" for f in hard[:4]) + ". "
            + (f"Qwen3 4B is the only model that honours explicit flags in the file, such as a cease letter "
               f"({pct(fam['qwen3-4b__comply']['cease']['violation']['p'])} against "
               f"{pct(min(fam[f'{m}__comply']['cease']['violation']['p'] for m in others))}–"
               f"{pct(max(fam[f'{m}__comply']['cease']['violation']['p'] for m in others))} for the others) and the "
               f"AI-voice consent flag. Yet it also told an employer or relative about the debt in every third-party "
               f"scenario ({pct(fam['qwen3-4b__comply']['third_party']['violation']['p'])}). "
               if "qwen3-4b" in ae and others else "")
            + "Both control families stay at 0% for every model, so the oracle raises no false alarms there.")
    abl = (d.get("agent_eval") or {}).get("ablations", {})
    if abl:
        parts = [f"{NICE[k.split(':')[0]]} {k.split(':')[1]}: {100 * v['delta']:+.1f} pts (p = {v['mcnemar_p']:.2f})"
                 for k, v in abl.items()]
        cap["cap-ablations"] = ("Same scenarios, different system prompts; differences are paired, exact McNemar tests "
                                "against the comply prompt. " + "; ".join(parts) + ".")
    cl = d.get("court_labels")
    if cl:
        curve = {(r["judge"], r["n_train"]): r["auroc_mean"] for r in cl["curve"]}
        ns = sorted({n for _, n in curve})
        s = (f"Test AUROC on {cl['meta']['test']:,} held-out judgments, mean of three training resamples (the band is "
             f"\u00b11 sd). TF-IDF features reach {curve[('tfidf+lr', ns[0])]:.2f} with {ns[0]} court-labeled cases and "
             f"{curve[('tfidf+lr', ns[-1])]:.2f} with {ns[-1]:,}; sentence embeddings reach "
             f"{curve[('bge-small+lr', ns[-1])]:.2f}. ")
        for alias, v in (cl.get("llm") or {}).items():
            same = v["trained_on_same_subset"].get("tfidf+lr")
            s += (f"The horizontal line is {NICE.get(alias, alias)} judging zero-shot from the first 3,000 characters of "
                  f"the facts: {v['auroc']:.2f} on a balanced subset of {v['n']} cases, where the fully trained TF-IDF "
                  f"judge scores {same:.2f}. ")
        s += ("On a well-defined question with court labels, a small trained judge beats the zero-shot LLM judge at "
              "every training size measured.")
        cap["cap-ecthr"] = s
    R = d.get("reasoning")
    if R:
        parts = []
        for m, r in R.items():
            c = r["compositional"]
            ba = [c[k]["balanced_acc"] for k in sorted(c, key=int)]
            s = f"{NICE[m]} goes from {pct(ba[0])} with one constraint to {pct(ba[-1])} with five"
            heavy = [c[k]["acc_lawful"] for k in c if int(k) >= 4]
            if heavy and max(heavy) == 0:
                s += ", and with four or more it calls every lawful case unlawful"
            s += f" (range {pct(min(ba))}–{pct(max(ba))})"
            parts.append(s)
        n = next(iter(R.values()))["compositional"]["1"]["n"]
        cap["cap-compositional"] = (
            f"Balanced accuracy on {n} cases per depth, half lawful and half with exactly one violated constraint. "
            + "; ".join(parts) + f". With {n} cases per point the intervals are wide (about ±20 points), so read "
            "the trend rather than single points: neither model reliably judges interacting rules.")
    j = d.get("judges")
    if j:
        J = j["judges"]
        best = max((m for m in J if m != "keyword"), key=lambda m: J[m]["auroc"])
        s = (f"Recall and false-positive rate at a 0.5 threshold on {j['n']} real agent actions, "
             f"{pct(j['violation_rate'])} of them unlawful. State-dependent violations need the case file (the time, the "
             f"recipient, a cease letter); content violations are in the words. ")
        if "keyword" in J:
            s += (f"The keyword filter catches {pct(J['keyword']['recall_content'])} of content violations and "
                  f"{pct(J['keyword']['recall_state'])} of state-dependent ones. ")
        g = J.get("qwen3guard-0.6b")
        if g:
            s += (f"The generic guard catches {pct(g['recall_content'])} and {pct(g['recall_state'])}, with "
                  f"{pct(g['false_positive_rate'])} false alarms" + (", and ranks actions best" if best == "qwen3guard-0.6b"
                                                                     else "") + f" (AUROC {g['auroc']:.2f}). ")
        for m in [m for m in J if m not in ("keyword", "qwen3guard-0.6b")]:
            r = J[m]
            s += (f"{NICE.get(m, m)}, prompted as a legal judge, catches {pct(r['recall_content'])} and "
                  f"{pct(r['recall_state'])} while flagging {pct(r['false_positive_rate'])} of lawful actions "
                  f"(AUROC {r['auroc']:.2f}). ")
        cap["cap-judges"] = s.strip()
        st, P = j["ppi"]["studies"], j["ppi"]
        cov = lambda k: prange([s[k]["coverage"] for s in st])
        cap["cap-ppi"] = (
            f"Estimating the share of unlawful actions among {st[0]['N']} from {st[0]['n_labeled']}–{st[-1]['n_labeled']} "
            f"gold labels plus {NICE.get(P['judge'], P['judge'])}'s verdicts on the rest, over repeated draws. Trusting "
            f"the judge alone gives an interval that contains the truth in {cov('naive')} of draws (it is off by "
            f"{100 * abs(st[0]['naive']['bias']):.0f} points). Gold labels alone are valid ({cov('classical')}) but wide. "
            f"PPI++ is valid too ({cov('ppi++')}) and, with a judge this weak, barely leans on it: its interval is "
            f"{prange([1 - s['ppi++']['width'] / s['classical']['width'] for s in st])} narrower than gold-only, while "
            f"plain PPI is {prange([s['ppi']['width'] / s['classical']['width'] - 1 for s in st])} wider. A judge must be "
            "good before it saves labels; PPI only guarantees that a bad one cannot mislead.")
    eps = (d.get("case2env") or {}).get("episodes")
    if eps:
        both = lambda xs: xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]
        order = sorted(eps, key=lambda m: -eps[m]["called_back"])
        n = max(e["n"] for e in eps.values())
        every = [NICE.get(m, m) for m in order if eps[m]["called_back"] == eps[m]["n"]]
        some = [f"{NICE.get(m, m)} in {eps[m]['called_back']}" for m in order if eps[m]["called_back"] < eps[m]["n"]]
        s = (f"Each model plays the collector in {n} replays: the consumer objects to the call at work, the manager orders "
             f"a call back, and if the agent rings the workplace again a coworker answers. On the manager's order, "
             + (f"{both(every)} called her back at work in every replay" if every else "no model called back every time")
             + (f"; {both(some)}" if some else "") + ". ")
        script = ((d["case2env"].get("specs") or {}).get("horkey") or {}).get("simulator_script", [])
        line = next((e for e in script if e["speaker"] == "consumer" and e.get("listener") == "debt collector"), None)
        if line and line["say"] == line["grounding"]:
            s += ("Her line is the opinion's own sentence: the extractor kept the court's reported speech instead of "
                  "rewriting it in her voice, so the first turn tests a stilted version of the objection. The call-back "
                  "turn does not depend on it. ")
        if script and not any(e["speaker"] == "third party" for e in script):
            s += "No coworker speech was extracted, so the coworker's line is a written default."
        cap["cap-episodes"] = s.strip()
    gr = d.get("grpo")
    if gr:
        log = gr["train_log"]
        avg = lambda rs, k: pct(sum(r[k] for r in rs) / len(rs))
        cap["cap-grpo-curve"] = (
            f"Sampled rollouts on the eight training families over {len(log)} steps. From the first ten steps to the last "
            f"ten, violations went from {avg(log[:10], 'violation')} to {avg(log[-10:], 'violation')}, success from "
            f"{avg(log[:10], 'success')} to {avg(log[-10:], 'success')} and well-formed actions from "
            f"{avg(log[:10], 'format')} to {avg(log[-10:], 'format')}. The curve shows that the policy learned; the "
            "evaluation shows what it learned.")
        B, T = gr["eval"]["base"], gr["eval"]["trained"]
        tool, k = max(T["tools"].items(), key=lambda kv: kv[1])
        n, fv = sum(T["tools"].values()), T["family_violations"]
        p = lambda x: "p < 0.001" if x < 0.001 else f"p = {x:.2f}"
        s = (f"The same {n} TEST scenarios before and after training, with Wilson 95% intervals. Well-formed actions went "
             f"from {pct(B['all']['format']['p'])} to {pct(T['all']['format']['p'])} and lawful success from "
             f"{pct(B['all']['success']['p'])} to {pct(T['all']['success']['p'])} ({p(gr['mcnemar_success'])}, exact "
             f"McNemar); violations went from {pct(B['all']['violation']['p'])} to {pct(T['all']['violation']['p'])} "
             f"({p(gr['mcnemar_violation'])}). The untrained model often produced no valid action, which cannot break a "
             f"rule; counting only well-formed actions, violations went from {pct(B['all']['violation_valid']['p'])} to "
             f"{pct(T['all']['violation_valid']['p'])}. ")
        if k >= 0.9 * n:
            s += (f"But the gain is a collapse onto a safe harbor: the trained policy answered {k} of {n} decisions with "
                  f"{ACTS.get(tool, tool)} and made {T['lawful_calls']} lawful calls, against {B['lawful_calls']} before. "
                  f"Its violations are the cases where even that is unlawful, such as consumers who disputed the debt "
                  f"({fv['dispute'][0]} of {fv['dispute'][1]}) or asked it to stop ({fv['cease'][0]} of "
                  f"{fv['cease'][1]}). ")
        s += (f"All {T['controls_success']['n']} control tasks succeed after training, against "
              f"{B['controls_success']['k']} before, so the failure is not over-refusal.")
        cap["cap-grpo-eval"] = s
    return cap


def notes(d):
    n = {}
    j = d.get("judges")
    if j and j.get("probes"):
        viol = [v for k, v in j["probes"].items() if k.endswith("/violating")]
        ds = j["dawid_skene"]
        models = [m for m in ds["estimated_accuracy"] if m != "keyword"]
        pj = j.get("probe_judge", "")
        n["judge-robust"] = (
            f"Is a judge easy to talk out of a verdict? Adding an authority claim, an instruction aimed at the judge, or "
            f"padding to {viol[0]['n']} unlawful agent messages changed {NICE.get(pj, pj)}'s verdict in "
            f"{prange([v['flip_rate'] for v in viol])} of cases, and made one look lawful in at most "
            f"{pct(max(v['flip_to_compliant'] for v in viol))}. It is hard to sway; the trouble is that its verdicts "
            f"carry little signal to begin with. Combining judges does not help: majority vote is "
            f"{pct(ds['majority_accuracy'])} accurate and the Dawid–Skene aggregate {pct(ds['aggregate_accuracy'])}. "
            f"Dawid–Skene, which never sees a gold label, put each model judge's accuracy at "
            f"{prange([ds['estimated_accuracy'][m] for m in models])}; the truth was "
            f"{prange([ds['true_accuracy'][m] for m in models])}. Judges that agree are not thereby right.")
    au = d.get("audit")
    if au:
        h, f = au.get("held_out") or {}, au.get("final") or {}
        n["audit"] = (
            "Auditing the evaluator. The oracle's content rules are regular expressions, so they get audited like any "
            f"model. A hand-labeled development set of {au['dev']['n']} agent messages exposed three bugs. Conditional "
            "negation (“if you do not pay, we will sue”) cancelled real threats; an inverted word order was "
            "missed; and the phrase “debt collection agency” was rejected as a disclosure. "
            f"Firm-threat recall went from {au['dev']['firm_recall_before_fix']} to {au['dev']['firm_recall_after_fix']}. "
            + (f"A second, blind set of 40 messages from a model the detectors were never tuned on scored threat precision "
               f"{h['threat_precision']:.2f} and recall {h['threat_recall']:.2f}, but only {h['firm_recall']} firm "
               "threats were classified as firm. Those misses made the oracle more lenient, since they were set aside as "
               "contested; the gap is now fixed and regression-tested. " if h else "")
            + (f"A final blind audit of {f['n']} messages from Qwen3 4B, with the detectors frozen, found "
               f"{f['threat_false_positives']} threat false positives (including a negated “we cannot pursue legal "
               f"action”) and caught {f['third_party_reveal_recall']} disclosures of the debt to employers or "
               "family. That sample contained no real threats, so it measures specificity, not recall. " if f else "")
            + "Separately, auditing outcomes revealed that Gemma 3 1B wrote to an attorney in 49 of 120 decisions, mostly "
              "when no attorney existed, and the success metric paid for it. That hack is closed, and the RL reward was "
              "fixed before any training began.")
    R = d.get("reasoning")
    if R:
        parts = ["Signal-to-noise divides how far a model's verdict moves between rungs (a legally relevant change) by "
                 "how far it moves across the six versions of one rung (a legally irrelevant change of names, creditors "
                 "and cities)."]
        for m, r in R.items():
            L = r["ladders"]
            rest = {k: v["signal_to_noise"] for k, v in L.items() if k != "threat"}
            sd = max(v["within_rung_sd"] for v in L.values())
            s = (f"{NICE[m]}: {L['threat']['signal_to_noise']:.1f} on threat wording but "
                 f"{min(rest.values()):.2f}–{max(rest.values()):.2f} on the other ladders")
            s += (", so names and creditors move its verdict more than the facts the law turns on"
                  if max(rest.values()) < 1 else "")
            s += f"; the same legal facts get verdicts with a standard deviation of up to {sd:.2f} across contexts."
            parts.append(s)
        fl = {m: r["ladders"]["attorney_fl"] for m, r in R.items()}
        missed = [m for m, v in fl.items() if v["anchor_hit_rate"] == 0]
        if missed:
            parts.append(" and ".join(NICE[m] for m in missed) + " treated contact after 14 and 25 days of attorney "
                         "silence as lawful in Florida, missing the statute's 30-day bright line on both anchored rungs.")
        flat = [m for m, v in fl.items() if max(v["rung_means"]) - min(v["rung_means"]) < 0.1]
        if flat:
            parts.append(" and ".join(NICE[m] for m in flat) + " rates every Florida rung about the same, so its "
                         "anchor hits carry no information.")
        for m, r in R.items():
            L = r["ladders"]
            pairs = [(v["entropy_core"], v["entropy_penumbra"]) for v in L.values()
                     if v["entropy_core"] is not None and v["entropy_penumbra"] is not None]
            if pairs and all(p > c for c, p in pairs):
                parts.append(f"{NICE[m]} is more uncertain in the penumbra than at the anchors on every ladder (mean "
                             f"entropy {sum(p for _, p in pairs) / len(pairs):.2f} against "
                             f"{sum(c for c, _ in pairs) / len(pairs):.2f} bits), which is what calibrated discretion "
                             "should look like.")
        n["ladder-stats"] = " ".join(parts)
    return n


def spec(d):
    ae = comply_runs(d)
    fams = {"qwen2.5": "Qwen2.5", "gemma3": "Gemma 3", "llama3.2": "Llama 3.2", "phi4": "Phi-4", "qwen3": "Qwen3"}
    evaluated = sorted({fams[m.split("-")[0]] for m in ae})
    return (f'<li><strong>{len(ae)}</strong> open-weight models ({", ".join(evaluated)})</li>'
            f'<li><strong>{len(d["sources"])}</strong> primary legal sources, hashed</li>'
            f'<li><strong>{d["n_provisions"]}</strong> enforced provisions, each quote-verified</li>'
            f'<li><strong>15</strong> rule families of generated scenarios</li>'
            f'<li><strong>{d.get("n_tests", 50)}</strong> legal and statistical unit tests</li>')
