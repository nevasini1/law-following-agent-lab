"""TEST (case2env): court opinion -> EnvSpec -> interactive episodes.

Segmentation uses a classifier trained on LegalBench's court-labeled paragraphs (validated by 5-fold CV in
03_court_labels.py); event extraction uses an LLM with span verification; episodes replay Horkey.

    python experiments/05_case2env.py --extractor qwen3-4b --agents qwen2.5-0.5b ... qwen3-4b
"""
import argparse
import json
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from lawful.corpus.sources import OPINIONS
from lawful.env.case2env import build_env_spec, extract_events, legalbench, paragraphs
from lawful.env.episode import run_episode
from lawful.llm.client import Server
from lawful.paths import CORPUS, RESULTS

OUT = RESULTS / "case2env"
OUT.mkdir(exist_ok=True)


def segmenter():
    rows = legalbench("function_of_decision_section")
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                        LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced"))
    return clf.fit([r["Paragraph"] for r in rows], [r["answer"] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", default="qwen3-4b")
    ap.add_argument("--agents", nargs="+", default=[])
    ap.add_argument("--samples", type=int, default=4)
    args = ap.parse_args()
    seg = segmenter()
    specs = {}
    spec_path = OUT / "envspecs.json"
    if spec_path.exists():
        specs = json.loads(spec_path.read_text())
    else:
        with Server(args.extractor, parallel=1, ctx_per_slot=8192) as llm:
            for key, _cite in OPINIONS:
                case = json.loads((CORPUS / "raw" / f"case_{key}.json").read_text())
                text = (CORPUS / "text" / f"case_{key}.txt").read_text()
                paras = paragraphs(text)
                labels = list(seg.predict(paras))
                facts = "\n".join(p for p, l in zip(paras, labels) if l == "Facts")
                events = extract_events(llm, facts) if facts else []
                specs[key] = build_env_spec(case, labels, paras, events) | {
                    "segments": dict(Counter(labels)), "events": events}
                print(f"{key}: {len(paras)} paragraphs, {specs[key]['n_verified']}/{specs[key]['n_events']} "
                      f"events verified, provisions={specs[key]['provisions']}", flush=True)
        spec_path.write_text(json.dumps(specs, indent=1))

    for alias in args.agents:
        with Server(alias) as llm:
            eps = [run_episode(llm, specs.get("horkey"), 0.0, 0)]
            eps += [run_episode(llm, specs.get("horkey"), 0.8, seed) for seed in range(1, args.samples + 1)]
        with open(OUT / f"episodes_{alias}.jsonl", "w") as f:
            for e in eps:
                f.write(json.dumps(e) + "\n")
        print(alias, "violations:", [e["any_violation"] for e in eps], flush=True)


if __name__ == "__main__":
    main()
