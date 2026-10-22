"""JUDGE (court-derived labels): how much labeled data from court decisions does a small trained judge need to
beat a zero-shot LLM judge?  Two tasks, both labeled by courts, not annotators' intuitions:

  * ECtHR-A (LexGLUE): facts of a European Court of Human Rights case -> did the Court find a violation of
    Article 3 (torture / inhuman or degrading treatment)?
  * LegalBench function_of_decision_section: paragraph of a U.S. opinion -> its function (Facts, Rule, ...).
    This validates the opinion segmenter used by case2env.

    python experiments/03_court_labels.py            # trained judges only (no LLM)
    python experiments/03_court_labels.py --llm qwen2.5-1.5b qwen3-4b
"""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from lawful.env.case2env import FUNCTIONS, classify_paragraph, legalbench
from lawful.llm.client import Server, pmap
from lawful.llm.registry import MODEL_DIR
from lawful.paths import DATA, RESULTS

OUT = RESULTS / "court_labels"
OUT.mkdir(exist_ok=True)
ART3 = 1  # LexGLUE ECtHR label order: 2, 3, 5, 6, 8, 9, 10, 11, 14, P1-1
ART3_TEXT = ("Article 3 of the European Convention on Human Rights: “No one shall be subjected to torture or to "
             "inhuman or degrading treatment or punishment.”")


def ecthr():
    def load(split):
        df = pd.read_parquet(DATA / "lexglue" / f"ecthr_a_{split}.parquet")
        df["facts"] = df.text.apply(lambda ps: "\n".join(ps))
        df["y"] = df.labels.apply(lambda ls: int(ART3 in ls))
        return df
    return load("train"), load("test")


def embed(texts, max_chars=2000):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(str(MODEL_DIR / "hf" / "bge-small-en-v1.5"), device="cpu")
    m.max_seq_length = 512
    return m.encode([t[:max_chars] for t in texts], batch_size=32, normalize_embeddings=True, show_progress_bar=False)


def learning_curve(train, test, sizes=(50, 100, 200, 500, 1000, 2000, 4000, 9000), seeds=(0, 1, 2)):
    rows = []
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True)
    Xtr_t, Xte_t = tfidf.fit_transform(train.facts), tfidf.transform(test.facts)
    Etr, Ete = embed(list(train.facts)), embed(list(test.facts))
    np.save(OUT / "ecthr_test_emb.npy", Ete)
    for n in sizes:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            pos, neg = np.where(train.y == 1)[0], np.where(train.y == 0)[0]
            k = min(len(pos), max(2, int(round(n * train.y.mean()))))
            idx = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, min(n - k, len(neg)), replace=False)])
            for name, Xtr, Xte, C in [("tfidf+lr", Xtr_t, Xte_t, 4.0), ("bge-small+lr", Etr, Ete, 1.0)]:
                clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr[idx], train.y.values[idx])
                p = clf.predict_proba(Xte)[:, 1]
                rows.append({"judge": name, "n_train": int(len(idx)), "seed": seed,
                             "auroc": roc_auc_score(test.y, p), "f1": f1_score(test.y, p > 0.5)})
                if n == sizes[-1] and seed == 0:
                    np.save(OUT / f"ecthr_test_p_{name}.npy", p)
        print(f"n={n}: " + ", ".join(f"{r['judge']} {r['auroc']:.3f}" for r in rows[-2:]), flush=True)
    return rows


def llm_ecthr(alias, test, n=80, max_chars=3000):
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(test.y == 1)[0], n // 2, replace=False),
                          rng.choice(np.where(test.y == 0)[0], n // 2, replace=False)])
    def ask(i):
        facts = test.facts.iloc[i][:max_chars]
        return llm.p_yes([{"role": "system", "content": "You are a judge of the European Court of Human Rights. "
                                                        "Answer with exactly one word: Yes or No."},
                          {"role": "user", "content": f"{ART3_TEXT}\n\nFacts of the case:\n{facts}\n\nOn these "
                                                      f"facts, did the respondent State violate Article 3? Yes or No."}])
    with Server(alias, parallel=2, ctx_per_slot=2048) as llm:
        p = pmap(ask, idx, workers=2, desc=f"{alias}/ecthr")
    return [{"idx": int(i), "y": int(test.y.iloc[i]), "p": float(pp)} for i, pp in zip(idx, p)]


def fods_trained():
    rows = legalbench("function_of_decision_section")
    X, y = [r["Paragraph"] for r in rows], [r["answer"] for r in rows]
    out = {}
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(X)
    for name, feats in [("tfidf+lr", tf), ("bge-small+lr", embed(X, 1500))]:
        pred = cross_val_predict(LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced"), feats, y, cv=cv)
        out[name] = {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"), "n": len(y)}
    return out


def fods_llm(alias, per_class=20):
    rows = legalbench("function_of_decision_section")
    rng = np.random.default_rng(0)
    pick = []
    for fn in FUNCTIONS:
        cand = [r for r in rows if r["answer"] == fn]
        pick += list(rng.choice(cand, min(per_class, len(cand)), replace=False))
    with Server(alias) as llm:
        pred = pmap(lambda r: classify_paragraph(llm, r["Paragraph"]), pick, desc=f"{alias}/fods")
    y = [r["answer"] for r in pick]
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"), "n": len(y),
            "pairs": list(zip(y, pred))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", nargs="*", default=[])
    args = ap.parse_args()
    train, test = ecthr()
    res_path = OUT / "results.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    if "ecthr_curve" not in res:
        res["ecthr_curve"] = learning_curve(train, test)
        res["fods_trained"] = fods_trained()
        res["ecthr_meta"] = {"train": len(train), "test": len(test), "train_pos": float(train.y.mean()),
                             "test_pos": float(test.y.mean())}
        res_path.write_text(json.dumps(res, indent=1))
    for alias in args.llm:
        res.setdefault("ecthr_llm", {})[alias] = llm_ecthr(alias, test)
        res.setdefault("fods_llm", {})[alias] = fods_llm(alias)
        res_path.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["fods_trained"], indent=1))


if __name__ == "__main__":
    main()
