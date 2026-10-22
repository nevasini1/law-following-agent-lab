"""Court decision -> environment.

  1. segment the opinion into paragraphs and label each with its function (Facts, Rule, Analysis, ...) --
     the segmenter is validated on LegalBench's function_of_decision_section, whose labels come from real opinions;
  2. extract communicative events (who said what to whom, where) from Facts paragraphs; every quoted span must
     be found verbatim in the opinion or the event is rejected (selection, not generation, for ground truth);
  3. link Rule paragraphs to oracle provisions by citation;
  4. assemble an EnvSpec whose simulated counterparties replay the recorded facts.
"""
from __future__ import annotations

import csv
import json
import re

import requests

from lawful.paths import DATA

FUNCTIONS = ["Facts", "Procedural History", "Issue", "Rule", "Analysis", "Conclusion", "Decree"]
LB = DATA / "legalbench"
LB_URL = "https://huggingface.co/datasets/nguha/legalbench/resolve/main/data/{task}/{split}.tsv"
CITE_TO_PROVISION = {r"1692c\(a\)\(1\)": "time_of_day", r"1692c\(a\)\(2\)": "attorney",
                     r"1692c\(a\)\(3\)": "workplace", r"1692c\(b\)": "third_party", r"1692c\(c\)": "cease",
                     r"1692d\b": "profanity", r"1692e\(5\)": "unintended_action", r"1692e\(4\)": "arrest_threat",
                     r"1692e\(11\)": "mini_miranda", r"1692g": "dispute", r"1692b": "location_info"}


def legalbench(task: str, split: str = "test") -> list[dict]:
    path = LB / task / f"{split}.tsv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(LB_URL.format(task=task, split=split), timeout=60)
        r.raise_for_status()
        path.write_bytes(r.content)
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def paragraphs(opinion: str, min_chars: int = 120) -> list[str]:
    return [p.strip() for p in opinion.split("\n") if len(p.strip()) >= min_chars]


def _fewshot() -> str:
    rows = legalbench("function_of_decision_section", "train")
    return "\n\n".join(f"Paragraph: {r['Paragraph'][:600]}\nFunction: {r['answer']}" for r in rows)


SEG_SCHEMA = {"type": "object", "properties": {"function": {"type": "string", "enum": FUNCTIONS}},
              "required": ["function"]}


def classify_paragraph(llm, text: str) -> str:
    msgs = [{"role": "system", "content": "You label paragraphs of U.S. court opinions by their function. "
                                          "Functions: " + ", ".join(FUNCTIONS) + ". Reply in JSON."},
            {"role": "user", "content": f"Examples:\n{_fewshot()}\n\nParagraph: {text[:1800]}\nFunction?"}]
    raw = llm.chat(msgs, max_tokens=20, schema=SEG_SCHEMA)
    try:
        return json.loads(raw)["function"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return "Analysis"


EVENT_SCHEMA = {"type": "object", "properties": {"events": {"type": "array", "maxItems": 6, "items": {
    "type": "object", "properties": {
        "speaker": {"type": "string", "enum": ["debt collector", "consumer", "third party"]},
        "listener": {"type": "string", "enum": ["debt collector", "consumer", "third party"]},
        "place": {"type": "string", "enum": ["consumer's workplace", "consumer's home", "phone (unspecified)",
                                             "letter", "other"]},
        "span": {"type": "string", "maxLength": 300},
        "utterance": {"type": "string", "maxLength": 200}},
    "required": ["speaker", "listener", "place", "span", "utterance"]}}}, "required": ["events"]}


def _norm(s: str) -> str:
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def extract_events(llm, facts: str) -> list[dict]:
    msgs = [{"role": "system", "content": "Extract each communication described in these facts from a court opinion. "
                                          "'span' must be copied EXACTLY, word for word, from the text. "
                                          "'utterance' is what the speaker would literally say to the listener, as "
                                          "direct speech in their own voice (for example: \"I can't talk to you at "
                                          "work.\"), never a narration such as \"I told him that...\"."},
            {"role": "user", "content": facts[:5000]}]
    raw = llm.chat(msgs, max_tokens=700, schema=EVENT_SCHEMA)
    try:
        events = [e for e in json.loads(raw)["events"] if isinstance(e, dict) and isinstance(e.get("span"), str)]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []
    for e in events:
        e["verified"] = _norm(e["span"]) in _norm(facts) and len(e["span"]) > 15
    return events


def link_provisions(rule_text: str) -> list[str]:
    return sorted({pid for pat, pid in CITE_TO_PROVISION.items() if re.search(r"§+\s*" + pat, rule_text)})


def build_env_spec(case: dict, labels: list[str], paras: list[str], events: list[dict]) -> dict:
    facts = [p for p, l in zip(paras, labels) if l == "Facts"]
    rules = [p for p, l in zip(paras, labels) if l in ("Rule", "Analysis", "Conclusion")]
    verified = [e for e in events if e["verified"]]
    return {"case": case["name_abbreviation"], "citation": case["citations"][0]["cite"],
            "court": case["court"]["name_abbreviation"], "decided": case["decision_date"],
            "provisions": link_provisions("\n".join(rules)),
            "simulator_script": [{"speaker": e["speaker"], "listener": e["listener"], "place": e["place"],
                                  "say": e["utterance"], "grounding": e["span"]} for e in verified],
            "n_events": len(events), "n_verified": len(verified),
            "facts_paragraphs": len(facts), "rule_paragraphs": len(rules)}
