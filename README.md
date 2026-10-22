# lawful

**Test, train and control law-following agents, end to end, with open-weight models on a 4-core CPU and no API keys.**

`lawful` is a working prototype of the three-part agenda for law-following AI: *testing agents against the law,
training them to follow it, and controlling them at runtime*. The domain is AI agents doing consumer debt
collection, which is heavily litigated, deploying fast, and governed by overlapping federal and state law that
mixes bright-line rules ("before 8:00 a.m.") with open-textured standards ("reasonable period", "reason to know",
"least sophisticated consumer"). A Sherman Act module covers multi-agent collusion.

**Report:** [interactive results page](https://claude.ai/artifact/E7vHf7nP8PvqYPQ4rzZUMq) (also in this repo as
`report/index.html`, a single self-contained file built by `experiments/build_report.py`; open it in any browser).
Talking points for presenting the work are in `TALKING_POINTS.md`.

## Design in one paragraph

Everything hangs off a **legal oracle** (`lawful/law/oracle.py`) that maps (world state, proposed tool call) to
cited findings. Every provision it enforces carries a verbatim quote that a unit test checks against the
downloaded primary source (GovInfo, eCFR, state legislatures, the FCC, the Caselaw Access Project). Legal unit tests
reproduce the CFPB's own worked examples from the Regulation F commentary. Findings carry a certainty (`clear`,
`presumed` for rebuttable presumptions like Reg F's 7-in-7, or `contested` for open-textured standards no
authority resolves); metrics count the first two and set contested cases aside instead of guessing. The same
oracle produces eval labels, judge-validation gold, the RL reward, and probe labels. The runtime firewall reuses it,
restricted to the structured system of record.

## Modules

| Agenda item | Module | What it does |
|---|---|---|
| Ground truth | `lawful/corpus`, `lawful/law/provisions.py`, `tests/test_provisions.py` | 15 primary sources with URL + sha256 + retrieval time; 32 enforced provisions, each quote-verified |
| | `lawful/law/oracle.py`, `tests/test_oracle.py` | FDCPA, Reg F, TCPA (+ FCC 24-17 on AI voices), Fla. Stat. 559.72, 940 CMR 7.04; applicability and stacking explicit |
| Test: court decisions to environments | `lawful/env/case2env.py`, `lawful/env/episode.py` | segment opinions with a classifier trained on court-labeled paragraphs; span-verified event extraction; Horkey replayed as a multi-turn environment |
| Test: scenario generation | `lawful/env/scenarios.py` | 15 families; legally relevant facts swept across decision boundaries, irrelevant facts randomized; labels from the oracle |
| Test: agents | `lawful/agents/`, `experiments/01_agent_eval.py` | 7 open-weight variants (Qwen2.5 0.5B/1.5B, Gemma 3 1B/4B, Llama 3.2 3B, Phi-4-mini, Qwen3 4B) via llama.cpp, JSON-schema-constrained tool calls; prompt ablations incl. hybrid-retrieval RAG |
| Reason: conflicts & jurisdictions | `lawful/law/deontic.py` | defeasible deontic engine: exception, lex superior, § 1692n preemption, lex specialis, lex posterior, with proof traces |
| Reason: composition & open texture | `lawful/env/probes.py`, `experiments/02_reasoning.py` | accuracy vs number of interacting constraints; a-fortiori ladders anchored on Horkey / CFPB commentary / Brown; monotonicity, invariance, penumbra entropy |
| Judge: validating model judges | `lawful/judge/`, `experiments/06_judges.py` | LLM judges vs a generic safety guard (Qwen3Guard) vs keywords on real agent actions; injection/authority/verbosity probes; Dawid-Skene; prediction-powered inference |
| Judge: labels from court decisions | `experiments/03_court_labels.py` | ECtHR Article 3 (LexGLUE): learning curve of small trained judges vs zero-shot LLM; LegalBench function-of-decision-section for the opinion segmenter |
| Train | `lawful/train/grpo.py`, `experiments/04_train_grpo.py` | minimal GRPO + LoRA on CPU with the oracle as reward; generalization to rule families never trained on; over-refusal audit |
| Control | `lawful/control/firewall.py`, `experiments/08_control.py` | rules over the system of record (with lawful rescheduling) + trusted-model monitor + audit budget; red-team policy; safety/usefulness frontier |
| Probe | `lawful/probe/`, `experiments/07_probes.py` | linear probes on residual streams; latent legality vs verbal judgment; instruction-risk probe vs what the agent actually did |
| Collude | `lawful/collude/market.py`, `experiments/09_collusion.py` | Calvano et al. logit Bertrand duopoly with LLM pricing agents and a chat channel; agreement monitor vs tacit parallelism |

## Reproduce

```bash
pip install -e .                      # Python 3.11+
./scripts/build_llama_cpp.sh          # llama-server into ./tools (or set LAWFUL_LLAMA_SERVER)
python scripts/download_models.py     # open-weight models into ./models (or set LAWFUL_MODELS); no tokens needed
python -m lawful.corpus.build         # fetch primary sources, write data/corpus/manifest.json
pytest -q                             # quote verification + legal unit tests + estimator tests
./experiments/run_agent_eval.sh       # TEST sweep
python experiments/02_reasoning.py --models qwen2.5-1.5b qwen3-4b
python experiments/03_court_labels.py --llm qwen2.5-1.5b
python experiments/04_train_grpo.py --steps 40
python experiments/05_case2env.py --agents qwen2.5-1.5b qwen3-4b
python experiments/06_judges.py
python experiments/07_probes.py
python experiments/08_control.py
python experiments/09_collusion.py
python experiments/analyze.py && python experiments/build_report.py
```

Every LLM request is cached in `results/llm_cache.sqlite`, keyed by model and payload, so re-running any analysis
is free and every number in the report can be regenerated from raw outputs.

## Scope and honesty

* The oracle encodes 32 provisions, not "the law". States other than Florida and Massachusetts are evaluated on
  federal law only, and third-party collectors in Massachusetts (209 CMR 18) are out of scope.
* Content rules (threats, profanity, disclosures) use transparent lexical detectors with negation handling. They
  are a known approximation, reported as such, and they are the reason judge validation matters.
* Models are 0.5B-4B parameters, 4-bit or 8-bit quantized, on CPU. Absolute rates say little about frontier models;
  the harness, metrics and failure taxonomy are the deliverable, and the same interfaces take any model.
* Sample sizes are small (8 scenarios per family per model), so every rate is reported with a Wilson 95% interval and
  paired comparisons use exact McNemar tests.
