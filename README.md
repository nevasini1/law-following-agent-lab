# Law-Following Agent Lab

**Can an AI agent be trusted to follow the law, even when its boss tells it not to?**

This repo is a working lab for that question. It **tests** AI agents against real laws, **trains** them to follow
those laws, and **guards** them while they run. It uses free, open-weight AI models on an ordinary 4-core CPU:
no API keys, no cloud GPUs.

**See the results:** [interactive report](https://claude.ai/artifact/E7vHf7nP8PvqYPQ4rzZUMq). A copy is saved here
as [`report/index.html`](report/index.html); download it and open it in any browser.

## The short version

We cast AI agents as debt collectors, working for a manager who sometimes asks for things the law forbids.
Here is what we found.

- **They broke the law often.** Five open-weight models from Alibaba, Google and Meta made 120 decisions each,
  and broke the law in 43% to 67% of them.
- **Most scenarios are traps, on purpose.** In 82% of them (not counting controls), doing exactly what the manager
  asked would have been illegal. The test is whether the agent notices.
- **Capable is not the same as lawful.** The strongest model, Qwen3 4B, always honoured letters asking it to stop
  contacting someone. But when told to pressure a person through their employer or a relative, it revealed the debt
  every single time, which the law forbids.
- **Replaying a real court case, most models repeated the collector's mistake.** In a simulation of *Horkey*, a woman
  tells a debt collector she can't talk at work, and his manager orders him to call her back there. Three of the five
  models did so in every run.
- **Reminding models of the law didn't help.** Telling Qwen2.5 1.5B to obey the law, handing it the relevant statute,
  or adding pressure moved its violation rate by 3 points at most, and none of the changes was statistically
  significant. Whatever we tried, it picked up the phone in at least 110 of 120 decisions.
- **Legal reasoning falls apart as rules pile up.** Asked whether a call was legal, the same model scored 71% when
  one rule mattered and 46%, worse than a coin flip, when five did. With four or more rules in play, it called every
  legal call illegal.
- **AI judges of legality were no better than a coin flip.** Two LLMs asked whether 200 real agent actions were legal
  scored 0.42 and 0.45, where 0.5 is guessing. A generic safety filter did best (0.63), mostly by catching threats and
  insults rather than violations that depend on the facts. Judged by how much they agreed with each other, the judges
  looked 76–80% accurate; they were really 42–57%.
- **Labels from real court decisions work.** A small model trained on just 50 European Court of Human Rights
  judgments predicted the court's verdicts much better than a general-purpose model asked directly: 0.90 versus 0.73,
  on a scale where 0.5 is guessing and 1.0 is perfect.
- **Training found a loophole instead of the law.** We rewarded a small model for lawful, useful actions. It learned
  to send an email in all 120 test cases, since email is legal in most situations and still earned partial credit.
  Its success rate rose from 24% to 67%, but it never made a phone call, and it kept emailing people who had disputed
  the debt or asked to be left alone.
- **Simple rules made the best guard.** A firewall that checks each action against the account's records stopped 76%
  of violations in 2 milliseconds, without blocking any legitimate work. An AI monitor on its own stopped 3% at that
  setting. Together, with a small human review budget, they stopped 84%.
- **We audited our own grader, too.** It had four flaws: three detection bugs, and one loophole where a model earned
  "success" by writing to lawyers who did not exist. All four are fixed and covered by tests.

## Why debt collection?

Companies are already deploying AI agents to collect debts. It is one of the most litigated corners of US consumer
law, so there are plenty of real cases to learn from. The rules mix crisp bright lines ("no calls before 8 a.m.")
with fuzzy standards ("a reasonable period of time", "reason to know"), and federal and state laws overlap and
sometimes conflict. That makes it a realistic, demanding test of whether an agent can *follow* the law, not just
recite it. A smaller module applies the same approach to AI pricing agents and price-fixing.

## How it works

![How the lab fits together: court opinions become scenarios, an agent acts in them through a firewall, and one legal oracle grades every action](docs/design.svg)

The diagram reads top to bottom. Court cases become test scenarios (top row). An AI agent acts inside those
scenarios, with a firewall checking each action (middle row). One **legal oracle** grades everything (bottom row).

The oracle is the heart of the project: a program that takes a situation and a proposed action, such as "call this
person at work at 7:45 a.m.", and says which laws, if any, the action would break, with exact citations.

1. **A rulebook built from the actual law.** We download 16 primary sources (federal and state statutes and
   regulations, an FCC ruling and seven court opinions) and fingerprint each one so any change is detectable. Each
   of the oracle's 32 rules quotes the law word for word, and a test checks every quote against its source. When the
   law is genuinely unclear, the oracle says "contested" instead of guessing, and those cases are left out of the
   scores.
2. **Scenarios from real court cases.** In *Horkey v. J.V.D.B.*, a collector kept pressing a woman at work after she
   told him she couldn't talk there. We turn cases like this into simulated, multi-turn situations. A generator then
   builds hundreds of variations, nudging the facts back and forth across the legal line (a call at 7:55 a.m.
   instead of 8:05 a.m., say).
3. **Agents act, the oracle grades.** Each model plays the debt collector and picks its next action, such as a call,
   an email or doing nothing. The oracle labels every action, and those same labels are reused everywhere: to score
   the models, to check AI judges, as the reward during training, and as targets when looking inside the models.

The firewall uses the same rulebook but sees only structured records (account fields and call logs), not free-text
notes. That is deliberate: it shows what a rules-based guard can and cannot catch.

## Results at a glance

How often each model broke the law, and how often it got the job done legally, over 120 decisions each:

| Model | Made by | Broke the law | Did the job legally |
|---|---|---|---|
| Gemma 3 1B | Google | 43% | 23% |
| Qwen3 4B | Alibaba | 53% | 41% |
| Llama 3.2 3B | Meta | 58% | 42% |
| Qwen2.5 1.5B | Alibaba | 63% | 37% |
| Qwen2.5 0.5B | Alibaba | 67% | 33% |

With 120 decisions per model, each number is accurate to roughly ±9 points. Gemma breaks the law least, but it also
gets little done: in 49 of its 120 decisions it wrote to the consumer's lawyer, and 41 of those consumers didn't have
one. It completed only 2 of the 16 control tasks, where acting was perfectly legal.

## What else is in the lab

| Part | What it does | Status |
|---|---|---|
| Conflict resolver | When federal law, state law and the manager's orders disagree, works out which rule wins and shows its reasoning | In the report |
| Reasoning stress tests | Checks whether a model's legal judgments stay consistent as the facts get stronger or weaker | In the report |
| Court-trained judges | Trains small "judge" models on labels taken from real court decisions | In the report |
| Training | Rewards a model for lawful, useful actions (reinforcement learning, on a CPU) | In the report |
| Judge validation | Measures how far an AI judge can be trusted, with honest error bars | In the report |
| Runtime firewall | Sits between the agent and the world, blocks illegal actions and fixes some (moving a 7:30 a.m. call to 8:00) | In the report |
| Probes | Looks inside a model to see whether it "knows" an action is illegal even as it takes it | Queued |
| Price-fixing test | Two AI pricing agents compete in a simulated market, watched for collusion | Queued |

The report gains each section as soon as its results land.

## Run it yourself

You need Python 3.11 or newer and about 18 GB of disk for the models. No API keys or accounts.

```bash
pip install -e .
./scripts/build_llama_cpp.sh          # build the local model server (llama.cpp)
python scripts/download_models.py     # download the open-weight models
python -m lawful.corpus.build         # download the legal sources
pytest -q                             # 50 tests: quotes match the law, legal rules, statistics
./experiments/run_agent_eval.sh       # the main test: every model, every scenario
python experiments/analyze.py && python experiments/build_report.py   # rebuild the report and the diagram
```

Each experiment has its own numbered script in `experiments/`. Every model response is cached, so re-running the
analysis is instant and every number in the report traces back to a raw model output.

## Honest limits

- **It covers 32 rules, not "the law".** Only Florida and Massachusetts state law is modeled; scenarios in other
  states are judged on federal law alone.
- **Some checks are approximate.** Rules about threats, profanity and required disclosures are checked by keyword
  detectors that understand negation ("we will *not* sue"). They are approximations, which is exactly why the
  judge-validation tools exist.
- **The models are small.** They have 0.5 to 4 billion parameters and run on a CPU, so the absolute numbers say
  little about frontier models. The point is the testing machinery, which works with any model.
- **Samples are small.** With 8 scenarios per rule type per model, every rate comes with a 95% confidence interval,
  and comparisons use exact statistical tests.

## What's where

| Folder | What's inside |
|---|---|
| `lawful/law/` | The rulebook, the legal oracle and the conflict resolver |
| `lawful/env/` | Scenario generator, court-case pipeline, multi-turn replay |
| `lawful/agents/` | Agent prompts and legal-text retrieval |
| `lawful/judge/`, `train/`, `control/`, `probe/`, `collude/` | Judges, training, firewall, probes, price-fixing |
| `lawful/corpus/` | Downloads and fingerprints the legal sources |
| `experiments/` | One script per experiment, plus the report builder |
| `results/` | Raw model outputs and summaries |
| `report/` | The results page |
| `docs/` | The diagram above, regenerated with the report |
| `tests/` | Legal, quote-verification and statistics tests |

<details>
<summary><b>Technical map, for researchers</b></summary>
<br>

| Agenda item | Module | What it does |
|---|---|---|
| Ground truth | `lawful/corpus`, `lawful/law/provisions.py`, `tests/test_provisions.py` | 16 primary sources with URL, sha256 and retrieval time; 32 enforced provisions, each quote-verified |
| | `lawful/law/oracle.py`, `tests/test_oracle.py` | FDCPA, Reg F, TCPA (+ FCC 24-17 on AI voices), Fla. Stat. 559.72, 940 CMR 7.04; applicability and stacking explicit; findings graded `clear`, `presumed` or `contested` |
| Test: court decisions to environments | `lawful/env/case2env.py`, `lawful/env/episode.py` | segment opinions with a classifier trained on court-labeled paragraphs; span-verified event extraction; Horkey replayed as a multi-turn environment |
| Test: scenario generation | `lawful/env/scenarios.py` | 15 families; legally relevant facts swept across decision boundaries, irrelevant facts randomized; labels from the oracle |
| Test: agents | `lawful/agents/`, `experiments/01_agent_eval.py` | 7 open-weight variants via llama.cpp with JSON-schema-constrained tool calls; prompt ablations incl. hybrid-retrieval RAG |
| Reason: conflicts and jurisdictions | `lawful/law/deontic.py` | defeasible deontic engine: exception, lex superior, § 1692n preemption, lex specialis, lex posterior, with proof traces |
| Reason: composition and open texture | `lawful/env/probes.py`, `experiments/02_reasoning.py` | accuracy vs number of interacting constraints; a-fortiori ladders anchored on Horkey, CFPB commentary and Brown; monotonicity, invariance, penumbra entropy |
| Judge: validating model judges | `lawful/judge/`, `experiments/06_judges.py` | LLM judges vs a generic safety guard (Qwen3Guard) vs keywords on real agent actions; injection, authority and verbosity probes; Dawid-Skene; prediction-powered inference |
| Judge: labels from court decisions | `experiments/03_court_labels.py` | ECtHR Article 3 (LexGLUE): learning curve of small trained judges vs a zero-shot LLM; LegalBench function-of-decision-section for the opinion segmenter |
| Train | `lawful/train/grpo.py`, `experiments/04_train_grpo.py` | minimal GRPO + LoRA on CPU with the oracle as reward; generalization to rule families never trained on; over-refusal audit |
| Control | `lawful/control/firewall.py`, `experiments/08_control.py` | rules over the system of record (with lawful rescheduling) + trusted-model monitor + audit budget; red-team policy; safety/usefulness frontier |
| Probe | `lawful/probe/`, `experiments/07_probes.py` | linear probes on residual streams; latent legality vs verbal judgment; instruction-risk probe vs what the agent actually did |
| Collude | `lawful/collude/market.py`, `experiments/09_collusion.py` | Calvano et al. logit Bertrand duopoly with LLM pricing agents and a chat channel; agreement monitor vs tacit parallelism |

Every LLM request is cached in `results/llm_cache.sqlite`, keyed by model and payload. Models and the llama.cpp
server default to `./models` and `./tools`; override with `LAWFUL_MODELS` and `LAWFUL_LLAMA_SERVER`.

</details>

Notes on the design choices, likely questions and next steps are in [`TALKING_POINTS.md`](TALKING_POINTS.md).
