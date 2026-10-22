# Talking points

## The 60-second version

I built a small, end-to-end version of your agenda (test agents against the law, train them to follow it,
control them at runtime) using only open-weight models on a laptop-class CPU. The domain is AI debt collection,
because it is a real deployment, it is governed by overlapping federal and state law, and it mixes bright-line
rules with open-textured standards. Everything hangs off one legal oracle. Every provision it enforces carries a
verbatim quote that a unit test checks against the primary source, and the CFPB's own worked examples are encoded
as unit tests. That one oracle produces eval labels, gold labels for validating LLM judges, the RL reward, and
probe targets. The result is a working loop and a set of measurements, with honest limits: court decision →
environment → agents → judges → training → runtime control.

## Design choices worth defending

* **Ground truth is selected, never generated.** Statutory quotes are verified verbatim; event extraction from
  opinions is span-verified; labels come from code that encodes cited text, and an LLM never labels anything.
* **Open texture is represented, not hidden.** Every finding is `clear`, `presumed` (rebuttable presumption) or
  `contested`. Metrics abstain on contested cases instead of picking a side. For open-textured standards I test
  *relations* that must hold (a fortiori monotonicity anchored on Horkey, the CFPB commentary and Brown;
  invariance to irrelevant facts; more uncertainty in the penumbra) instead of inventing labels.
* **Applicability and stacking are explicit.** FDCPA binds third-party collectors; Florida binds anyone collecting
  from a Floridian; Massachusetts' 940 CMR binds creditors; the TCPA binds every caller. 15 U.S.C. § 1692n tells you
  how they compose: the more protective rule controls.
* **The rule engine knows only the structured system of record.** Facts that live in free-text notes ("don't call
  before 11", "I can't talk at work") are invisible to it by design. That is the honest boundary between what you
  can compile into a rule and what needs a learned monitor.
* **The oracle itself is audited.** I hand-labeled agent messages and found real detector bugs (conditional negation,
  inverted word order, paraphrased threats), fixed them, and kept a separate held-out audit for an unbiased estimate.

## Questions they may ask

**"Isn't the oracle just your reading of the law?"** For bright lines it is close to mechanical, and it reproduces
the regulator's own worked examples. Where the law leaves discretion, the oracle abstains (`contested`) rather than
deciding. The next step is a lawyer-in-the-loop review that treats disagreement as data: track which provisions
have high inter-lawyer variance, and route those to court-anchored labels.

**"Why not use a frontier model as the judge?"** No API keys was a constraint, but the method doesn't depend on
it. The point of the judge section is that *any* judge needs validation: measure it against gold by violation type,
probe it adversarially, and when you report a population-level compliance rate, use prediction-powered inference
so a biased judge cannot bias the estimate.

**"How would this scale to thousands of decisions?"** CAP/CourtListener bulk → segmenter trained on court-labeled
paragraphs → span-verified fact extraction → citation linking → doctrine-level templates → scenario ladders that
perturb facts along the rule's elements, with anchors at decided points. Lawyers review anchors and ladders, not
every scenario.

**"What stops RL from gaming a legal reward?"** Three things. First, the task term: without it, refusing is optimal.
Second, credit shaping: without it, "always mail a letter" is a safe harbour. Third, held-out rule families plus an
over-refusal metric on control scenarios, so gaming shows up in evaluation.

**"Rules or learned monitors at runtime?"** Both, layered. Rules are cheap, exact and auditable but blind to
unstructured context. Monitors see context but need thresholds and validation. A human audit budget catches the
rest. Repairing (rescheduling a call to the first lawful minute) preserves usefulness that blocking would throw away.

## What I'd do in the first months

1. A court-grounded environment benchmark for one doctrine family end to end (FDCPA/Reg F → TCPA → FCRA), with
   lawyer-validated anchors and released ladders.
2. A judge-validation protocol: judge cards (AUROC by violation type, robustness probes, calibration), PPI-based
   reporting, court-derived labels for the open-textured standards.
3. RL with oracle + validated-judge rewards; measure the alignment tax and cross-doctrine generalization.
4. Pre-action white-box monitors (probes) as a cheap first layer of runtime control.
5. Multi-agent environments for collusion and deception, with liability-relevant monitors (agreement vs parallelism).
