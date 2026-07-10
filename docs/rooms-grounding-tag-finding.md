# Finding: the room grounding tag is a no-op for synthesis

**Date:** 2026-07-10 · **Reproduce:** `python .scripts/ab_grounding_tag.py`

## Question
Rooms tag each message `[cited:N]` (N citation-shaped tokens) or `[asserted: no
citations]` and tell the synthesizer to "weight accordingly." Does that tag
actually change the synthesizer's output, or are we hardening a no-op? (Raised as
the decision-bet blind spot by the tier2 self-audit fusion room: everyone argued
about whether the tag is *sound*; nobody tested whether it *does anything*.)

## Method
Controlled A/B, all confounds removed after a first attempt was invalid:
- **Synthetic, uncheckable transcript** — a fictional treaty-date fork. Cited side
  claims 1847 (with citation-shaped refs), asserted side claims 1852, neither
  externally verifiable, so the tag is the only distinguishing signal.
- **Tool-free** (`--allowedTools ""`) and run from an empty temp dir, so the model
  cannot read the repo or fetch anything (v1 failed here: `claude -p` re-verified
  against live code and cited a commit newer than the transcript).
- **Variance baseline** — 3 samples/condition; compare within-condition similarity
  (opus noise floor) vs cross-condition (tag vs stripped). Real effect only if
  cross is materially below the floor. Invalid/errored runs are retried, not
  averaged in (v1's noise floor was faked to 0.08 by two dead runs).

## Result (6/6 valid runs)
```
within-condition similarity (noise floor): 0.151
cross-condition  similarity (tag effect):  0.170   # no lower than noise
Directional: all 3 tagged AND all 3 stripped runs endorse 1847 (the cited side)
=> NO-OP
```
The tag does not move synthesis output. The *stripped* runs reach the same
conclusion, for the same reason: the model reads the citation-shaped references
out of the message text directly (`"carries three citation-shaped references
(archive scan, DOI, arXiv)"`) and is already appropriately skeptical
(`"citation count is not evidence of correctness… none verified here"`).

## Implications
- The **grounding-tag line of work is cosmetic** for model behavior. Keep the
  honest labels (`cited` not `grounded`) for *humans* reading transcripts, but do
  not invest in tag/verification-pipeline machinery expecting it to steer the judge.
- The frontier judge **does not over-trust citations** — the premise motivating a
  claim/evidence-graph rewrite (Tier-2) is largely absent. Deprioritize it.
- What actually matters is **control flow**, not tags: convergence correctness
  (the `[UNVERIFIED]`→convergence sign-inversion, `or`→`and`), not persisting
  failed syntheses to memory, and long-run reliability (background execution).
  Those change what the system *does*; the tag only changes what a label *says*.

## Caveat
The noise floor is high (~0.15), so this rules out a *large* tag effect, not a
subtle one. But a signal that cannot clear opus's own run-to-run variance cannot
be reliably steering anything either.
