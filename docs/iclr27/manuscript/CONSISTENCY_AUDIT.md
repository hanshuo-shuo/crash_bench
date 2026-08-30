# Glass-scoped paper-package consistency audit

**Audit date:** 2026-08-30

**Terminal status:** `PAPER_PACKAGE_READY_GLASS_SCOPED`

**Scope:** one frozen OpenVLA--LIBERO `glass_recovery` exact-state design;
existing evidence only

**Excluded actions:** no Quest access, no simulator rollout, no new outcome,
no model fit, no router rescue, no threshold change, no sequential rescue, and
no claim expansion

The status above is valid only together with a clean committed branch and a
passing `audit_paper_package.py --require-clean` run. This file is included in
that commit; the command is rerun after the commit so the condition is checked
rather than assumed.

## Definition-of-done audit

| Requirement | Evidence | Result |
|---|---|---|
| Complete manuscript, no unfinished sections | `manuscript.tex` plus `appendix.tex`; unfinished-marker scan | pass |
| Required abstract facts | 19/273; three pairs/two sources; 0.2544, 0.5642, 0.3097; +0.0178 null; 0.2311; 23 and 51/60 glass; 24/24, 0/8, 2/2 | pass |
| Privileged/exposed status visible | abstract says privileged Detour and all 20 sources exposed development | pass |
| Main figures and tables complete | 3 reviewed figures, 3 main tables, 8 appendix tables; inventory records effective source counts | pass |
| Source is the independent unit | all captions report numeric source counts; repeated units are labeled | pass |
| One family, three conditions | captions and prose distinguish `glass_recovery` from `glass`/`offpath`/`noglass` | pass |
| Learned-method null retained | OutcomeRouter +0.0178 with no pass; full tiny ADR 0.2311 below reference | pass |
| Sequential null retained | Direct selects Base 24/24, recovers 0/8, misses 2/2 | pass |
| Non-glass v1 excluded | zero outcome rows; excluded from all scientific denominators | pass |
| Claim expansion prohibited | affirmative-claim scan over submission-facing TeX/Markdown | pass |
| Artifact map complete | 8 headline keys, canonical selectors, SHA-256, generation/record commits, availability, caveats | pass |
| Reproducibility contracts documented | checkpoint, source, split, utility, options, restoration, RNG limit, raw availability | pass |
| Limitations and ethics complete | main text plus standalone statement | pass |
| Venue-neutral cover summary complete | `COVER_SUMMARY.md` | pass |

## Evidence integrity

The paper-package audit checks the exact 12-item PIVOT required-artifact set and
every artifact SHA-256 promised by the PIVOT manifest. The audited workspace has
zero missing or mismatched PIVOT outputs. The publication artifact map also
verifies each canonical headline artifact against current bytes.

The top-level ICLR manifest is intentionally not rewritten to include the later
PIVOT package. The additive artifact map records that gap. The PIVOT manifest
is self-unpinned and has current byte hash
`dbb3df3b3c94b26cd0b923a864b47ea25da36a279f3f4d36b0272ccf93110bd7`.
Raw capture files are `local_untracked_hash_pinned`; this is not described as
repository-tracked availability.

## Claim audit

The scan covers only submission-facing `.tex` and `.md` files. It does not scan
historical frozen plans whose purpose is to preserve earlier wording. Explicit
negations and the appendix's `Not established` column are recognized as limits,
not affirmative claims.

The scan enforces these explicit limits:

- The manuscript does not claim a successful, superior, deployable, or
  production-ready router.
- The manuscript does not claim a broad VLA-safety benchmark.
- The manuscript does not claim generality across tasks, hazards, backbones,
  mechanisms, architectures, or VLAs.
- The manuscript does not claim reliable sequential intervention or recovery.
- The manuscript does not claim end-to-end learned recovery.
- The manuscript does not claim multiple independent mechanical families.
- The manuscript does not use non-glass v1 as scientific evidence.
- The manuscript does not claim first-method novelty already occupied by nearby
  work.

The final package contains none of those affirmative claims. It explicitly
retains the two-source tight-witness limit, one-source Detour/Retreat limit,
single-family condition semantics, privileged Detour, exposed-development
status, and negative sequential closeout.

## Restoration audit boundary

The manuscript accurately reports the five verified branch-start identities:
simulator state, controller/dynamics state, complete numeric continuation,
policy-boundary observation, and model XML. It does not silently promote fixed
seed and deterministic decoding into a per-decision global RNG-byte audit.
This resolves the only material wording gap found between the publication
resolution's shorthand and the tracked implementation contract.

## Build and visual audit

- Build command: `make -C docs/iclr27/manuscript pdf`.
- Final output: 14-page letter PDF, PDF 1.5, unencrypted, no JavaScript, no
  suspected structural issue according to `pdfinfo`.
- Final PDF SHA-256:
  `f2e0191424c074192d309c6fafe01c4440a469192d8c3d23749e9a97f5ebe2c0`.
- LaTeX log: no undefined citations/references, no multiply defined labels, no
  overfull boxes, no oversized floats, and no narrow-`tabularx` warning.
- Visual review: all 14 rendered pages inspected at 110 DPI; no blank page,
  clipping, overlap, broken glyph, unreadable main figure/table, or inconsistent
  page number was found.

## Executed checks

| Check | Result |
|---|---|
| `python scripts/iclr27/audit_paper_package.py` | pass |
| `PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q tests/iclr27` | 81 passed |
| `PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q` | 278 passed |
| `PYTHONDONTWRITEBYTECODE=1 python scripts/audit_repo.py` | pass |
| `git diff --check` | pass |

An earlier pytest invocation omitted `PYTHONPATH=.` and failed during test
collection because local `crashbench` and `scripts` modules were not importable.
No test body ran in that invocation. Repeating the exact suites with the
repository root on `PYTHONPATH` produced the passing results above.

## Final interpretation

`PAPER_PACKAGE_READY_GLASS_SCOPED` is a publication-package status, not a
scientific broadening. It certifies that the scoped manuscript, appendix,
figures, tables, provenance map, reproducibility statement, limitations,
cover summary, tests, and claim audit agree with the existing frozen evidence.
It does not certify a deployable router, a broad benchmark, cross-mechanism
generality, reliable non-glass Retreat, or reliable sequential intervention.
