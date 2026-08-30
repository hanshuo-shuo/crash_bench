# Glass-scoped paper package

This directory is the submission-facing package for:

> **Risk Does Not Specify Intervention: Exact-State Diagnostics for an
> OpenVLA--LIBERO Safety Case Study**

The package is deliberately scoped to the frozen `glass_recovery` design. It
uses existing reviewed evidence only. It does not authorize or contain a Quest
run, a new outcome, a new model fit, a router rescue, or a broader benchmark
claim.

## Contents

- `manuscript.tex`: complete venue-neutral manuscript;
- `appendix.tex`: appendix included by the manuscript;
- `references.bib`: references checked against primary paper pages;
- `ARTIFACT_MAP.md` and `artifact_map.csv`: claim-to-artifact provenance;
- `FIGURE_TABLE_INVENTORY.md`: display inventory and source-count contract;
- `REPRODUCIBILITY_STATEMENT.md`: checkpoint, split, utility, option, restore,
  availability, and build contracts;
- `LIMITATIONS_AND_ETHICS.md`: submission-facing limitation and safety scope;
- `CONSISTENCY_AUDIT.md`: adversarial claim and package audit record;
- `COVER_SUMMARY.md`: venue-neutral cover summary;
- `Makefile`: deterministic local build and audit entry points.

The three main data figures are immutable reviewed PDFs in
`results/iclr27/risk_value_decoupling_dc48ff317dad_20260829T154351Z/`.
The manuscript links those files directly instead of rewriting frozen result
artifacts.

## Build and audit

From the repository root:

```bash
make -C docs/iclr27/manuscript pdf
python scripts/iclr27/audit_paper_package.py
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q tests/iclr27
python scripts/audit_repo.py
git diff --check
```

The final PDF is written to
`output/pdf/risk_does_not_specify_intervention_glass_scoped.pdf`. Build
intermediates stay under `tmp/pdfs/manuscript/`.

## Status semantics

`PAPER_PACKAGE_READY_GLASS_SCOPED` means that the manuscript and appendix are
complete, every displayed number maps to frozen evidence, the package audit and
repository audit pass, the rendered PDF has been visually inspected, and all
files are committed on a clean branch. It does not mean that a deployable
router, broad VLA-safety benchmark, cross-mechanism result, or reliable
sequential intervention has been established.
