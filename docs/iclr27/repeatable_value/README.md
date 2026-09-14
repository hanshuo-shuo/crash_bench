# Repeatable intervention value manuscript

Current manuscript: [manuscript.md](manuscript.md). Chinese story: [STORY_ZH.md](STORY_ZH.md).
Readable PDF: [research draft](../../../output/pdf/crashbench_repeatable_value_draft.pdf).

This is the paper-development synthesis begun on 2026-09-13. It preserves the
historical glass-only manuscript in the adjacent manuscript/ directory.

## Result structure

1. Paired branching makes observed rescue measurable.
2. Same-policy pseudo-options reveal what outcome reuse can manufacture.
3. Repeat-disjoint evaluation and complete deadline curves distinguish different gains.
4. Frozen forecasts are tested in a new C execution block.
5. Fresh-reset source transport tests whether old opportunities reappear.
6. Retained recovery opportunities and execution-time learned selection remain distinct.

## Evidence entry points

- [Existing-record contract](../../audits/20260913/repeat_value/PLAN.md)
- [Existing-record tables and figures](../../audits/20260913/repeat_value/evidence/RESULTS.md)
- [Prospective C contract](../../audits/20260913/repeat_value/C_PLAN.md)
- [C machine freeze](../../audits/20260913/repeat_value/c_contract.json)
- [Quest provenance](../../audits/20260913/repeat_value/RUN.md)
- [Completed fresh-source results](../../audits/20260913/fresh_value/RESULTS_ZH.md)
- [Fresh-source plan](../../audits/20260913/fresh_value/PLAN.md)
- [External recovery asset assessment](../../audits/20260913/external_recovery/ASSESSMENT_ZH.md)
- [Claim-to-row mapping](EVIDENCE_MAP.md)

The completed C experiment has 332 branches on saved existing states. It measures subsequent
execution, not new-source generalization. A/B forecasts and all method choices were
published before either C job began. No model or recovery controller is refitted. The subsequent twelve-source fresh-reset
study completed384 branches: primary440 gain1.39→0→0pp. All36 C cells have
nonpositive recorded Detour-minus-Base success difference. These unscreened resets
are reported separately from the historical selected source population.

## Rendering

`scripts/paper/build_repeat_value_iclr.py` renders the Markdown through the official
ICLR 2027 LaTeX style. The unchanged official files, citation database and style
provenance live in `latex/`. The header marks this as an unsubmitted working draft.
The current PDF contains main text, references and appendices; verify the main-text
page separately from total PDF pages. The required AI-use statement records the
actual assistance and pending human-author review.

`build_repeat_value_pdf.py` remains the historical reportlab review builder.
Use the ICLR builder for the current draft. Experimental figures are generated on
Quest by `scripts/paper/render_iclr_value_figures.py`; no scientific analyses run
in the local PDF builder.

Before sharing an updated PDF, re-render and visually inspect every page. Temporary
equation and page images live under tmp/pdfs/ and are ignored. The PDF is a generated
artifact; manuscript.md is the editable source.
