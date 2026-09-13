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
5. Retained recovery opportunities and execution-time learned selection remain distinct.

## Evidence entry points

- [Existing-record contract](../../audits/20260913/repeat_value/PLAN.md)
- [Existing-record tables and figures](../../audits/20260913/repeat_value/evidence/RESULTS.md)
- [Prospective C contract](../../audits/20260913/repeat_value/C_PLAN.md)
- [C machine freeze](../../audits/20260913/repeat_value/c_contract.json)
- [Quest provenance](../../audits/20260913/repeat_value/RUN.md)
- [Claim-to-row mapping](EVIDENCE_MAP.md)

The completed C experiment has 332 branches on saved existing states. It measures subsequent
execution, not new-source generalization. A/B forecasts and all method choices were
published before either C job began. No model or recovery controller is refitted.

## Rendering

`scripts/paper/build_repeat_value_pdf.py` renders the Markdown into the review PDF.
It uses reportlab, Pillow, a local TeX installation for two equations, Poppler,
and DejaVu fonts; paths presently match this macOS workspace. It is a review format,
not the official conference submission template. Experimental figures are generated
on Quest by the corresponding analysis entry points.

Before sharing an updated PDF, re-render and visually inspect every page. Temporary
equation and page images live under tmp/pdfs/ and are ignored. The PDF is a generated
artifact; manuscript.md is the editable source.
