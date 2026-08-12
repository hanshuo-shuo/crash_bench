# Legacy executable helpers

This directory contains superseded orchestration helpers that are no longer safe
or useful as default entrypoints. They are retained for provenance rather than
left beside current submission scripts.

- `legacy/setup/submit_next_round.sh`: old multi-job paper-round submitter.
- `legacy/setup/commit_p0_core.sh`: old P0 verify/stage/commit/push convenience helper.

Both fail closed unless `CB_ENABLE_LEGACY_SUBMISSIONS=1` is explicitly set. The
current run guide is [setup/README.md](../setup/README.md).

Most legacy research code remains at historical paths when manifests, tests, or
imports require path stability; see
[docs/SCRIPT_INDEX.md](../docs/SCRIPT_INDEX.md).
