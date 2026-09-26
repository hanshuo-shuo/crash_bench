# Current — SafeLIBERO reproduction preparation, 2026-09-26

The user authorized cleanup around eight retained results and reproduction of
nominal pi0.5-LIBERO and complete AEGIS. Current scientific code is pinned to
THU-RCSCT/vlsa-aegis 2457feed5968ae803926e178c8ce8243b9ecdcf9.

- Retained evidence: ../retained/README.md and ../retained/manifest.json.
- All previously published history: GitHub branch codex/archive-crashbench-20260926
  at 4351008e0dbaae47adae05af2e49a961bbd161e0. Every former public branch is its ancestor.
- Unpublished local manuscripts/build files: checksum-verified local archive
  .git/crashbench-archive/20260926/local-worktree.tar and private stash
  3f0b0978ad884db576ce8e15949c83dbb621c60c. These have not been published.
- Run protocol and engineering deviations: REPRODUCTION.md.
- OpenRouter GLM-4.5V key is user-owned and not yet configured by this task.

Preparation/smoke results are not benchmark reproduction results. Do not train
an intervention gate or change the benchmark while establishing these baselines.

## Completed cleanup

Main now contains 59 byte-identical selected evidence files, the user's selected
narrative and reproduction tooling. Four redundant public branches were deleted
only after ancestry verification; GitHub retains main and the dedicated archive.

Quest archive job 7531231 completed in 3m04s at ae2406c, exit 0. It verified and
archived 8323 legacy result entries plus 127 logs under
/projects/p33100/siosio/crashbench_archive/20260926/. Ten leftover grasp-authoring
files were separately verified there. Original source/project-data symlinks remain
resolvable. Existing shared model environments and other projects were not deleted.
The first archive job 7531195 failed before changing files because Git was absent
from the compute node PATH; explicit git/2.37.2 module loading fixed startup.

Model checkpoint, tokenizer, BERT and GroundingDINO downloads are complete.
Python distributions, the standalone runtime and the Ubuntu 20.04 CUDA/OpenGL
container are being staged. No benchmark episode has completed yet.
