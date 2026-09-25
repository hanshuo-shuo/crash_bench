#!/usr/bin/env python3
"""Compare the bounded E13 no-stop rerun with its frozen historical matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.careful_prompt_eval import build_summary


NEW_CONDITIONS = (
    "hazard_specific",
    "hazard_specific_no_stop",
    "hazard_specific_goal_first_no_stop",
)

LABELS = {
    "hazard_specific": "原危险提示（本次重跑）",
    "hazard_specific_no_stop": "只删 stop 子句",
    "hazard_specific_goal_first_no_stop": "目标前置且无 stop",
}


def load_and_check(hazard: str, old_path: Path, new_path: Path) -> tuple[dict, dict]:
    old = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())
    for name, payload in (("old", old), ("new", new)):
        cfg = payload["config"]
        if cfg["hazard"] != hazard:
            raise ValueError(f"{name}: expected {hazard}, got {cfg['hazard']}")
        if cfg["repeats"] != 3 or len(payload["scenarios"]) != 10:
            raise ValueError(f"{name}: expected 10 scenes x 3 repeats")
        if payload["summary"] != build_summary(payload["episodes"]):
            raise ValueError(f"{name}: saved summary differs from episode rows")
        expected_count = 10 * 3 * len(cfg["conditions"])
        if len(payload["episodes"]) != expected_count:
            raise ValueError(f"{name}: expected {expected_count} episode rows")
    if tuple(new["config"]["conditions"]) != NEW_CONDITIONS:
        raise ValueError("new condition set differs from fixed plan")
    for key in ("checkpoint", "checkpoint_revision", "unnorm_key"):
        if old["config"][key] != new["config"][key]:
            raise ValueError(f"{hazard}: {key} changed")
    old_scenes = {x["scenario_id"]: x for x in old["scenarios"]}
    new_scenes = {x["scenario_id"]: x for x in new["scenarios"]}
    if old_scenes.keys() != new_scenes.keys():
        raise ValueError(f"{hazard}: scenario identities changed")
    for scene_id in old_scenes:
        for key in ("fingerprint_sha256", "regime", "base_instruction"):
            if old_scenes[scene_id][key] != new_scenes[scene_id][key]:
                raise ValueError(f"{hazard}: {scene_id} {key} changed")
    if (
        old["config"]["prompt_templates"]["hazard_specific"][hazard]
        != new["config"]["prompt_templates"]["hazard_specific"][hazard]
    ):
        raise ValueError(f"{hazard}: original hazard prompt changed")
    for row in new["episodes"]:
        expected = new["config"]["prompt_templates"][row["condition"]][hazard].format(
            instruction=row["base_instruction"]
        )
        if row["effective_instruction"] != expected:
            raise ValueError(f"{hazard}: effective instruction mismatch")
    return old, new


def counts(payload: dict, condition: str, regime: str) -> dict:
    return payload["summary"]["by_condition_and_regime"][condition][regime]


def fmt_counts(cell: dict) -> str:
    n = cell["n"]
    return (
        f"{cell['n_crash']}/{n} | {cell['n_recovery_success']}/{n} | "
        f"{cell['n_safe_abort']}/{n} | {cell['n_timeout']}/{n}"
    )


def report(wall: tuple[dict, dict], glass: tuple[dict, dict], paths: dict) -> str:
    commits = {wall[1]["config"]["git_commit"], glass[1]["config"]["git_commit"]}
    if len(commits) != 1:
        raise ValueError("wall/glass new runs have different commits")
    commit = next(iter(commits))
    lines = [
        "# 去掉 stop 子句的固定场景提示词对照",
        "",
        "本次重跑原危险提示，并比较两个新措辞：仅删除 `stop before it or`；以及把原任务放在最前、删除 stop 并要求绕开后继续任务。第一种是 stop 子句的最小消融；第二种同时改变任务顺序和措辞，只作探索性对照。每种危险各 5 个 on-path 和 5 个 off-path 固定场景，每格 3 次，共 180 个新闭环 episode。旧 E13 只作历史参照；本次重跑的原提示才是直接对照。",
        "",
        "| 危险 | 提示 | 场景 | 碰撞 | 任务成功 | 稳定未完成 | 其他超时 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for hazard, (old, new) in (("wall", wall), ("glass", glass)):
        for regime in ("treatment", "control"):
            lines.append(
                f"| {hazard} | 原危险提示（历史 E13） | {regime} | "
                f"{fmt_counts(counts(old, 'hazard_specific', regime))} |"
            )
            for condition in NEW_CONDITIONS:
                lines.append(
                    f"| {hazard} | {LABELS[condition]} | {regime} | "
                    f"{fmt_counts(counts(new, condition, regime))} |"
                )
    lines += [
        "",
        "**指标说明：**这里的“稳定未完成”是评估器的 `safe_abort`：220 个动作到期后没有触发碰撞或任务成功，且最后一步接触力小于 1 N。它不证明机器人主动停下，亦不证明全程没有危险接触。5 个场景才是几何来源；3 次重复不等于 15 个独立危险。",
        "",
        "## 每个场景的任务成功次数（各 3 次）",
        "",
        "| 危险 | 场景 | 原提示本次 | 只删 stop | 目标前置无 stop |",
        "|---|---|---:|---:|---:|",
    ]
    for hazard, (_, new) in (("wall", wall), ("glass", glass)):
        by_scene = new["summary"]["by_condition_and_scenario"]
        for scene in new["scenarios"]:
            scene_id = scene["scenario_id"]
            values = [
                by_scene[condition][scene_id]["n_recovery_success"]
                for condition in NEW_CONDITIONS
            ]
            lines.append(
                f"| {hazard} | {scene_id} ({scene['regime']}) | "
                + " | ".join(str(value) for value in values)
                + " |"
            )
    lines += [
        "",
        "## 逐字提示与溯源",
        "",
        f"- 新代码 commit：`{commit}`；wall job `{wall[1]['config']['slurm_job_id']}`，glass job `{glass[1]['config']['slurm_job_id']}`。",
        f"- 新原始结果：`{paths['wall']}`、`{paths['glass']}`。",
        "- 旧 E13：`results/careful_prompt/wall_prompt_matrix.json`、`results/careful_prompt/glass_prompt_matrix.json`。",
    ]
    templates = glass[1]["config"]["prompt_templates"]
    for condition in NEW_CONDITIONS:
        for hazard in ("wall", "glass"):
            lines.append(f"- `{condition}` / {hazard}: `{templates[condition][hazard]}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wall", type=Path, required=True)
    parser.add_argument("--glass", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    wall = load_and_check(
        "wall", Path("results/careful_prompt/wall_prompt_matrix.json"), args.wall
    )
    glass = load_and_check(
        "glass", Path("results/careful_prompt/glass_prompt_matrix.json"), args.glass
    )
    output = report(wall, glass, {"wall": args.wall, "glass": args.glass})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(output)
    print(args.out)


if __name__ == "__main__":
    main()
