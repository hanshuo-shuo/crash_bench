#!/usr/bin/env python
"""Phase 2-④ reconnaissance: find IN-DISTRIBUTION (no injected wall) crash hazards in the
libero_10 kitchen scenes (README §4.3 cat-1/cat-7).

Hypothesis to enable: the wall-injection crash is NOT an artifact of an OOD red slab — a VLA
with no pre-crash policy will also drive into a FAMILIAR obstacle when it lies on the path.
The cleanest in-distribution hazard is an *articulation perturbation*: KITCHEN_SCENE4's
cabinet drawer and KITCHEN_SCENE6's microwave door both START OPEN in the BDDL and the task is
to close them. If we instead start them CLOSED, an OpenVLA trained only on open-fixture demos
should drive the held object straight into the closed door/drawer.

This job is pure information-gathering so authoring can be done precisely:
  1. enumerate the suite's task ids -> descriptions (confirm SCENE order)
  2. for a shortlist of scenes, dump every joint (name, qpos addr, range, default value) so we
     can locate the door/drawer articulation and its index in the flat init_state vector
  3. dump furniture geom world positions/sizes (the rigid bodies the arm can hit)
  4. record OpenVLA's NOMINAL eef xyz trajectory (so we know the real path)
  5. render the default (open) init frame AND both range-extreme (candidate "closed") frames for
     each articulation joint, so we can eyeball which extreme blocks the path

Outputs -> results/phase2_recon/{recon.json, <scene>_open.png, <scene>_<joint>_lo/hi.png}
Run on a GPU node (MUJOCO_GL=egl). Needs the libero-10 checkpoint (downloaded on login node).
"""

from __future__ import annotations

import os
import json

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import imageio
import mujoco

from crashbench.envs import LiberoEnv

SUITE = "libero_10"
CKPT = "openvla/openvla-7b-finetuned-libero-10"
UNNORM = "libero_10"
SHORTLIST = [1, 2, 9]          # expected: drawer (SCENE4), microwave (SCENE6), caddy (STUDY1)
SETTLE = 10
NOMINAL_STEPS = 160
ART_KEYS = ("microwave", "door", "drawer", "cabinet", "caddy", "hinge", "slide")
OUT = "results/phase2_recon"
os.makedirs(OUT, exist_ok=True)


def _live_mj(env):
    sim = env.env.sim
    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    return model, data


def introspect(env):
    model, data = _live_mj(env)
    joints = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = int(model.jnt_qposadr[j])
        joints.append({
            "name": name, "qposadr": adr, "type": int(model.jnt_type[j]),
            "range": [round(float(x), 4) for x in model.jnt_range[j]],
            "default_val": round(float(data.qpos[adr]), 4),
        })
    geoms = []
    for g in range(model.ngeom):
        bid = int(model.geom_bodyid[g])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        geoms.append({
            "geom": gname, "body": bname,
            "pos": [round(float(x), 3) for x in data.geom_xpos[g]],
            "size": [round(float(x), 3) for x in model.geom_size[g]],
            "type": int(model.geom_type[g]),
        })
    return {"nq": int(model.nq), "nv": int(model.nv), "joints": joints, "geoms": geoms}


def furniture_joints(introspection):
    out = []
    for jt in introspection["joints"]:
        nm = (jt["name"] or "").lower()
        if any(k in nm for k in ART_KEYS):
            out.append(jt)
    return out


def main():
    from crashbench.policies import OpenVLAPolicy

    # (1) enumerate the suite -------------------------------------------------
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()[SUITE]()
    n_tasks = suite.n_tasks
    task_index = []
    for tid in range(n_tasks):
        t = suite.get_task(tid)
        task_index.append({"task_id": tid, "name": t.name,
                           "language": getattr(t, "language", None)})
        print(f"  task {tid}: {t.name}")

    policy = OpenVLAPolicy(pretrained_checkpoint=CKPT, unnorm_key=UNNORM)

    recon = {"suite": SUITE, "checkpoint": CKPT, "tasks": task_index, "scenes": {}}

    for tid in SHORTLIST:
        env = LiberoEnv(SUITE, tid)
        base = np.asarray(env.default_init_states()[0])
        tag = f"t{tid}"
        print(f"\n=== {tag}: {env.task_description!r}  (init_state len={len(base)}) ===")

        obs = env.reset_to(base)
        for _ in range(SETTLE):
            obs, _, _, _ = env.step(env.dummy_action())

        intro = introspect(env)
        intro["init_state_len"] = int(len(base))
        intro["layout_is_time_qpos_qvel"] = bool(len(base) == 1 + intro["nq"] + intro["nv"])
        fjoints = furniture_joints(intro)
        print(f"    nq={intro['nq']} nv={intro['nv']} init_len={len(base)} "
              f"time+qpos+qvel={intro['layout_is_time_qpos_qvel']}")
        print(f"    furniture joints: {[(j['name'], j['qposadr'], j['range'], j['default_val']) for j in fjoints]}")

        # render default (open) init frame
        imageio.imwrite(f"{OUT}/{tag}_open.png", env.render(obs, 256))

        # render both range-extremes for each furniture joint (candidate 'closed')
        art_renders = []
        for jt in fjoints:
            adr = jt["qposadr"]
            idx = 1 + adr if intro["layout_is_time_qpos_qvel"] else None
            for which, val in (("lo", jt["range"][0]), ("hi", jt["range"][1])):
                if idx is None or val == jt["default_val"]:
                    continue
                pert = base.copy()
                pert[idx] = val
                o2 = env.reset_to(pert)
                for _ in range(SETTLE):
                    o2, _, _, _ = env.step(env.dummy_action())
                fn = f"{tag}_{jt['name']}_{which}.png".replace("/", "_")
                imageio.imwrite(f"{OUT}/{fn}", env.render(o2, 256))
                art_renders.append({"joint": jt["name"], "which": which, "val": val,
                                    "init_state_idx": idx, "frame": fn})
                print(f"    rendered {jt['name']} {which}={val} -> {fn}")

        # (4) nominal OpenVLA trajectory (from the DEFAULT/open state)
        obs = env.reset_to(base)
        for _ in range(SETTLE):
            obs, _, _, _ = env.step(env.dummy_action())
        eef_xyz, success = [], False
        for _ in range(NOMINAL_STEPS):
            o = env.policy_observation(obs, policy.resize_size)
            a = policy.act(o, env.task_description)
            obs, _, done, _ = env.step(a.tolist() if hasattr(a, "tolist") else a)
            eef_xyz.append([round(float(x), 4) for x in np.asarray(obs["robot0_eef_pos"])])
            if done:
                success = True
                break
        print(f"    nominal: steps={len(eef_xyz)} success={success} "
              f"eef_start={eef_xyz[0]} eef_end={eef_xyz[-1]}")

        recon["scenes"][tag] = {
            "task_id": tid, "task_description": env.task_description,
            "introspection": intro, "furniture_joints": fjoints,
            "art_renders": art_renders,
            "nominal": {"steps": len(eef_xyz), "success": success, "eef_xyz": eef_xyz},
        }

    json.dump(recon, open(f"{OUT}/recon.json", "w"), indent=2)
    print(f"\nwrote {OUT}/recon.json  (+ frames)")


if __name__ == "__main__":
    main()
