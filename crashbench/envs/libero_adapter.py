"""LIBERO/robosuite substrate adapter (PLAN.md §1, §5).

Wraps the exact API used by OpenVLA's `run_libero_eval.py` so CrashBench reuses the
proven observation/action bridge instead of rebuilding it:

    benchmark.get_benchmark_dict()[suite]()  -> task_suite
    task_suite.get_task(task_id)             -> task
    get_libero_env(task, model_family, 256)  -> (env, task_description)
    env.reset(); env.set_init_state(state)   -> obs        # our pre-crash state goes here
    env.step(action) -> obs, reward, done, info            # done == LIBERO task success
    get_libero_image(obs, resize_size)       -> img (for the policy)

`LiberoSimView` exposes the read-only quantities the predicates need. We read them
from the robosuite *observation dict* (which already contains `<object>_pos`,
`robot0_eef_pos`, `robot0_gripper_qpos`, ...) rather than poking the low-level
mujoco `sim.data`, which is both simpler and more stable across robosuite versions.

Object names in predicates use the obs convention WITHOUT the `_main` body suffix,
e.g. `akita_black_bowl_1` (obs key `akita_black_bowl_1_pos`).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


# Dynamic MuJoCo fields that are part of a continuation, but are omitted by
# LIBERO's flattened ``[time, qpos, qvel]`` state.  Not every MuJoCo version
# exposes every field, so capture is capability based and restore is driven by
# the keys present in the snapshot.  In particular, qacc_warmstart carries the
# iterative solver warm start while the applied-force, mocap, equality, user,
# and plugin fields are inputs to the next integration step.
_SIM_CONTINUATION_FIELDS = (
    "qacc_warmstart",
    "act",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "eq_active",
    "userdata",
    "plugin_state",
)

_ROBOT_RUNTIME_BUFFERS = (
    "recent_qpos",
    "recent_actions",
    "recent_torques",
    "recent_ee_forcetorques",
    "recent_ee_pose",
    "recent_ee_vel",
    "recent_ee_vel_buffer",
    "recent_ee_acc",
)

def _default_openvla_root() -> str:
    """Repo whose `experiments.robot.*` we import. Defaults to base OpenVLA, but the
    env var CRASHBENCH_OPENVLA_ROOT overrides it — Path 3 (OpenVLA-OFT) sets this to the
    OFT repo so BOTH the env builder and the OFT policy resolve the SAME `experiments.robot`
    namespace (the two repos share that package name and cannot coexist in one process).
    Resolved per-call (not a frozen import-time constant) so a policy can set the env var
    before any LiberoEnv is constructed."""
    return os.environ.get("CRASHBENCH_OPENVLA_ROOT") or os.path.expanduser(
        "~/crash_bench/third_party/openvla")


DEFAULT_OPENVLA_ROOT = _default_openvla_root()   # back-compat module constant (base default)


def add_openvla_to_path(openvla_root: str | None = None) -> None:
    """Put the OpenVLA (or OFT) repo on sys.path so `experiments.robot.*` imports resolve.
    openvla_root=None -> resolve via CRASHBENCH_OPENVLA_ROOT / base default."""
    root = str(Path(openvla_root or _default_openvla_root()).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _native_get_libero_env(task, resolution: int = 256, seed: int = 0):
    """torch-free replica of experiments.robot.libero.libero_utils.get_libero_env, for the π0
    (openpi/JAX) path. That helper's module imports tensorflow + experiments.robot.robot_utils
    (→ torch), which we must NOT pull into the JAX env. This builds the SAME OffScreenRenderEnv
    (identical args + seed(0)) straight from LIBERO, so scenarios / saved init_states stay
    apples-to-apples across architectures — only the OpenVLA/torch dependency is dropped."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)  # IMPORTANT: seed affects object positions even with a fixed init_state
    return env, task.language


def _native_quat2axisangle(quat):
    """torch-free copy of robosuite's quat2axisangle (x,y,z,w)->axis-angle vec3, matching the
    OpenVLA/OFT proprio-state preprocessing exactly (π0 packs the same 8-dim state)."""
    import math
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = min(1.0, max(-1.0, float(q[3])))
    den = np.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * math.acos(q[3])) / den


def inject_obstacles_xml(xml: str, obstacles: list[dict]) -> str:
    """Insert static (jointless) obstacle bodies into a robosuite model XML string.

    Each obstacle: {"name", "pos":[x,y,z], "size":[...], "type"="box", "rgba"=[...]};
    box sizes are [sx,sy,sz], while cylinder sizes are [radius,half_height]. Optional
    ``euler`` / ``quat`` and ``friction`` fields are copied onto the geom. This keeps
    shallow static support ramps in the same exact-XML mechanism as upright walls.
    Static bodies add geoms but NO qpos/qvel DOF, so the LIBERO state vector layout is
    unchanged and saved init_states stay valid (verified in scripts/probe_wall_inject.py).

    group="1" puts the geom in the VISUAL render group so the agentview camera (hence the
    VLA) actually SEES the obstacle — a group-0 collision geom is invisible to the camera,
    which would unfairly test crashing into an unperceivable wall. Collision is governed by
    contype/conaffinity (=1), independent of the render group, so it still collides.
    """
    if not obstacles:
        return xml
    assert "</worldbody>" in xml, "model xml has no </worldbody> to inject into"
    blocks = []
    for o in obstacles:
        px, py, pz = o["pos"]
        size = " ".join(str(v) for v in o["size"])
        gtype = o.get("type", "box")
        rgba = " ".join(str(v) for v in o.get("rgba", [0.85, 0.2, 0.2, 1.0]))
        pose = ""
        if "quat" in o and "euler" in o:
            raise ValueError(f"obstacle {o['name']!r} cannot declare both quat and euler")
        if "quat" in o:
            pose = f' quat="{" ".join(str(v) for v in o["quat"])}"'
        elif "euler" in o:
            pose = f' euler="{" ".join(str(v) for v in o["euler"])}"'
        friction = (
            "" if "friction" not in o
            else f' friction="{" ".join(str(v) for v in o["friction"])}"'
        )
        blocks.append(
            f'<body name="{o["name"]}" pos="{px} {py} {pz}">'
            f'<geom name="{o["name"]}_g" type="{gtype}" size="{size}" '
            f'rgba="{rgba}"{pose}{friction} group="1" contype="1" '
            f'conaffinity="1"/></body>'
        )
    return xml.replace("</worldbody>", "".join(blocks) + "</worldbody>", 1)


def inject_movable_objects_xml(xml: str, movables: list[dict]) -> str:
    """Insert FREE-JOINTED (movable) primitive objects at the END of worldbody
    (README §4.3 cat-2, the fragile-object-on-path hazard).

    Unlike `inject_obstacles_xml` (static walls), each object here gets a `<freejoint>`
    so it can be SWEPT or TOPPLED — the object-collision crash signal (a struck free object
    slides/tips rather than resisting, so contact force stays low; displacement/tilt is the
    reliable "was it hit" signal). Appended LAST in worldbody so their free joints are the
    last joints in the model: their 7 qpos + 6 qvel land at the very end of the state vector,
    leaving every existing DOF index unchanged. `reset_to` therefore only has to APPEND the
    new objects' pose to a saved init_state (which was authored for the un-injected model).

    Each object: {"name", "pos":[x,y,z], "size":[...], "type"="cylinder"(default; a "glass"
    is a slender upright cylinder [radius, half_height]), "rgba"=[...], "density"=float}.
    The free joint's default quat is identity, so a cylinder rests UPRIGHT.

    group="1" (visual render group) so the agentview camera — hence the VLA — actually SEES
    the object; contype/conaffinity=1 make it collidable (same rationale as the wall inject).
    """
    if not movables:
        return xml
    assert "</worldbody>" in xml, "model xml has no </worldbody> to inject into"
    blocks = []
    for o in movables:
        px, py, pz = o["pos"]
        size = " ".join(str(v) for v in o["size"])
        gtype = o.get("type", "cylinder")
        rgba = " ".join(str(v) for v in o.get("rgba", [0.55, 0.78, 0.95, 0.55]))
        density = o.get("density", 400.0)   # light like a real glass/plastic cup
        blocks.append(
            f'<body name="{o["name"]}" pos="{px} {py} {pz}">'
            f'<freejoint name="{o["name"]}_joint"/>'
            f'<geom name="{o["name"]}_g" type="{gtype}" size="{size}" rgba="{rgba}" '
            f'group="1" contype="1" conaffinity="1" density="{density}"/></body>'
        )
    return xml.replace("</worldbody>", "".join(blocks) + "</worldbody>", 1)


# Robot/gripper bodies that can legitimately strike the environment. Contact force
# on these = an env-collision (README §4.3 cat-1). Verified body names from the live
# LIBERO Panda model (scripts/probe_contact_force.py). The distal arm links never
# touch anything in a nominal grasp, so force there is unambiguous collision; the
# fingers DO press objects during a grasp (~20-70 N), so collision scenarios on the
# gripper should use a high threshold or an explicit `against` obstacle body.
ROBOT_CONTACT_BODIES = (
    "gripper0_leftfinger", "gripper0_rightfinger",
    "gripper0_finger_joint1_tip", "gripper0_finger_joint2_tip",
    "gripper0_right_gripper", "robot0_right_hand",
    "robot0_link7", "robot0_link6", "robot0_link5",
)

# Per-contact force ceiling (N). Real gripper-vs-rigid impacts read ~200-600 N; values far
# above this come from MuJoCo soft-contact DEEP PENETRATION (a numerical blow-up, e.g. a
# gripper jammed into a wall for many steps reading tens of thousands of N), not physics. We
# clamp per contact so the impact-severity metric (README §5) stays physically meaningful.
FORCE_CLAMP = 2000.0


class LiberoSimView:
    """Read-only view backing the predicates, sourced from the robosuite obs dict
    plus live MuJoCo contact forces.

    Implements crashbench.predicates.SimView. Updated every step by LiberoEnv.

    NOTE: robosuite REBUILDS the sim (new MjModel/MjData) on every env.reset(), so we
    NEVER cache the raw mujoco structs — we re-fetch them live from env.sim on each
    query (verified in scripts/probe_contact_force.py; caching across a reset reads a
    dead sim and silently returns frozen/zero forces).
    """

    def __init__(self, env):
        self._env = env
        self._obs: dict = {}
        self._last_done = False
        self.peak_force = 0.0  # running max contact force on robot bodies (impact severity)

    def update(self, obs: dict, done: bool) -> None:
        self._obs = obs or {}
        self._last_done = bool(done)
        # track peak impact force on the robot for the impact-severity metric (README §5)
        self.peak_force = max(self.peak_force, self.max_contact_force(list(ROBOT_CONTACT_BODIES)))

    @property
    def libero_done(self) -> bool:
        return self._last_done

    def object_z(self, object_name: str) -> float:
        """World z (m) of an object. Reads obs `<object_name>_pos` for BDDL objects;
        falls back to the live MuJoCo body pose for INJECTED objects (which have no obs
        observable, cat-2 fragile objects)."""
        key = f"{object_name}_pos"
        if key in self._obs:
            return float(np.asarray(self._obs[key])[2])
        return float(self._body_pos(object_name)[2])

    def object_xy(self, object_name: str) -> tuple[float, float]:
        """World (x, y) (m) of an object. Reads obs `<object_name>_pos` for BDDL objects,
        else the live MuJoCo body pose for injected objects. Used by the object_displaced
        predicate (object-collision crashes, README §4.3 cat-2): a struck object SLIDES
        rather than resisting, so displacement — not contact force — is the reliable signal
        of being swept."""
        key = f"{object_name}_pos"
        if key in self._obs:
            p = np.asarray(self._obs[key])
            return float(p[0]), float(p[1])
        p = self._body_pos(object_name)
        return float(p[0]), float(p[1])

    def object_tilt_deg(self, object_name: str) -> float:
        """Tilt of an object's local +z axis away from world +z, in degrees (0 = upright).
        Used by the object_toppled predicate (a swept fragile object TIPS OVER, README §4.3
        cat-2). Read from the live MuJoCo body orientation, so it works for both BDDL and
        injected objects."""
        model, data = self._live_mj()
        import mujoco
        bid = self._body_id(model, object_name)
        # world-frame local z axis = 3rd column of the body rotation matrix (row-major 3x3)
        zc = float(np.asarray(data.xmat[bid]).reshape(3, 3)[2, 2])
        return float(np.degrees(np.arccos(np.clip(zc, -1.0, 1.0))))

    def _body_id(self, model, name: str) -> int:
        import mujoco
        for cand in (name, f"{name}_main"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cand)
            if bid >= 0:
                return bid
        raise KeyError(f"no MuJoCo body {name!r} (also tried {name}_main)")

    def _body_pos(self, name: str) -> np.ndarray:
        model, data = self._live_mj()
        return np.asarray(data.xpos[self._body_id(model, name)]).copy()

    def is_grasped(self, object_name: str) -> bool:
        """Heuristic grasp check: object near the eef AND gripper not fully open.

        TODO(verify): replace with robosuite `env._check_grasp` for grasp_dropped
        scenarios. The pilot (unsafe-terminal / object_fell) does not use this.
        """
        rel = self._obs.get(f"{object_name}_to_robot0_eef_pos")
        grip = self._obs.get("robot0_gripper_qpos")
        if rel is None or grip is None:
            return False
        near = float(np.linalg.norm(rel)) < 0.06
        closed = float(np.sum(np.abs(grip))) < 0.06  # TODO(verify) gripper-closed threshold
        return near and closed

    def _live_mj(self):
        """Live raw (mujoco.MjModel, mujoco.MjData) from env.sim. Re-fetched every call
        because robosuite rebuilds the sim on reset (see class docstring)."""
        sim = self._env.sim
        model = getattr(sim.model, "_model", sim.model)
        data = getattr(sim.data, "_data", sim.data)
        return model, data

    def joint_state(self) -> dict[str, list[float]]:
        """Policy-relevant robot joint position/velocity from the live simulator."""
        model, data = self._live_mj()
        import mujoco
        qpos, qvel, names = [], [], []
        for jid in range(int(model.njnt)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
            if not (name.startswith("robot0_") or name.startswith("gripper0_")):
                continue
            qadr = int(model.jnt_qposadr[jid])
            dadr = int(model.jnt_dofadr[jid])
            names.append(name)
            qpos.append(float(data.qpos[qadr]))
            qvel.append(float(data.qvel[dadr]) if dadr >= 0 else 0.0)
        return {"names": names, "qpos": qpos, "qvel": qvel}

    def robot_geom_aabbs(self, bodies: list[str]) -> list[dict]:
        """World AABBs of collision geoms attached to the requested robot bodies.

        MuJoCo stores a local geom AABB as center+half-size.  Transforming it by
        ``abs(R)`` gives the exact world AABB of that local AABB and a conservative
        bound for the underlying rotated geometry.
        """
        model, data = self._live_mj()
        import mujoco
        body_ids = {self._body_id(model, name): name for name in bodies}
        rows = []
        for gid in range(int(model.ngeom)):
            bid = int(model.geom_bodyid[gid])
            if bid not in body_ids or int(model.geom_contype[gid]) == 0:
                continue
            local = np.asarray(model.geom_aabb[gid], dtype=float)
            local_center, local_half = local[:3], local[3:]
            rotation = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3)
            center = np.asarray(data.geom_xpos[gid], dtype=float) + rotation @ local_center
            half = np.abs(rotation) @ local_half
            if not np.any(half > 0):
                half = np.full(3, float(model.geom_rbound[gid]))
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
            rows.append({
                "body": body_ids[bid], "geom": geom_name,
                "lo": (center - half).tolist(), "hi": (center + half).tolist(),
            })
        return rows

    def max_contact_force(self, bodies: list[str], against: list[str] | None = None) -> float:
        """Max ||contact force|| (N) over contacts that touch any of `bodies`.

        If `against` is given, only count contacts where the OTHER body is in `against`
        (e.g. a specific wall/obstacle) — this isolates an env-collision from the normal
        gripper-on-object grasp contacts. Force from mujoco's per-contact mj_contactForce
        (verified in scripts/probe_contact_force.py: ~350-400 N gripper-vs-stove, ~20-70 N
        gripper-vs-bowl grasp). Background static-clutter contacts are excluded because we
        filter by body id.
        """
        import mujoco

        model, data = self._live_mj()

        def ids(names):
            out = set()
            for n in names:
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
                if bid >= 0:
                    out.add(bid)
            return out

        want = ids(bodies)
        if not want:
            return 0.0
        against_ids = ids(against) if against else None

        res = np.zeros(6, dtype=np.float64)
        peak = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = int(model.geom_bodyid[c.geom1])
            b2 = int(model.geom_bodyid[c.geom2])
            hit1, hit2 = b1 in want, b2 in want
            if not (hit1 or hit2):
                continue
            if against_ids is not None:
                other = b2 if hit1 else b1
                if other not in against_ids:
                    continue
            mujoco.mj_contactForce(model, data, i, res)
            peak = max(peak, min(float(np.linalg.norm(res[:3])), FORCE_CLAMP))
        return peak

    def _robot_bodies(self) -> list[str]:
        return list(ROBOT_CONTACT_BODIES)


class LiberoEnv:
    """Thin wrapper over a LIBERO task env using OpenVLA's verified helpers."""

    def __init__(self, task_suite: str, task_id: int, model_family: str = "openvla",
                 resolution: int = 256, openvla_root: str | None = None, seed: int = 0):
        from libero.libero import benchmark   # LIBERO (not OpenVLA) — safe on every path

        suite = benchmark.get_benchmark_dict()[task_suite]()
        self.task = suite.get_task(task_id)
        self.task_suite = task_suite
        self.task_id = task_id
        self._model_family = model_family
        if model_family == "pi0":
            # π0 (openpi/JAX) path: build the env WITHOUT the OpenVLA repo (no torch/tensorflow).
            self.env, self.task_description = _native_get_libero_env(
                self.task, resolution=resolution, seed=seed)
        else:
            add_openvla_to_path(openvla_root)   # None -> CRASHBENCH_OPENVLA_ROOT / base default
            from experiments.robot.libero.libero_utils import get_libero_env
            self.env, self.task_description = get_libero_env(self.task, model_family, resolution=resolution)
            self.env.seed(seed)
        self.seed_value = int(seed)
        self.sim_view = LiberoSimView(self.env)
        # Keep the exact source bytes associated with the live model. MuJoCo's
        # get_xml() output is not a canonical model identity, so reserializing
        # the same compiled model later can produce a different text hash.
        self._continuation_model_xml = str(self.env.sim.model.get_xml())

    def seed(self, seed: int) -> None:
        self.seed_value = int(seed)
        self.env.seed(self.seed_value)

    def rng_sources(self) -> dict[str, dict[str, object]]:
        """Return named env/controller/sensor NumPy RNG objects for exact restore.

        Capability discovery is deliberately shallow and allowlisted. It avoids
        serializing arbitrary environment objects while still covering the RNG
        attributes used across Gym, robosuite controllers, and observables.
        Aliases are deduplicated by object identity.
        """

        owners: list[tuple[str, object]] = [("env", self.env)]
        try:
            raw = self._raw_env()
        except RuntimeError:
            raw = None
        if raw is not None:
            owners.append(("raw_env", raw))
            for index, robot in enumerate(getattr(raw, "robots", ())):
                owners.append((f"robot{index}", robot))
                controller = getattr(robot, "controller", None)
                if controller is not None:
                    owners.append((f"robot{index}.controller", controller))
            for name, observable in sorted(getattr(raw, "_observables", {}).items()):
                owners.append((f"observable.{name}", observable))

        generators: dict[str, np.random.Generator] = {}
        random_states: dict[str, np.random.RandomState] = {}
        seen: set[int] = set()
        for owner_name, owner in owners:
            for attribute in (
                "np_random", "_np_random", "rng", "_rng", "random_state", "_random_state"
            ):
                value = getattr(owner, attribute, None)
                if id(value) in seen:
                    continue
                if isinstance(value, np.random.Generator):
                    generators[f"{owner_name}.{attribute}"] = value
                    seen.add(id(value))
                elif isinstance(value, np.random.RandomState):
                    random_states[f"{owner_name}.{attribute}"] = value
                    seen.add(id(value))
        return {"generators": generators, "random_states": random_states}

    def default_init_states(self) -> np.ndarray:
        """Load trusted, repository-pinned LIBERO initial states.

        PyTorch 2.6 changed ``torch.load`` to ``weights_only=True`` by default,
        but LIBERO's historical init-state files contain NumPy arrays. They are
        not model checkpoints and come from the locally pinned LIBERO checkout,
        so this narrow loader explicitly opts into the legacy trusted format.
        """

        import torch
        from libero.libero import benchmark, get_libero_path

        suite = benchmark.get_benchmark_dict()[self.task_suite]()
        task = suite.get_task(self.task_id)
        path = os.path.join(
            get_libero_path("init_states"), task.problem_folder, task.init_states_file
        )
        try:
            values = torch.load(path, weights_only=False)
        except TypeError:
            values = torch.load(path)
        return np.asarray(values)

    # ---- rollout API (mirrors run_libero_eval.py) --------------------------
    def reset_to(self, init_state: np.ndarray, obstacles: list[dict] | None = None,
                 movable_objects: list[dict] | None = None):
        """Reset to a pre-crash state, optionally injecting static `obstacles` (walls,
        cat-1) and/or free-jointed `movable_objects` (fragile objects on the path, cat-2).

        Injection flow (verified in scripts/probe_wall_inject.py): a plain env.reset()
        rebuilds the scene from the BDDL task and would WIPE injected bodies, so we
        reset_from_xml_string(modified_xml) to rebuild WITH them, then set the state.

        Static obstacles add geoms but NO DOF, so `init_state` stays valid as-is. Movable
        objects each add a free joint (+7 qpos, +6 qvel). Because they are appended LAST in
        worldbody (see inject_movable_objects_xml), their DOFs land at the very end of the
        state vector, so we splice the saved (un-injected) init_state by APPENDING each
        object's pose (x,y,z + identity quat) and zero velocity — every existing index is
        untouched. The initial pose comes from each object's XML `pos` attribute.
        """
        if not obstacles and not movable_objects:
            self.env.reset()
            model_xml = str(self.env.sim.model.get_xml())
            obs = self.env.set_init_state(init_state)
        else:
            self.env.reset()                                   # clean rebuild from BDDL
            # DOF count of the un-injected model, to know where appended qpos/qvel begin
            m0 = self._raw_model()
            nq0, nv0 = int(m0.nq), int(m0.nv)
            assert len(init_state) == 1 + nq0 + nv0, (
                f"init_state len {len(init_state)} != 1+nq+nv={1+nq0+nv0}")
            xml = self.env.sim.model.get_xml()
            xml = inject_obstacles_xml(xml, obstacles or [])
            xml = inject_movable_objects_xml(xml, movable_objects or [])
            self.env.reset_from_xml_string(xml)                # rebuild WITH injected bodies
            model_xml = xml
            state = self._splice_movable_state(init_state, nq0, nv0, movable_objects or [])
            obs = self.env.set_init_state(state)               # set spliced state, no reset
        self._continuation_model_xml = model_xml
        self.sim_view.peak_force = 0.0                         # reset impact tracker per episode
        self.sim_view.update(obs, done=False)
        return obs

    def flat_state(self) -> np.ndarray:
        """Return the exact live MuJoCo state in LIBERO's flattened convention.

        This includes simulation time, every qpos, and every qvel.  It is used by
        the glass paired-data collector to restore the nominal and oracle branch
        to byte-identical on-path pre-crash states.
        """

        model, data = self.sim_view._live_mj()
        return np.concatenate([
            np.asarray([data.time], dtype=np.float64),
            np.asarray(data.qpos[:int(model.nq)], dtype=np.float64).copy(),
            np.asarray(data.qvel[:int(model.nv)], dtype=np.float64).copy(),
        ])

    def model_xml(self) -> str:
        """Return the exact compiled scene source used by the live simulator.

        Reconstructing an injected model from a newly randomized LIBERO reset is
        not an exact-model contract even when qpos/qvel happen to hash equally.
        Branches that claim continuation equivalence should rebuild from these
        exact XML bytes.
        """

        model_xml = getattr(self, "_continuation_model_xml", None)
        if model_xml is None:
            model_xml = str(self.env.sim.model.get_xml())
            self._continuation_model_xml = model_xml
        return model_xml

    def _raw_env(self):
        """Unwrap LIBERO/robosuite wrappers to the environment owning runtime state."""

        raw = self.env
        # LIBERO's ControlEnv exposes ``robots`` as a forwarding property, but
        # episode counters, observables, and the observation cache live on its
        # nested robosuite task env. Stopping merely because ``robots`` exists
        # silently omits done/timestep and observable history from snapshots.
        while not hasattr(raw, "_observables") and hasattr(raw, "env"):
            raw = raw.env
        if not hasattr(raw, "robots") or not hasattr(raw, "_observables"):
            raise RuntimeError("could not locate robosuite runtime environment")
        return raw

    @staticmethod
    def _encode_runtime_value(value) -> np.ndarray:
        if value is None:
            return np.empty(0, dtype=np.float64)
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("runtime state cannot contain object arrays")
        return array.copy()

    @staticmethod
    def _assign_runtime_value(owner, name: str, encoded: np.ndarray) -> None:
        """Restore a numeric attribute while preserving scalar Python types."""

        current = getattr(owner, name, None)
        value = np.asarray(encoded)
        if value.size == 0 and (current is None or name == "ori_ref"):
            setattr(owner, name, None)
        elif value.shape == () and value.dtype.kind == "b":
            setattr(owner, name, bool(value.item()))
        elif value.shape == () and value.dtype.kind in "iu":
            setattr(owner, name, int(value.item()))
        elif value.shape == () and value.dtype.kind in "fc":
            setattr(owner, name, float(value.item()))
        else:
            setattr(owner, name, value.copy())

    def controller_state(self) -> dict[str, np.ndarray]:
        """Capture the complete policy-boundary continuation state.

        The historical name is retained for artifact compatibility, but the
        snapshot now covers simulator integration inputs, OSC/interpolators,
        robosuite episode counters, robot temporal buffers, observable values
        and timers, and the observation cache.  All values remain plain numeric
        arrays so the snapshot is safe to store in ``npz`` without pickle.
        """

        encode = self._encode_runtime_value
        raw = self._raw_env()
        snapshot: dict[str, np.ndarray] = {}
        _, data = self.sim_view._live_mj()
        for name in _SIM_CONTINUATION_FIELDS:
            if hasattr(data, name):
                snapshot[f"sim.{name}"] = encode(getattr(data, name))

        for name in ("cur_time", "timestep", "done"):
            if hasattr(raw, name):
                snapshot[f"env.{name}"] = encode(getattr(raw, name))

        for robot_index, robot in enumerate(raw.robots):
            controller = robot.controller
            prefix = f"robot{robot_index}"
            for name in (
                "goal_pos", "goal_ori", "ori_ref", "relative_ori",
                "initial_joint", "kp", "kd", "torques",
            ):
                if hasattr(controller, name):
                    snapshot[f"{prefix}.controller.{name}"] = encode(
                        getattr(controller, name)
                    )
            snapshot[f"{prefix}.controller.new_update"] = np.asarray(
                bool(controller.new_update), dtype=np.bool_
            )
            for interpolator_name in ("interpolator_pos", "interpolator_ori"):
                interpolator = getattr(controller, interpolator_name, None)
                if interpolator is None:
                    continue
                for name in ("start", "goal", "step"):
                    snapshot[f"{prefix}.{interpolator_name}.{name}"] = encode(
                        getattr(interpolator, name)
                    )

            if hasattr(robot, "torques"):
                snapshot[f"{prefix}.torques"] = encode(robot.torques)
            for buffer_name in _ROBOT_RUNTIME_BUFFERS:
                buffer = getattr(robot, buffer_name, None)
                if buffer is None:
                    continue
                for name, value in getattr(buffer, "__dict__", {}).items():
                    try:
                        snapshot[f"{prefix}.buffer.{buffer_name}.{name}"] = encode(value)
                    except TypeError:
                        # Buffer metadata can include implementation helpers; only
                        # numeric continuation state belongs in the portable file.
                        continue

        observables = getattr(raw, "_observables", {})
        for observable_name, observable in observables.items():
            prefix = f"observable.{observable_name}"
            for name in (
                "_time_since_last_sample", "_current_delay",
                "_current_observed_value", "_sampled",
            ):
                if hasattr(observable, name):
                    snapshot[f"{prefix}.{name}"] = encode(getattr(observable, name))
        for cache_name, value in getattr(raw, "_obs_cache", {}).items():
            snapshot[f"obs_cache.{cache_name}"] = encode(value)
        # Observable internals define future sampling, but they do not always
        # reconstruct the exact wrapper-level dictionary received by the
        # policy. Persist that policy-boundary observation explicitly.
        for observation_name, value in getattr(self.sim_view, "_obs", {}).items():
            snapshot[f"observation.{observation_name}"] = encode(value)
        return snapshot

    def restore_controller_state(
        self,
        snapshot: dict[str, np.ndarray],
        *,
        restore_observables: bool = True,
    ):
        """Restore a continuation snapshot and return its restored observation."""

        raw = self._raw_env()
        for robot_index, robot in enumerate(raw.robots):
            controller = robot.controller
            controller.update(force=True)
            prefix = f"robot{robot_index}"
            for name in (
                "goal_pos", "goal_ori", "ori_ref", "relative_ori", "new_update",
                "initial_joint", "kp", "kd", "torques",
            ):
                key = f"{prefix}.controller.{name}"
                if key in snapshot:
                    self._assign_runtime_value(controller, name, snapshot[key])
            for interpolator_name in ("interpolator_pos", "interpolator_ori"):
                interpolator = getattr(controller, interpolator_name, None)
                if interpolator is None:
                    continue
                for name in ("start", "goal", "step"):
                    key = f"{prefix}.{interpolator_name}.{name}"
                    if key in snapshot:
                        self._assign_runtime_value(interpolator, name, snapshot[key])
            torque_key = f"{prefix}.torques"
            if torque_key in snapshot:
                self._assign_runtime_value(robot, "torques", snapshot[torque_key])
            for buffer_name in _ROBOT_RUNTIME_BUFFERS:
                buffer = getattr(robot, buffer_name, None)
                if buffer is None:
                    continue
                field_prefix = f"{prefix}.buffer.{buffer_name}."
                for key, value in snapshot.items():
                    if key.startswith(field_prefix):
                        self._assign_runtime_value(
                            buffer, key[len(field_prefix):], value
                        )

        # controller.update(force=True) calls mj_forward, so integration inputs
        # must be restored after controller refresh.
        _, data = self.sim_view._live_mj()
        for name in _SIM_CONTINUATION_FIELDS:
            key = f"sim.{name}"
            if key not in snapshot:
                continue
            target = np.asarray(getattr(data, name))
            value = np.asarray(snapshot[key])
            if target.shape != value.shape:
                raise ValueError(
                    f"{key} shape {value.shape} != restored model shape {target.shape}"
                )
            target[...] = value

        for name in ("cur_time", "timestep", "done"):
            key = f"env.{name}"
            if key in snapshot:
                self._assign_runtime_value(raw, name, snapshot[key])

        if restore_observables:
            observables = getattr(raw, "_observables", {})
            for observable_name, observable in observables.items():
                prefix = f"observable.{observable_name}."
                for key, value in snapshot.items():
                    if key.startswith(prefix):
                        self._assign_runtime_value(observable, key[len(prefix):], value)
            cache = {
                key[len("obs_cache."):]: np.asarray(value).copy()
                for key, value in snapshot.items() if key.startswith("obs_cache.")
            }
            if cache or any(key.startswith("obs_cache.") for key in snapshot):
                raw._obs_cache = cache
        captured_observation = {
            key[len("observation."):]: np.asarray(value).copy()
            for key, value in snapshot.items() if key.startswith("observation.")
        }
        observation = (
            captured_observation
            if restore_observables and captured_observation
            else raw._get_observations()
            if hasattr(raw, "_get_observations")
            else dict(getattr(self.sim_view, "_obs", {}))
        )
        if hasattr(self.sim_view, "update"):
            self.sim_view.update(observation, done=bool(getattr(raw, "done", False)))
        return observation

    def reset_to_exact(self, flat_state: np.ndarray, obstacles: list[dict] | None = None,
                       movable_objects: list[dict] | None = None,
                       model_xml: str | None = None):
        """Rebuild a requested injected scene and restore its *expanded* state.

        In contrast to :meth:`reset_to`, ``flat_state`` already contains the
        qpos/qvel slots of all injected movable objects.  This is intentionally a
        separate method so an expanded pre-crash state can never be accidentally
        spliced twice.
        """

        if model_xml is not None:
            if obstacles or movable_objects:
                raise ValueError("model_xml cannot be combined with obstacle reinjection")
            self.env.reset_from_xml_string(model_xml)
            continuation_model_xml = str(model_xml)
        else:
            self.env.reset()
            continuation_model_xml = str(self.env.sim.model.get_xml())
        if model_xml is None and (obstacles or movable_objects):
            xml = self.env.sim.model.get_xml()
            xml = inject_obstacles_xml(xml, obstacles or [])
            xml = inject_movable_objects_xml(xml, movable_objects or [])
            self.env.reset_from_xml_string(xml)
            continuation_model_xml = xml
        model = self._raw_model()
        expected = 1 + int(model.nq) + int(model.nv)
        state = np.asarray(flat_state, dtype=np.float64)
        if state.shape != (expected,):
            raise ValueError(
                f"exact state shape {state.shape} != ({expected},) for rebuilt model"
            )
        obs = self.env.set_init_state(state)
        self._continuation_model_xml = continuation_model_xml
        self.sim_view.peak_force = 0.0
        self.sim_view.update(obs, done=False)
        return obs

    def restore_to_exact_in_place(
        self,
        flat_state: np.ndarray,
        runtime_state: dict[str, np.ndarray],
    ):
        """Rewind the existing MjModel/MjData without rebuilding the scene.

        This is the in-memory arm of the continuation diagnostic.  Comparing it
        with :meth:`reset_to_exact` using captured ``model_xml`` separates a
        dynamic snapshot defect from a model reconstruction defect.
        """

        model = self._raw_model()
        state = np.asarray(flat_state, dtype=np.float64)
        expected = 1 + int(model.nq) + int(model.nv)
        if state.shape != (expected,):
            raise ValueError(f"exact state shape {state.shape} != ({expected},)")
        self.env.set_state(state)
        self.env.sim.forward()
        observation = self.restore_controller_state(runtime_state)
        self.sim_view.peak_force = 0.0
        self.sim_view.update(observation, done=False)
        return observation

    def _raw_model(self):
        sim = self.env.sim
        return getattr(sim.model, "_model", sim.model)

    @staticmethod
    def _splice_movable_state(init_state, nq0, nv0, movables):
        """Insert each movable object's initial [x,y,z, 1,0,0,0] qpos and zero qvel at the
        end of the qpos / qvel blocks (their free joints are the last joints). Returns the
        flat [time, qpos(nq0+7M), qvel(nv0+6M)] vector for set_state_from_flattened."""
        s = np.asarray(init_state, dtype=np.float64)
        t, qpos, qvel = s[:1], s[1:1 + nq0], s[1 + nq0:1 + nq0 + nv0]
        add_q, add_v = [], []
        for o in movables:
            px, py, pz = o["pos"]
            add_q.extend([px, py, pz, 1.0, 0.0, 0.0, 0.0])     # identity quat -> upright
            add_v.extend([0.0] * 6)
        return np.concatenate([t, qpos, np.asarray(add_q), qvel, np.asarray(add_v)])

    @staticmethod
    def _strip_movable_state(expanded_state, nq, nv, movable_count):
        """Inverse of ``_splice_movable_state`` for append-last free joints.

        ``nq`` and ``nv`` describe the currently injected model.  Removing the
        final ``7*M`` qpos and ``6*M`` qvel entries preserves the exact robot and
        original task-object state.  A different matched glass scene can then be
        rebuilt with :meth:`reset_to`.
        """

        count = int(movable_count)
        if count < 0 or 7 * count > nq or 6 * count > nv:
            raise ValueError("invalid movable_count for model dimensions")
        state = np.asarray(expanded_state, dtype=np.float64)
        expected = 1 + int(nq) + int(nv)
        if state.shape != (expected,):
            raise ValueError(f"expanded state shape {state.shape} != ({expected},)")
        nq0, nv0 = int(nq) - 7 * count, int(nv) - 6 * count
        qpos = state[1:1 + int(nq)]
        qvel = state[1 + int(nq):]
        return np.concatenate([state[:1], qpos[:nq0], qvel[:nv0]])

    def dummy_action(self):
        if self._model_family == "pi0":
            return [0, 0, 0, 0, 0, 0, -1]   # LIBERO no-op (same as get_libero_dummy_action)
        from experiments.robot.libero.libero_utils import get_libero_dummy_action
        return get_libero_dummy_action(self._model_family)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.sim_view.update(obs, done=done)
        return obs, reward, done, info

    def episode_terminated(self) -> bool:
        """Return robosuite's internal terminal flag, including its horizon.

        LIBERO overwrites the ``done`` returned by robosuite with task success.
        Consequently a horizon termination can be returned as ``done=False``;
        the following step then raises ``executing action in terminated
        episode``.  Long structured options use this separate lifecycle signal
        to label the rollout safe noncompletion without confusing horizon
        exhaustion with task success.
        """

        return bool(getattr(self._raw_env(), "done", False))

    def render(self, obs, resize_size):
        from experiments.robot.libero.libero_utils import get_libero_image
        return get_libero_image(obs, resize_size)

    def policy_observation(self, obs, resize_size):
        """Build the dict the active VLA's get_action expects (image[s] + proprio state).

        Auto-adapts to whichever repo is on sys.path (base OpenVLA vs OpenVLA-OFT), which
        have DIFFERENT `get_libero_image` signatures:
          * base OpenVLA:  get_libero_image(obs, resize_size)  -> already-resized image,
                           single third-person view, no wrist.
          * OpenVLA-OFT:   get_libero_image(obs)               -> raw image; resize
                           separately, AND add a wrist camera (num_images_in_input=2).
        We detect by trying the 2-arg (base) call; a TypeError means the OFT 1-arg helper
        is active, so we fall back to the OFT path and attach `wrist_image`.

        π0 (openpi/JAX) uses its own torch-free path (raw 180°-rotated cameras; the Pi0Policy
        wrapper does resize_with_pad→224 + tokenization).
        """
        if self._model_family == "pi0":
            return self._policy_observation_pi0(obs)
        from experiments.robot.libero.libero_utils import get_libero_image, quat2axisangle
        wrist = None
        try:
            img = get_libero_image(obs, resize_size)          # base OpenVLA signature
        except TypeError:
            from experiments.robot.libero.libero_utils import get_libero_wrist_image
            from experiments.robot.openvla_utils import resize_image_for_policy
            img = resize_image_for_policy(get_libero_image(obs), resize_size)   # OFT
            wrist = resize_image_for_policy(get_libero_wrist_image(obs), resize_size)
        out = {
            "full_image": img,
            "state": np.concatenate(
                (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
            ),
        }
        if wrist is not None:
            out["wrist_image"] = wrist        # OFT reads any key containing "wrist"
        return out

    def _policy_observation_pi0(self, obs):
        """openpi LIBERO obs contract (examples/libero/main.py:115-140): RAW 180°-rotated
        third-person + wrist cameras (the Pi0Policy wrapper applies resize_with_pad→224 itself),
        plus the same 8-dim proprio state = eef_pos + axisangle(eef_quat) + gripper_qpos.
        No experiments.robot import -> torch-free (keeps the JAX env clean)."""
        return {
            "full_image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
            "wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
            "state": np.concatenate((
                obs["robot0_eef_pos"],
                _native_quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )),
        }
