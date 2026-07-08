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


def inject_obstacles_xml(xml: str, obstacles: list[dict]) -> str:
    """Insert static (jointless) obstacle bodies into a robosuite model XML string.

    Each obstacle: {"name", "pos":[x,y,z], "size":[sx,sy,sz], "type"="box", "rgba"=[...]}.
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
        sx, sy, sz = o["size"]
        gtype = o.get("type", "box")
        rgba = " ".join(str(v) for v in o.get("rgba", [0.85, 0.2, 0.2, 1.0]))
        blocks.append(
            f'<body name="{o["name"]}" pos="{px} {py} {pz}">'
            f'<geom name="{o["name"]}_g" type="{gtype}" size="{sx} {sy} {sz}" '
            f'rgba="{rgba}" group="1" contype="1" conaffinity="1"/></body>'
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
                 resolution: int = 256, openvla_root: str | None = None):
        add_openvla_to_path(openvla_root)   # None -> CRASHBENCH_OPENVLA_ROOT / base default
        from libero.libero import benchmark
        from experiments.robot.libero.libero_utils import get_libero_env

        suite = benchmark.get_benchmark_dict()[task_suite]()
        self.task = suite.get_task(task_id)
        self.task_suite = task_suite
        self.task_id = task_id
        self.env, self.task_description = get_libero_env(self.task, model_family, resolution=resolution)
        self.sim_view = LiberoSimView(self.env)
        self._model_family = model_family

    def default_init_states(self) -> np.ndarray:
        from libero.libero import benchmark
        suite = benchmark.get_benchmark_dict()[self.task_suite]()
        return suite.get_task_init_states(self.task_id)

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
            state = self._splice_movable_state(init_state, nq0, nv0, movable_objects or [])
            obs = self.env.set_init_state(state)               # set spliced state, no reset
        self.sim_view.peak_force = 0.0                         # reset impact tracker per episode
        self.sim_view.update(obs, done=False)
        return obs

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

    def dummy_action(self):
        from experiments.robot.libero.libero_utils import get_libero_dummy_action
        return get_libero_dummy_action(self._model_family)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.sim_view.update(obs, done=done)
        return obs, reward, done, info

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
        """
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
