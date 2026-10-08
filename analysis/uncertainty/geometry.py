"""Read-only signed collision-geometry distances using separate MuJoCo data."""
import hashlib
import numpy as np

ARRAYS = ['qpos', 'qvel', 'act', 'ctrl', 'qacc', 'qacc_warmstart', 'qfrc_applied',
          'xfrc_applied', 'mocap_pos', 'mocap_quat', 'userdata']


def physical_hash(env):
    h = hashlib.sha256()
    h.update(np.asarray([env.sim.data.time], dtype='<f8').tobytes())
    for k in ARRAYS:
        h.update(np.asarray(getattr(env.sim.data, k)).tobytes())
    h.update(np.asarray(env.sim.model.body_pos).tobytes())
    h.update(np.asarray(env.sim.model.body_quat).tobytes())
    return h.hexdigest()


class Distance:
    def __init__(self, env, obstacle, maximum):
        import mujoco
        self.mujoco, self.env, self.maximum = mujoco, env, maximum
        self.model = env.sim.model._model
        self.shadow = mujoco.MjData(self.model)
        m = env.sim.model
        root = m.body_name2id(env.robots[0].robot_model.root_body)
        self.robot, self.protected = [], []
        for geom in range(m.ngeom):
            if not (m.geom_contype[geom] or m.geom_conaffinity[geom]):
                continue
            body = int(m.geom_bodyid[geom]); ancestor = body
            while ancestor:
                if ancestor == root:
                    self.robot.append(geom); break
                ancestor = int(m.body_parentid[ancestor])
            gn, bn = m.geom_id2name(geom) or '', m.body_id2name(body) or ''
            if gn.startswith(obstacle+'_') or bn.startswith(obstacle+'_'):
                self.protected.append(geom)
        if not self.robot or not self.protected or not hasattr(mujoco, 'mj_geomDistance'):
            raise RuntimeError('Signed geometry distance unavailable')

    def read(self):
        before = physical_hash(self.env)
        live = self.env.sim.data
        for name in ARRAYS:
            getattr(self.shadow, name)[:] = np.asarray(getattr(live, name))
        self.shadow.time = live.time
        self.mujoco.mj_forward(self.model, self.shadow)
        minimum, pair = self.maximum, None
        for a in self.robot:
            for b in self.protected:
                value = float(self.mujoco.mj_geomDistance(self.model, self.shadow, a, b, self.maximum, None))
                if value < minimum:
                    minimum, pair = value, [a, b]
        if pair is None or not np.isfinite(minimum):
            raise RuntimeError('Geometry distance saturated or unsupported')
        if before != physical_hash(self.env):
            raise RuntimeError('Read-only geometry changed live simulation')
        return minimum, pair
