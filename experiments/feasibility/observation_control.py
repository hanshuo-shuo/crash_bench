"""Exact-scene RGB control; native images retained, no simulator state writes."""
import hashlib
import json
from pathlib import Path
import numpy as np

def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()

def render_model_signature(model):
    names = ['body_pos', 'body_quat', 'geom_rgba', 'mat_rgba', 'cam_pos', 'cam_quat',
             'cam_fovy', 'light_pos', 'light_dir', 'light_diffuse', 'light_ambient', 'light_specular']
    return hashlib.sha256(json.dumps({name: array_hash(getattr(model, name)) for name in names}, sort_keys=True).encode()).hexdigest()

def scene_signature(env, obs):
    d = env.sim.data
    values = {name: array_hash(getattr(d, name)) for name in
              ['qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'qfrc_applied', 'xfrc_applied',
               'mocap_pos', 'mocap_quat', 'act', 'cam_xpos', 'cam_xmat']}
    values['sim_state'] = array_hash(env.get_sim_state())
    values['render_model'] = obs['_render_model_signature']
    values['non_rgb_observation'] = {k: array_hash(v) for k, v in obs.items()
                                     if isinstance(v, np.ndarray) and not k.endswith('_image')}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()

class RGBControl:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)

    def apply(self, mapping, keys, signature, scope, evidence_path):
        directory = self.root / scope
        directory.mkdir(exist_ok=True)
        path = directory / (signature + '.npz')
        native = {key: np.asarray(mapping[key]).copy() for key in keys}
        for image in native.values():
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError('RGB control expects unchanged uint8 RGB camera/preprocessed images')
        reused = path.exists()
        if reused:
            with np.load(path, allow_pickle=False) as archive:
                saved = {key: archive[key].copy() for key in keys}
        else:
            saved = native
            # New roots are single-worker experiments; retain the first sample for every exact scene.
            np.savez_compressed(path, **saved)
        differences = {}
        changed = False
        for key in keys:
            if native[key].shape != saved[key].shape:
                raise ValueError('Same scene signature has a different camera shape')
            delta = np.abs(native[key].astype(np.int16) - saved[key].astype(np.int16))
            pixels = int(np.any(delta, axis=-1).sum())
            changed |= pixels > 0
            differences[key] = {'native_sha256': array_hash(native[key]), 'controlled_sha256': array_hash(saved[key]),
                                'changed_pixels': pixels, 'changed_channels': int(np.count_nonzero(delta)),
                                'max_channel_difference': int(delta.max())}
            mapping[key] = saved[key].copy()
        if changed:
            np.savez_compressed(evidence_path, **native)
        return {'scope': scope, 'scene_signature': signature, 'template': str(path),
                'template_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'reused': reused,
                'native_variant_saved': str(evidence_path) if changed else None, 'differences': differences}
