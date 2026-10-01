"""Hooks around the exact pinned AEGIS loop; retains the six-axis QP."""
import hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from run_safelibero import EXPECTED_SOURCE_SHA256,replace_once

def adapt(source):
 if hashlib.sha256(source.encode()).hexdigest()!=EXPECTED_SOURCE_SHA256:raise RuntimeError('Source changed')
 start=source.index('    from groundingdino.util.inference import load_model')
 end=source.index('    # Start evaluation\n',start)
 source=source[:start]+source[end:]
 start=source.index('            # Detect obstacles\n');end=source.index('            t = 0\n',start)
 source=source[:start]+'''            _settled(env, obs, locals())
            img_out_dir = out_dir/f"{episode_idx}"
            img_out_dir.mkdir(parents=True, exist_ok=True)
            p2, R2, Q2_diag, flag_safety_control = _geometry(env, obs, task_description, img_out_dir)
            if flag_safety_control:
                z_fixed = p2 - p1
                z_fixed /= np.linalg.norm(z_fixed)
                dt = 0.05
'''+source[end:]
 source=replace_once(source,'    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)','    client = _client(args.host, args.port)')
 source=replace_once(source,'            # Reset environment\n','            _begin(client)\n            # Reset environment\n')
 source=replace_once(source,'            initial_obstacle_pos = obs[obstacle_name + "_pos"]','            initial_obstacle_pos = obs[obstacle_name + "_pos"].copy()\n            _ready(obstacle_name, initial_obstacle_pos)')
 source=replace_once(source,'                    # Get preprocessed image\n','                    _checkpoint(locals())\n                    # Get preprocessed image\n')
 source=replace_once(source,'                        action_chunk = client.infer(element)["actions"]','                        element = _element(element, t)\n                        action_chunk = client.infer(element)["actions"]')
 source=replace_once(source,'                    action = action_plan.popleft()','                    action = _next(action_plan.popleft(), locals())')
 source=replace_once(source,'                    if flag_safety_control:\n','                    if _filter(flag_safety_control, t):\n')
 source=replace_once(source,'                        obs, reward, done, info = env.step(action_input.tolist()) # Crucial step','                        _candidate(action, action_input, t, str(prob.status), locals())\n                        obs, reward, done, info = env.step(action_input.tolist()) # Crucial step\n                        _after(obs, done, t)')
 source=replace_once(source,'                        obs, reward, done, info = env.step(action.tolist()) # Crucial step','                        _candidate(action, action, t, "off", locals())\n                        obs, reward, done, info = env.step(action.tolist()) # Crucial step\n                        _after(obs, done, t)')
 source=replace_once(source,'            # Log current results\n','            _finish(bool(done), bool(collide_flag), str(video_path))\n            # Log current results\n')
 anchor='                    logging.error(f"Caught exception: {e}")'
 source=source.replace(anchor,'                    raise RuntimeError("Settling failed") from e',1)
 source=source.replace(anchor,'                    _caught(e, t)\n'+anchor,1)
 source=replace_once(source,'            while t < max_steps:\n','            while t < _horizon(max_steps):\n')
 return source
