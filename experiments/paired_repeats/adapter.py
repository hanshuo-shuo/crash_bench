"""Checked instrumentation of main_aegis.py; never edit the upstream checkout."""
import hashlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
from run_safelibero import EXPECTED_SOURCE_SHA256, replace_once

def adapt(source, method, idle=False, settle_only=False):
    if hashlib.sha256(source.encode()).hexdigest() != EXPECTED_SOURCE_SHA256:
        raise RuntimeError('Unexpected scientific source')
    if method == 'nominal':
        start = source.index('    from groundingdino.util.inference import load_model')
        end = source.index('    # Start evaluation\n', start)
        source = source[:start] + '    model_groundingdino = None\n' + source[end:]
        start = source.index('            # Detect obstacles\n')
        end = source.index('            # print("Joint names (qpos):"', start)
        source = source[:start] + '            flag_safety_control = False\n            t = 0\n' + source[end:]
    elif method != 'aegis':
        raise ValueError(method)
    else:
        source = replace_once(source, 'model_groundingdino = load_model(CONFIG_PATH, CHECKPOINT_PATH)',
                              'model_groundingdino = _load_detector(load_model, CONFIG_PATH, CHECKPOINT_PATH)')
    source = replace_once(source, '            # Reset environment\n',
                          '            _begin(client)\n            # Reset environment\n')
    # Settling errors are a failed self-check, never a scientific outcome.
    anchor = '                    logging.error(f"Caught exception: {e}")'
    if source.count(anchor) != 2:
        raise RuntimeError('Exception anchors changed')
    source = source.replace(anchor, '                    raise RuntimeError("Settling failed") from e', 1)
    source = source.replace(anchor, '                    _caught(e, t)\n'+anchor, 1)
    before = '            # Detect obstacles\n' if method == 'aegis' else '            flag_safety_control = False\n'
    source = replace_once(source, before, '            _settled(env, obs, t)\n'+before)
    source = replace_once(source, '            initial_obstacle_pos = obs[obstacle_name + "_pos"]',
                          '            initial_obstacle_pos = obs[obstacle_name + "_pos"].copy()\n'
                          '            _ready(obstacle_name, initial_obstacle_pos, flag_safety_control)')
    source = replace_once(source, '                        obs, reward, done, info = env.step(action_input.tolist()) # Crucial step',
                          '                        _candidate(action, action_input, t, str(prob.status))\n'
                          '                        obs, reward, done, info = env.step(action_input.tolist()) # Crucial step\n'
                          '                        _after(obs, done, t)')
    source = replace_once(source, '                        obs, reward, done, info = env.step(action.tolist()) # Crucial step',
                          '                        _candidate(action, action, t, "off")\n'
                          '                        obs, reward, done, info = env.step(action.tolist()) # Crucial step\n'
                          '                        _after(obs, done, t)')
    source = replace_once(source, '            # Log current results\n',
                          '            _finish(bool(done), bool(collide_flag), str(video_path))\n            # Log current results\n')
    if idle or settle_only:
        source = replace_once(source, '    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)', '    client = None')
    else:
        source = replace_once(source, '    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)',
                              '    client = _client(args.host, args.port)')
    if idle:
        start = source.index('                    if not action_plan:')
        end = source.index('                    t3 = time.time()', start)
        source = source[:start] + '                    action = np.array(LIBERO_DUMMY_ACTION)\n' + source[end:]
        source = replace_once(source, '                    if done:\n', '                    if done and False:  # Idle check always executes 300 steps.\n')
        start = source.index('            imageio.mimwrite(')
        end = source.index('            _finish(', start)
        source = source[:start] + source[end:]
    return source
