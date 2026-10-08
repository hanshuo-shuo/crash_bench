"""Observation hooks around pinned evaluation; scientific QP and scoring stay intact."""
from common import config
import hashlib


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('Pinned hook changed: '+old[:90])
    return text.replace(old, new)


def adapt(source, method):
    if hashlib.sha256(source.encode()).hexdigest() != config()['source_sha256']:
        raise RuntimeError('Unexpected upstream evaluator')
    if method == 'nominal':
        start = source.index('    from groundingdino.util.inference import load_model')
        end = source.index('    # Start evaluation\n', start)
        source = source[:start]+'    model_groundingdino = None\n'+source[end:]
        start = source.index('            # Detect obstacles\n')
        end = source.index('            # print("Joint names (qpos):"', start)
        source = source[:start]+'            flag_safety_control = False\n            t = 0\n'+source[end:]
    elif method != 'aegis':
        raise ValueError(method)
    else:
        source = replace_once(source, 'model_groundingdino = load_model(CONFIG_PATH, CHECKPOINT_PATH)',
                              'model_groundingdino = observer.load_detector(load_model, CONFIG_PATH, CHECKPOINT_PATH)')
    source = replace_once(source, '    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)',
                          '    client = observer.client(args.host, args.port)')
    source = replace_once(source, '            # Reset environment\n', '            observer.begin(client)\n            # Reset environment\n')
    boundary = '            # Detect obstacles\n' if method == 'aegis' else '            flag_safety_control = False\n'
    source = replace_once(source, boundary, '            observer.settled(env, obs, t)\n'+boundary)
    source = replace_once(source, '            initial_obstacle_pos = obs[obstacle_name + "_pos"]',
                          '            initial_obstacle_pos = obs[obstacle_name + "_pos"].copy()\n'
                          '            observer.ready(env, obs, obstacle_name, flag_safety_control)')
    source = replace_once(source, '                    # Get preprocessed image\n', '                    observer.before(locals())\n                    # Get preprocessed image\n')
    source = replace_once(source, '                        obs, reward, done, info = env.step(action_input.tolist()) # Crucial step',
                          '                        observer.candidate(action, action_input, t, str(prob.status))\n'
                          '                        obs, reward, done, info = env.step(action_input.tolist()) # Crucial step\n'
                          '                        observer.after(obs, done, t)')
    source = replace_once(source, '                        obs, reward, done, info = env.step(action.tolist()) # Crucial step',
                          '                        observer.candidate(action, action, t, "off")\n'
                          '                        obs, reward, done, info = env.step(action.tolist()) # Crucial step\n'
                          '                        observer.after(obs, done, t)')
    source = replace_once(source, '            # Log current results\n',
                          '            observer.finish(bool(done), bool(collide_flag), str(video_path), locals())\n            # Log current results\n')
    anchor = '                    logging.error(f"Caught exception: {e}")'
    if source.count(anchor) != 2:
        raise RuntimeError('Exception hooks changed')
    source = source.replace(anchor, '                    raise RuntimeError("Settling failed") from e', 1)
    source = source.replace(anchor, '                    observer.caught(e, t)\n'+anchor, 1)
    return source
