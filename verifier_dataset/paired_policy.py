"""Existing native pi0.5 as an offline witness generator for a reciprocal task."""
import collections
import math
from .inventory import write
from .replay import clean
from .paired_inputs import receiver_phrase


OBJECT_WORDS={
    'milk_1':'the milk carton','butter_1':'the butter','chocolate_pudding_1':'the chocolate pudding',
    'cream_cheese_1':'the cream cheese','alphabet_soup_1':'the alphabet soup',
    'tomato_sauce_1':'the tomato sauce','ketchup_1':'the ketchup','bbq_sauce_1':'the barbecue sauce',
    'orange_juice_1':'the orange juice','salad_dressing_1':'the salad dressing',
    'cookies_1':'the cookies','wine_bottle_1':'the wine bottle',
    'porcelain_mug_1':'the white mug','white_yellow_mug_1':'the yellow and white mug',
    'glazed_rim_porcelain_ramekin_1':'the white ramekin'}


def prompt_for(goal):
    target=goal[1]
    if target not in OBJECT_WORDS:
        raise RuntimeError('No unambiguous audited natural-language object name for reciprocal policy')
    return 'put '+OBJECT_WORDS[target]+' '+receiver_phrase(goal).replace(' at the green destination','')


def policy_input(obs,prompt):
    import numpy as np
    from openpi_client import image_tools
    quat=obs['robot0_eef_quat'].copy();quat[3]=np.clip(quat[3],-1,1)
    den=np.sqrt(1-quat[3]*quat[3]);axis=np.zeros(3) if math.isclose(den,0) else quat[:3]*2*math.acos(quat[3])/den
    return {'observation/image':image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs['agentview_image'][::-1,::-1]),224,224)),
        'observation/wrist_image':image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs['robot0_eye_in_hand_image'][::-1,::-1]),224,224)),
        'observation/state':np.concatenate((obs['robot0_eef_pos'],axis,obs['robot0_gripper_qpos'])),
        'prompt':str(prompt)}


class Policy:
    def __init__(self,env,folder,source,context,goal,port,run_id):
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        self.env,self.folder,self.goal=env,folder,goal
        self.prompt=prompt_for(goal);self.queue=collections.deque();self.requests=0;self.run_id=run_id
        self.obs=env.env._get_observations(force_update=True)
        self.client=WebsocketClientPolicy('127.0.0.1',port)
        ack=self.client.infer({'__uncertainty_reset__':True,'run_id':run_id,
            'scene_id':source['source_id'],'seed':int(source['seed']),'enabled':False})
        if ack['run_id']!=run_id or ack['seed']!=source['seed'] or ack['requests']!=0:
            raise RuntimeError('Native policy reset identity mismatch')
        write(folder/'POLICY_CONFIG.json',dict(controller='existing_native_pi05_nominal',prompt=self.prompt,
            task_goal=goal,source_seed=source['seed'],remaining_horizon=context['remaining_horizon'],replan_steps=5,
            uncertainty_sampling=False,learned_intervention=False,api_calls=0,reset=clean(ack)))
        self.log=(folder/'POLICY_INFERENCES.jsonl').open('w')

    def observe(self,obs):self.obs=obs

    def commands(self,horizon):
        import numpy as np
        import json
        for step in range(1,horizon+1):
            if not self.queue:
                data=policy_input(self.obs,self.prompt);result=self.client.infer(data);self.requests+=1
                info=result['uncertainty'];actions=np.asarray(result['actions'])
                if info['run_id']!=self.run_id or info['infer_index']!=self.requests or actions.shape!=(10,7) or not np.isfinite(actions).all():
                    raise RuntimeError('Native policy stream or action shape mismatch')
                np.savez_compressed(self.folder/('policy_input_%03d.npz'%self.requests),
                    image=data['observation/image'],wrist_image=data['observation/wrist_image'],state=data['observation/state'],
                    prompt=data['prompt'],raw_actions=actions)
                self.log.write(json.dumps(clean(dict(step=step,prompt=self.prompt,metadata=info,raw_actions=actions)))+'\n');self.log.flush()
                self.queue.extend(actions[:5])
            yield dict(step=None,applied=self.queue.popleft().tolist(),physics_after_sha256=None,
                       command_kind='native_pi05_derived_task')

    def close(self):
        self.log.close();self.client._ws.close();self.env=None;self.obs=None
