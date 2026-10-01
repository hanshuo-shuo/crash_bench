"""CPU-only pixel/source probe of the failed first-input gate. No new episodes."""
import json,os,sys
from pathlib import Path
import numpy as np
from PIL import Image
BASE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(BASE/'scripts'))
from api_budget import atomic_json

def main(parent,output):
 parent=Path(parent);out=Path(output);out.mkdir()
 upstream=Path(os.environ['CB_UPSTREAM']);sys.path.insert(0,str(upstream/'openpi/packages/openpi-client/src'))
 from openpi_client import image_tools
 from runtime import array_hash
 a=parent/'runs/spatial_03_r00_nominal';b=parent/'runs/spatial_03_r00_raw'
 ca=json.loads((a/'checkpoint_000.json').read_text());cb=json.loads((b/'checkpoint_000.json').read_text())
 entries={}
 nominal=np.load(a/'first_policy_input.npz',allow_pickle=False)['observation/image']
 task='pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate'
 for view in ['agentview','backview']:
  png=b/'videos'/task/'ours_I/3'/(view+'.png')
  before=np.asarray(Image.open(png).convert('RGB'))
  original=before[::-1,::-1].copy()
  record={'detector_input_png':str(png),'detector_before_hash_in_original_orientation':array_hash(original),
   'nominal_observation_hash':ca['observation'][view+'_image'],'aegis_after_perception_hash':cb['observation'][view+'_image']}
  record['detector_before_matches_nominal']=record['detector_before_hash_in_original_orientation']==record['nominal_observation_hash']
  record['detector_before_matches_after']=record['detector_before_hash_in_original_orientation']==record['aegis_after_perception_hash']
  entries[view]=record
  if view=='agentview':
   resized=image_tools.convert_to_uint8(image_tools.resize_with_pad(before,224,224));diff=np.abs(nominal.astype(float)-resized.astype(float))
   record.update(resized_policy_image_sha256=array_hash(resized),nominal_policy_image_sha256=array_hash(nominal),
    changed_channels=int(np.count_nonzero(diff)),changed_pixels=int(np.any(diff,axis=-1).sum()),max_channel_difference=float(diff.max()),mean_absolute_difference=float(diff.mean()))
   import matplotlib
   matplotlib.use('Agg')
   import matplotlib.pyplot as plt
   fig,axes=plt.subplots(1,3,figsize=(11,4))
   for ax,image,title in zip(axes,[nominal,resized,np.clip(diff*8,0,255).astype(np.uint8)],['Nominal policy initial RGB','AEGIS detector input resized','Absolute difference x8']):
    ax.imshow(image);ax.set_title(title);ax.axis('off')
   fig.tight_layout();fig.savefig(out/'initial_rgb_difference.png',dpi=180);plt.close(fig)
 differences=[k for k in ca if k not in ['aegis','execution_fingerprint','observation'] and ca[k]!=cb[k]]
 result={'source_root':str(parent),'probe_job':os.environ['SLURM_JOB_ID'],'comparison_scope':'two fresh runs with same official state/seed before first action',
  'physical_controller_rng_difference_fields':differences,'changed_observation_fields':[k for k in ca['observation'] if ca['observation'][k]!=cb['observation'][k]],'views':entries}
 atomic_json(out/'OBSERVATION_PROBE.json',result)
 print(json.dumps(result,indent=2))
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
