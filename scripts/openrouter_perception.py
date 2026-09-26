#!/usr/bin/env python3
"""Exact-input GLM cache. Run API filling on a networked host, never in a GPU job."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

MODEL = 'z-ai/glm-4.5v'

def make_request(image_bytes, instruction, suite):
    prefer_list = ['yellow rectangular book', 'blue moka pot', 'red mug', 'white storage box', 'black wine bottle', 'red milk carton']
    if 'long' in suite:prefer_list.append('gray rectangular binder')
    prompt = f"The robot must follow this instruction: {instruction}. Based on both the instruction and the image, identify exactly one non-robot object that is most likely to obstruct the robot's motion during task execution. You must output a uniquely identifiable obstacle name including both color and object type, preferably from this list when applicable: {prefer_list}. Output only the object name, with no additional words."
    return {'model':MODEL,'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image_bytes).decode()}},{'type':'text','text':prompt}]}],'temperature':0.1,'top_p':0.1,'reasoning':{'enabled':True},'provider':{'only':['z-ai'],'allow_fallbacks':False,'require_parameters':True}}

def request_hash(request):
    return hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def export_request(image_bytes, instruction, suite, cache):
    request=make_request(image_bytes,instruction,suite);key=request_hash(request)
    directory=Path(cache)/key;directory.mkdir(parents=True,exist_ok=True)
    payload=json.dumps(request,indent=2)+'\n'
    path=directory/'request.json'
    if path.exists() and json.loads(path.read_text())!=request:raise RuntimeError('Request hash collision')
    path.write_text(payload);(directory/'input.png').write_bytes(image_bytes)
    return directory

def read_response(directory):
    directory=Path(directory);request=json.loads((directory/'request.json').read_text())
    if request_hash(request)!=directory.name:raise RuntimeError('Request hash mismatch')
    data=json.loads((directory/'response.json').read_text())
    if data['request_sha256']!=directory.name:raise RuntimeError('Response belongs to another input')
    result=data['response'];model=result.get('model','')
    if model.split(':')[0]!=MODEL:raise RuntimeError('Unexpected returned model: '+model)
    text=result['choices'][0]['message']['content']
    if not isinstance(text,str) or not text.strip():raise RuntimeError('Empty obstacle response')
    return text.replace('<|begin_of_box|>','').replace('<|end_of_box|>','').strip()

def fill(directory):
    directory=Path(directory)
    if (directory/'response.json').exists():return read_response(directory)
    request=json.loads((directory/'request.json').read_text())
    if request_hash(request)!=directory.name:raise RuntimeError('Request hash mismatch')
    key=os.environ.get('OPENROUTER_API_KEY')
    if not key:raise RuntimeError('OPENROUTER_API_KEY is required in this networked process; never put it in Git or logs')
    req=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=json.dumps(request).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=180) as response:result=json.load(response)
    if 'error' in result:raise RuntimeError('OpenRouter returned an API error; no cache accepted')
    record={'request_sha256':directory.name,'created_unix':time.time(),'endpoint':'https://openrouter.ai/api/v1/chat/completions','response':result}
    # Validate before atomic publication; a failed call must not become a cached obstacle.
    if result.get('model','').split(':')[0]!=MODEL or not result.get('choices') or not result['choices'][0].get('message',{}).get('content'):
        (directory/'failed_response.json').write_text(json.dumps(record,indent=2)+'\n')
        raise RuntimeError('Missing content or unexpected model; response retained as failed')
    tmp=directory/'response.tmp';tmp.write_text(json.dumps(record,indent=2)+'\n');tmp.replace(directory/'response.json')
    return read_response(directory)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('request_directory');args=p.parse_args()
    print(fill(args.request_directory))
