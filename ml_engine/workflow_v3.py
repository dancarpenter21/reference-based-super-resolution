from __future__ import annotations

import json
from pathlib import Path

from .indexed_media import build_index, cancelled
from .review_v3 import indexes, propose_initial, edit_manifest, training_pairs
from .restoration_v3 import geometry, train_adaptation, render
from .media import ffmpeg_executable
from .inference.composition import _run_cancellable
from .weights import ensure_pretrained


def analyze(low_path,reference_path,job_dir,update=None,cancel=None,output_resolution='reference',**kwargs):
    root=Path(job_dir);root.mkdir(parents=True,exist_ok=True);idx={}
    for n,(source,path) in enumerate((('low',low_path),('reference',reference_path))):
        def progress(count,total):
            if update:update('analyzing',.01+n*.025+.02*min(1,count/(total or max(1,count*2))),f'Indexing {source}: frame {count+1}',None)
        idx[source]=build_index(path,root/'indexes'/source,progress,cancel)
        cancelled(cancel)
        if update:update('analyzing',.02+n*.025,f'Building {source} playback proxy',None)
        code,error=_run_cancellable([ffmpeg_executable(),'-y','-v','error','-i',str(path),'-vf',"scale='min(640,iw)':-2",'-an','-c:v','libx264','-preset','veryfast','-crf','27','-fps_mode','passthrough','-force_key_frames','expr:gte(t,n_forced*1)','-movflags','+faststart',str(root/f'{source}-proxy.mp4')],cancel)
        if code:raise ValueError(error[-2000:])
    if update:update('analyzing',.06,'Finding candidate shared sections',None)
    review={'schema_version':3,'revision':1,'locks':[],'ranges':propose_initial(idx,cancel),
            'geometry':{'framing':'fit','crops':{}},'stabilization':True,'history':[],
            'media':{s:{**i.meta,'duration':float(i.duration)} for s,i in idx.items()},'summary':{}}
    names=root/'source-names.json'
    review['source_names']=json.loads(names.read_text()) if names.exists() else {'low':Path(low_path).name,'reference':Path(reference_path).name}
    w,h=geometry(idx,review,output_resolution)
    review['output_geometry']={'width':w,'height':h,'timing':'source','choice':output_resolution}
    (root/'match-review.json').write_text(json.dumps(review,indent=2))
    return review


def preview_clips(clips,idx,training):
    samples=[]
    held=training.get('split_manifest',{}).get('validation',[])
    if held:
        start=held[0]['low_frame'];end=min(len(idx['low']),idx['low'].nearest(idx['low'].times[start]+2)+1)
        samples.append(('Held-out matching footage',{'source':'low','source_range':{'start_frame':start,'end_frame':end},'role':'held_out'}))
    low=next((c for c in clips if c['source']=='low'),None)
    if low:samples.append(('Supplemental restoration',low))
    if len(clips)>1:samples.append(('Edit transition',clips[max(0,len(clips)//2-1)]))
    if not samples:samples=[('Reference footage',clips[0])]
    result=[]
    for label,c in samples:
        i=idx[c['source']];a=c['source_range']['start_frame'];z=c['source_range']['end_frame']
        b=min(z,max(a+1,i.nearest(i.times[a]+2)))
        start,end=i.interval(a,b)
        result.append({**c,'label':label,'source_range':{'start_frame':a,'end_frame':b},'frame_count':b-a,'source_start_seconds':float(start),'source_end_seconds':float(end),'output_duration_seconds':float(end-start)})
        if label=='Edit transition':
            pos=clips.index(c)
            if pos+1<len(clips):
                following=clips[pos+1];j=idx[following['source']];x=following['source_range']['start_frame'];y=min(following['source_range']['end_frame'],max(x+1,j.nearest(j.times[x]+1)))
                begin,finish=j.interval(x,y)
                result.append({**following,'label':'After cut','source_range':{'start_frame':x,'end_frame':y},'frame_count':y-x,'source_start_seconds':float(begin),'source_end_seconds':float(finish),'output_duration_seconds':float(finish-begin)})
    return result


def process(job,update,cancel,export=False):
    root=Path(job['job_dir']);review=job['review_data'];idx=indexes(root)
    clips=edit_manifest(review,idx);size=geometry(idx,review,job['output_resolution'])
    (root/'edit-manifest.json').write_text(json.dumps({'clips':clips},indent=2))
    preparation=root/'preparation-v3.json'
    if export:
        prepared=json.loads(preparation.read_text())
        if prepared['revision']!=review['revision']:raise ValueError('Review changed; rebuild the quality preview')
        choice=review.get('render_method','selected');method='lanczos' if choice=='lanczos' else 'model'
        checkpoint=prepared['checkpoints'].get(choice)
        if method=='model' and not checkpoint and any(c['source']=='low' for c in clips):raise ValueError('Requested checkpoint is unavailable')
        result=render(clips,idx,review,size,root/'result.mp4',checkpoint,method,
                      lambda p,m,v:update('upscaling',.5+p*.49,m,v),cancel)
        report={'schema_version':3,'media':review['media'],'review_revision':review['revision'],'training':prepared['training'],'composition':result,'selected_method':choice,'output':str(root/'result.mp4')}
        (root/'report.json').write_text(json.dumps(report,indent=2));return report
    pairs=training_pairs(review,root)
    training={'selected':'reference','reason':'No supplemental frames require restoration','split_manifest':{}}
    checkpoints={};pretrained=None
    if any(c['source']=='low' for c in clips):
        update('training',.08,'Verifying spatial alignment and independent evaluation blocks',None)
        pretrained=ensure_pretrained()
        selected,training=train_adaptation(pretrained,root,pairs,idx,review,job['preset'],lambda p,m,v:update('training',.08+p*.32,m,v),cancel)
        checkpoints={'selected':str(selected),'pretrained':str(pretrained)}
        if training.get('candidate_checkpoint'):checkpoints['adapted']=training['candidate_checkpoint']
    (root/'training.json').write_text(json.dumps(training,indent=2))
    samples=preview_clips(clips,idx,training);artifacts={}
    choices=['lanczos']+(['pretrained','selected'] if pretrained else [])
    if checkpoints.get('adapted') and checkpoints['adapted']!=checkpoints.get('selected'):choices.append('adapted')
    for choice in choices:
        cancelled(cancel);name=f'preview-{choice}.mp4'
        render(samples,idx,review,size,root/name,checkpoints.get(choice),'lanczos' if choice=='lanczos' else 'model',lambda p,m,v:update('upscaling',.40+.09*p,f'Quality preview · {choice} · {m}',v),cancel)
        artifacts[choice]=name
    prepared={'revision':review['revision'],'training':training,'checkpoints':checkpoints,'previews':artifacts,'samples':samples,'output_geometry':{'width':size[0],'height':size[1]},'output_duration_seconds':sum(c['output_duration_seconds'] for c in clips)}
    preparation.write_text(json.dumps(prepared,indent=2));return prepared
