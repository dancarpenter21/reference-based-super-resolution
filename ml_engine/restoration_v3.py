"""Geometry-preserving restoration, held-out evaluation, timestamp-aware composition."""
from __future__ import annotations

import json
import math
import time
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .config import PRESETS
from .indexed_media import FrameReader, cancelled, fit_frame, picture
from .models.generator import RRDBNet, load_weights
from .training.train import require_rocm, charbonnier, gradient_loss


def geometry(idx, review, resolution):
    ref=idx['reference']; crop=review.get('geometry',{}).get('crops',{}).get('reference')
    w,h=ref.meta['width'],ref.meta['height']
    if crop:w,h=crop[2]-crop[0],crop[3]-crop[1]
    w=round(w*Fraction(ref.meta['sample_aspect_ratio'].replace(':','/')))
    if resolution!='reference':
        target=int(resolution.removesuffix('p')); w=round(w*target/h);h=target
    factor=min(1,3840/w,2160/h)
    return max(2,int(w*factor)//2*2),max(2,int(h*factor)//2*2)


def tensor(frame,device):
    return torch.from_numpy(np.ascontiguousarray(frame[...,::-1])).permute(2,0,1).unsqueeze(0).float().to(device)/255


def predict(model,low,size):
    result=model(low)
    return F.interpolate(result,size=(size[1],size[0]),mode='bicubic',align_corners=False,antialias=True)


@torch.inference_mode()
def restore(model,frame,device,tile=256,cancel=None):
    h,w=frame.shape[:2]
    # Blend-free core tiles with a generous receptive-field halo.
    try:
        result=np.empty((h*2,w*2,3),np.uint8)
        for y in range(0,h,tile):
            for x in range(0,w,tile):
                cancelled(cancel)
                # Even origins preserve pixel-unshuffle phase.
                y0=max(0,y-64);x0=max(0,x-64);y1=min(h,y+tile+64);x1=min(w,x+tile+64)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                    out=model(tensor(frame[y0:y1,x0:x1],device)).clamp(0,1)
                out=(out[0].permute(1,2,0).float().cpu().numpy()[...,::-1]*255).round().astype(np.uint8)
                eh=min(tile,h-y);ew=min(tile,w-x)
                result[y*2:(y+eh)*2,x*2:(x+ew)*2]=out[(y-y0)*2:(y-y0+eh)*2,(x-x0)*2:(x-x0+ew)*2]
        return result
    except torch.cuda.OutOfMemoryError:
        if tile<=64:raise
        torch.cuda.empty_cache()
        return restore(model,frame,device,tile//2,cancel)


def stabilize(current,previous,current_lr,previous_lr,delta):
    if previous is None or delta>.15 or delta<=0:return current
    size=(max(16,current_lr.shape[1]//4),max(16,current_lr.shape[0]//4))
    a=cv2.cvtColor(cv2.resize(current_lr,size),cv2.COLOR_BGR2GRAY)
    b=cv2.cvtColor(cv2.resize(previous_lr,size),cv2.COLOR_BGR2GRAY)
    if np.mean(cv2.absdiff(a,b))>30:return current
    back=cv2.calcOpticalFlowFarneback(a,b,None,.5,3,15,3,5,1.1,0)
    forward=cv2.calcOpticalFlowFarneback(b,a,None,.5,3,15,3,5,1.1,0)
    yy,xx=np.mgrid[:size[1],:size[0]].astype(np.float32)
    rev=cv2.remap(forward,xx+back[...,0],yy+back[...,1],cv2.INTER_LINEAR)
    reliable=(np.linalg.norm(back+rev,axis=2)<1).astype(np.float32)
    h,w=current.shape[:2];flow=cv2.resize(back,(w,h));flow*=np.array([w/size[0],h/size[1]])
    yy,xx=np.mgrid[:h,:w].astype(np.float32);mx=xx+flow[...,0];my=yy+flow[...,1]
    warped=cv2.remap(previous,mx,my,cv2.INTER_LINEAR)
    diff=np.mean(cv2.absdiff(warped,current),axis=2)
    weight=(cv2.resize(reliable,(w,h))*(diff<15)*(mx>=0)*(mx<w-1)*(my>=0)*(my<h-1)*.1)[...,None]
    return np.clip(current*(1-weight)+warped*weight,0,255).astype(np.uint8)


def prepare_examples(pairs,idx,review,cancel=None):
    """Deterministic temporal split; nearby source frames cannot enter training."""
    if not pairs:return [],[],[],{'accepted':0,'rejected':0,'reason':'No approved real pairs'}
    ordered=sorted(pairs,key=lambda p:p['low_frame'])
    # Reserve whole two-second blocks spread across the low timeline.
    blocks={int(idx['low'].times[p['low_frame']]//2) for p in ordered}
    blocks=sorted(blocks)
    if len(blocks)<5:return [],[],[],{'accepted':0,'rejected':len(pairs),'reason':'Need at least five temporal blocks for independent evaluation'}
    validation_blocks=set(blocks[1::5]); test_blocks=set(blocks[3::5])
    reserve=[p for p in ordered if int(idx['low'].times[p['low_frame']]//2) in validation_blocks|test_blocks]
    # Reference time exclusions also prevent reuse through repeated/cadence matches.
    reserved={s:np.array(sorted(idx[s].times[p[f'{s}_frame']] for p in reserve)) for s in idx}
    groups=[[],[],[]]
    for p in ordered:
        block=int(idx['low'].times[p['low_frame']]//2)
        kind=1 if block in validation_blocks else 2 if block in test_blocks else 0
        if kind==0 and any(np.min(np.abs(reserved[s]-idx[s].times[p[f'{s}_frame']]))<=1 for s in idx):continue
        groups[kind].append(p)
    selected=[g[::max(1,math.ceil(len(g)/limit))] for g,limit in zip(groups,(96,16,16))]
    crops=review.get('geometry',{}).get('crops',{})
    examples=[[],[],[]];rejected=0; manifests=[[],[],[]]
    with FrameReader(idx['low']) as lr_reader,FrameReader(idx['reference']) as hr_reader:
        for group,entries in enumerate(selected):
            for p in entries:
                cancelled(cancel)
                lr=picture(lr_reader.read(p['low_frame']),idx['low'],crops.get('low'))
                hr=picture(hr_reader.read(p['reference_frame']),idx['reference'],crops.get('reference'))
                scale=min(hr.shape[0]/lr.shape[0],hr.shape[1]/lr.shape[1])
                if scale<1:rejected+=1;continue
                small_size=(min(320,lr.shape[1]),max(16,round(lr.shape[0]*min(320,lr.shape[1])/lr.shape[1])))
                lg=cv2.cvtColor(cv2.resize(lr,small_size),cv2.COLOR_BGR2GRAY).astype(np.float32)/255
                hg=cv2.cvtColor(cv2.resize(hr,small_size),cv2.COLOR_BGR2GRAY).astype(np.float32)/255
                warp=np.eye(2,3,dtype=np.float32)
                try:
                    score,warp=cv2.findTransformECC(lg,hg,warp,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,60,1e-5))
                    if score<.9 or np.max(np.abs(warp[:,:2]-np.eye(2)))>.15:rejected+=1;continue
                except cv2.error:rejected+=1;continue
                target=(round(lr.shape[1]*scale),round(lr.shape[0]*scale))
                hr=cv2.resize(hr,target,interpolation=cv2.INTER_LANCZOS4)
                warp[0,2]*=target[0]/small_size[0];warp[1,2]*=target[1]/small_size[1]
                hr=cv2.warpAffine(hr,warp,target,flags=cv2.INTER_LANCZOS4|cv2.WARP_INVERSE_MAP)
                mask=cv2.warpAffine(np.ones((target[1],target[0]),np.uint8),warp,target,flags=cv2.INTER_NEAREST|cv2.WARP_INVERSE_MAP)
                patch=min(96,lr.shape[0]//2,lr.shape[1]//2)
                rng=np.random.default_rng(p['low_frame'])
                accepted=False
                for _ in range(12):
                    x=int(rng.integers(0,lr.shape[1]-patch+1));y=int(rng.integers(0,lr.shape[0]-patch+1))
                    hx,hy=round(x*scale),round(y*scale);hp=round(patch*scale)
                    high=hr[hy:hy+hp,hx:hx+hp];valid=mask[hy:hy+hp,hx:hx+hp]
                    if high.shape[:2]!=(hp,hp) or not valid.all():continue
                    low=lr[y:y+patch,x:x+patch]
                    if np.mean(cv2.absdiff(cv2.resize(high,(patch,patch)),low))>25:continue
                    examples[group].append((low,high));manifests[group].append(p);accepted=True;break
                if not accepted:rejected+=1
    return *examples,{'accepted':sum(map(len,examples)),'rejected':rejected,'split_manifest':dict(zip(('train','validation','test'),manifests))}


def metrics(pred,target):
    a=pred.astype(np.float64)/255;b=target.astype(np.float64)/255
    mse=np.mean((a-b)**2)
    ma=cv2.GaussianBlur(a,(11,11),1.5);mb=cv2.GaussianBlur(b,(11,11),1.5)
    va=cv2.GaussianBlur(a*a,(11,11),1.5)-ma*ma;vb=cv2.GaussianBlur(b*b,(11,11),1.5)-mb*mb
    cov=cv2.GaussianBlur(a*b,(11,11),1.5)-ma*mb
    ssim=((2*ma*mb+.01**2)*(2*cov+.03**2))/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))
    return {'l1':float(np.mean(np.abs(a-b))),'psnr':float(-10*math.log10(max(mse,1e-10))),'ssim':float(np.mean(ssim))}


@torch.inference_mode()
def evaluate(model,examples,device,cancel=None):
    results=[]
    for low,high in examples:
        cancelled(cancel)
        if model is None:out=cv2.resize(low,(high.shape[1],high.shape[0]),interpolation=cv2.INTER_LANCZOS4)
        else:
            out=predict(model,tensor(low,device),(high.shape[1],high.shape[0])).clamp(0,1)
            out=(out[0].permute(1,2,0).cpu().numpy()[...,::-1]*255).round().astype(np.uint8)
        results.append(metrics(out,high))
    return {k:float(np.mean([r[k] for r in results])) for k in ('l1','psnr','ssim')} if results else None


def train_adaptation(pretrained,job_dir,pairs,idx,review,preset_name,progress=None,cancel=None,device=None):
    from .indexed_media import identity
    device=device or require_rocm(); preset=PRESETS[preset_name]
    directory=Path(job_dir)/'checkpoints-v3';directory.mkdir(exist_ok=True)
    train,val,test,report=prepare_examples(pairs,idx,review,cancel)
    report.update(selected='pretrained',elapsed_seconds=0,real_validation={},synthetic_validation=None)
    (directory/'split-manifest.json').write_text(json.dumps(report.get('split_manifest',{}),indent=2))
    if len(train)<4 or len(val)<2 or len(test)<2:
        report['reason']='Insufficient spatially verified, temporally independent real pairs; keeping pretrained'
        return Path(pretrained),report
    model=RRDBNet().to(device);load_weights(model,pretrained);model.eval()
    baseline=evaluate(model,val,device,cancel)
    report['real_validation']={'lanczos':evaluate(None,val,device,cancel),'pretrained':baseline}
    signature=__import__('hashlib').sha256(json.dumps({'sources':[i.meta['source_id'] for i in idx.values()], 'review':review['revision'],'geometry':review.get('geometry'), 'split':report.get('split_manifest'),'pretrained':identity(pretrained)},sort_keys=True).encode()).hexdigest()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=0)
    scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    step=0;elapsed=0.;best=baseline['l1'];patience=0;history=[]
    latest=directory/'latest.pth';best_path=directory/'best.pth'
    if latest.exists():
        state=torch.load(latest,map_location=device,weights_only=True)
        if state.get('signature')==signature:
            model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
            step=state['step'];elapsed=state['elapsed'];best=state['best'];history=state['history'];patience=state['patience']
            torch.set_rng_state(state['rng'].cpu())
        else:best_path.unlink(missing_ok=True)
    started=time.monotonic();last_save=started
    def save():
        state={'signature':signature,'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'step':step,'elapsed':elapsed+time.monotonic()-started,'best':best,'history':history,'patience':patience,'rng':torch.get_rng_state()}
        temp=latest.with_suffix('.partial');torch.save(state,temp);temp.replace(latest)
    try:
        while step<preset.max_steps and elapsed+time.monotonic()-started<preset.max_minutes*60 and patience<5:
            cancelled(cancel);model.train()
            low,high=train[step%len(train)]
            # Real supervision alternates with deterministic native-reference degradation.
            if step%2:
                rng=np.random.default_rng(step)
                low=cv2.resize(cv2.GaussianBlur(high,(0,0),float(rng.uniform(.2,.8))),(low.shape[1],low.shape[0]),interpolation=cv2.INTER_AREA)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                pred=predict(model,tensor(low,device),(high.shape[1],high.shape[0]));target=tensor(high,device)
                loss=charbonnier(pred,target)+.15*gradient_loss(pred,target)
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();step+=1
            if progress and step%10==0:progress(min(.99,(elapsed+time.monotonic()-started)/(preset.max_minutes*60)),f'Training step {step}',{'step':step,'train_loss':float(loss.item())})
            if step%preset.validation_every==0 or step==preset.max_steps:
                model.eval();score=evaluate(model,val,device,cancel);history.append({'step':step,**score})
                if score['l1']<best:
                    best=score['l1'];patience=0;torch.save({'params_ema':model.state_dict()},best_path)
                else:patience+=1
                save();last_save=time.monotonic()
            elif time.monotonic()-last_save>30:save();last_save=time.monotonic()
    finally:save()
    chosen=Path(pretrained)
    if best_path.exists():
        load_weights(model,best_path);model.eval();adapted=evaluate(model,val,device,cancel)
        report['real_validation']['adapted']=adapted
        if adapted['l1']<baseline['l1'] and adapted['psnr']>=baseline['psnr']+.1:
            chosen=best_path;report['selected']='adapted'
    load_weights(model,chosen);model.eval()
    report.update(history=history,elapsed_seconds=elapsed+time.monotonic()-started,test=evaluate(model,test,device,cancel),reason='Adaptation selected only after real validation improvement',candidate_checkpoint=str(best_path) if best_path.exists() else None)
    del model
    return chosen,report


def render(clips,idx,review,size,output,checkpoint=None,method='model',progress=None,cancel=None,device=None,model_factory=RRDBNet):
    output=Path(output);temp=output.with_name(f'.{output.stem}-video.mp4')
    model=None
    if any(c['source']=='low' for c in clips) and method=='model':
        device=device or require_rocm();model=model_factory().to(device);load_weights(model,checkpoint);model.eval()
    # Exact common clock; no accumulating rounding at edit boundaries.
    denominator=math.lcm(*(i.time_base.denominator for i in idx.values()))
    if denominator>2_000_000_000:raise ValueError('Source clocks need explicit normalization before MP4 export')
    tb=Fraction(1,denominator);cursor=Fraction(0);emitted=0
    total=sum(c['frame_count'] for c in clips);crops=review.get('geometry',{}).get('crops',{})
    framing=review.get('geometry',{}).get('framing','fit');durations={};started=time.monotonic()
    try:
        with av.open(str(temp),'w') as container:
            stream=container.add_stream('libx264',rate=30)
            stream.width,stream.height=size;stream.pix_fmt='yuv420p';stream.time_base=tb;stream.codec_context.time_base=tb
            stream.options={'crf':'18','preset':'medium','bf':'0'}
            def mux(packets):
                for packet in packets:
                    if packet.pts in durations:packet.duration=durations[packet.pts]
                    container.mux(packet)
            for clip in clips:
                source=clip['source'];index=idx[source];start=clip['source_range']['start_frame'];end=clip['source_range']['end_frame']
                previous=previous_lr=None;previous_time=None
                with FrameReader(index) as reader:
                    for ordinal in range(start,end):
                        cancelled(cancel)
                        frame=picture(reader.read(ordinal),index,crops.get(source))
                        if source=='low' and model is not None:
                            sr=restore(model,frame,device,cancel=cancel)
                            if review.get('stabilization',True):sr=stabilize(sr,previous,frame,previous_lr,0 if previous_time is None else index.times[ordinal]-previous_time)
                            previous,previous_lr,previous_time=sr,frame,index.times[ordinal];frame=sr
                        frame=fit_frame(frame,size,framing)
                        duration=index.rows[ordinal][2]*index.time_base
                        f=av.VideoFrame.from_ndarray(frame,format='bgr24');f.pts=int(cursor/tb);f.time_base=tb;f.duration=int(duration/tb)
                        durations[f.pts]=f.duration;mux(stream.encode(f));cursor+=duration;emitted+=1
                        if progress and (emitted%10==0 or emitted==total):
                            elapsed=time.monotonic()-started
                            progress(emitted/total,f'Frame {emitted} of {total}',{'frames':emitted,'total_frames':total,'fps':emitted/max(.001,elapsed),'eta_seconds':(total-emitted)*elapsed/emitted})
            mux(stream.encode())
        if any(i.meta['has_audio'] for i in idx.values()):
            from .inference.composition import _run_cancellable
            from .media import ffmpeg_executable
            filters=[];labels=[]
            for n,c in enumerate(clips):
                s=c['source'];label=f'a{n}';duration=c['output_duration_seconds']
                if idx[s].meta['has_audio']:
                    source_number=1 if s=='low' else 2
                    # Preserve source audio/video relative timestamps, including nonzero starts.
                    start=c['source_start_seconds'];end=c['source_end_seconds']
                    filters.append(f'[{source_number}:a:0]atrim=start={start:.12f}:end={end:.12f},asetpts=PTS-({start:.12f})/TB,aresample=48000:async=1:first_pts=0,aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=duration={duration:.12f}[{label}]')
                else:filters.append(f'anullsrc=r=48000:cl=stereo,atrim=duration={duration:.12f}[{label}]')
                labels.append(f'[{label}]')
            filters.append(f'{"".join(labels)}concat=n={len(clips)}:v=0:a=1[aout]')
            cmd=[ffmpeg_executable(),'-y','-v','error','-copyts','-i',str(temp),'-i',idx['low'].meta['path'],'-i',idx['reference'].meta['path'],'-filter_complex',';'.join(filters),'-map','0:v:0','-map','[aout]','-c:v','copy','-c:a','aac','-b:a','192k','-video_track_timescale',str(denominator),'-movflags','+faststart',str(output)]
            code,error=_run_cancellable(cmd,cancel)
            if code:raise RuntimeError(error[-2000:])
            temp.unlink()
        else:temp.replace(output)
    except BaseException:
        temp.unlink(missing_ok=True);output.unlink(missing_ok=True);raise
    return {'frames':emitted,'duration_seconds':float(cursor),'width':size[0],'height':size[1],'time_base':str(tb),'timing':'source','clips':clips,'elapsed_seconds':time.monotonic()-started}
