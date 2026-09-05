"""Explicit anchors, independently approved coverage, and inspectable correspondences."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import numpy as np

from .indexed_media import FrameIndex, cancelled


def indexes(job_dir):
    return {s: FrameIndex(Path(job_dir) / 'indexes' / s) for s in ('low', 'reference')}


def normalized(features):
    x = np.asarray(features, dtype=np.float32)
    x -= x.mean(axis=-1, keepdims=True)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-6)


def propose_initial(idx, cancel=None):
    from .alignment import _sequence_pairs
    low, ref = idx['low'], idx['reference']
    # Bound the global matrix independently of video length. Fine search uses original frames.
    step = max(1., max(float(low.duration), float(ref.duration))/1800)
    li = sorted({low.nearest(t) for t in np.arange(0, float(low.duration), step)})
    ri = sorted({ref.nearest(t) for t in np.arange(0, float(ref.duration), step)})
    sim = normalized(low.features[li]) @ normalized(ref.features[ri]).T
    cancelled(cancel)
    seeds = []
    for a,b in _sequence_pairs(sim):
        cancelled(cancel)
        if sim[a,b] < .88: continue
        i,j=li[a],ri[b]
        left=max(0,int(np.searchsorted(ref.times,ref.times[j]-.75)))
        right=min(len(ref),int(np.searchsorted(ref.times,ref.times[j]+.75)))
        scores=normalized(ref.features[left:right]) @ normalized(low.features[i:i+1])[0]
        if len(scores) and float(scores.max()) > .92:
            seeds.append((i,left+int(np.argmax(scores))))
    groups = []
    for a,b in seeds:
        if not groups or abs((low.times[a]-low.times[groups[-1][-1][0]])-(ref.times[b]-ref.times[groups[-1][-1][1]])) > .15 or low.times[a]-low.times[groups[-1][-1][0]] > step*2.5:
            groups.append([])
        groups[-1].append((a,b))
    return [{'id':str(uuid4()), 'low_range':[g[0][0],g[-1][0]+1],
             'reference_range':[g[0][1],g[-1][1]+1], 'status':'proposed', 'origin':'automatic', 'pairs_file':None}
            for g in groups if len(g)>=2]


def validate_ranges(ranges, idx):
    last = {'low':0,'reference':0}
    for r in sorted(ranges, key=lambda r:r['low_range'][0]):
        for s in last:
            a,b = r[f'{s}_range']
            if not isinstance(a,int) or not isinstance(b,int) or not last[s] <= a < b <= len(idx[s]):
                raise ValueError('Ranges must be chronological and nonoverlapping in both sources')
            last[s] = b


def correspondence(r, idx, locks, cancel=None):
    low, ref = idx['low'], idx['reference']
    a,z = r['low_range']; b,y = r['reference_range']
    anchors = [(a,b),(z-1,y-1)] + [(p['low_frame'],p['reference_frame']) for p in locks
                if a <= p['low_frame'] < z and b <= p['reference_frame'] < y]
    anchors = sorted(set(anchors))
    for first, second in zip(anchors, anchors[1:]):
        if second[0] <= first[0] or second[1] <= first[1]:
            raise ValueError('Locked pairs conflict with the range endpoints or chronological order')
    manual = {p['low_frame']:p['reference_frame'] for p in locks if a <= p['low_frame'] < z}
    if any(not b <= v < y for v in manual.values()):
        raise ValueError('A locked pair is outside this proposed reference range; remove the proposal and create a range from your anchors')
    af = np.array([low.times[v[0]] for v in anchors]); bf = np.array([ref.times[v[1]] for v in anchors])
    rows = []; previous = b-1
    ref_features = normalized(ref.features[b:y])
    low_features = normalized(low.features[a:z])
    for offset,i in enumerate(range(a,z)):
        cancelled(cancel)
        estimate = np.interp(low.times[i], af, bf)
        start = max(b, previous, int(np.searchsorted(ref.times,estimate-.25)))
        end = min(y, int(np.searchsorted(ref.times,estimate+.25,side='right')))
        score=0.; j=None; status='missing'
        if i in manual:
            j=manual[i]; score=1.; status='manual'
        elif end > start:
            scores = ref_features[start-b:end-b] @ low_features[offset]
            pos = int(np.argmax(scores)); j=start+pos; score=float(scores[pos])
            # Similarity is evidence, never presented as a calibrated probability.
            margin = float(score-np.partition(scores,-2)[-2]) if len(scores)>1 else 1.
            status = 'verified' if score >= .96 and margin >= .005 and j > previous else 'approximate'
            if score < .80: j=None; status='missing'
        if j is not None: previous=j
        rows.append({'low_frame':i,'reference_frame':j,'status':status,'score':score,
                     'eligible':status in ('manual','verified')})
    # Reusing a reference frame is ambiguous supervision, including the first reuse.
    counts={}
    for p in rows:
        if p['reference_frame'] is not None: counts[p['reference_frame']]=counts.get(p['reference_frame'],0)+1
    for p in rows:
        if counts.get(p['reference_frame'],0)>1 and p['status']!='manual':
            p.update(status='approximate',eligible=False)
    return rows


def read_pairs(job_dir, r):
    return json.loads((Path(job_dir)/r['pairs_file']).read_text()) if r.get('pairs_file') else []


def write_pairs(job_dir, r, rows):
    directory=Path(job_dir)/'correspondences'; directory.mkdir(exist_ok=True)
    filename=f'correspondences/{uuid4()}.json'
    (Path(job_dir)/filename).write_text(json.dumps(rows))
    r['pairs_file']=filename
    r['counts']={k:sum(p['status']==k for p in rows) for k in ('manual','verified','approximate','missing')}
    r['training_pairs']=sum(p['eligible'] for p in rows)


def edit_review(review, payload, job_dir):
    idx=indexes(job_dir); value=deepcopy(review); operation=payload.get('operation')
    history=value.pop('history',[])
    snapshot=deepcopy(value)
    if operation=='undo':
        if not history: raise ValueError('There is no edit to undo')
        restored=history.pop(); restored['history']=history
        return restored
    if operation in ('lock_pair','remove_pair'):
        a=payload.get('low_frame'); b=payload.get('reference_frame')
        idx['low'].frame(a); idx['reference'].frame(b)
        locks=value['locks']
        old = next((p for p in locks if p['id']==payload.get('replace_id')),None)
        if operation=='lock_pair':
            locks=[p for p in locks if p['low_frame']!=a and p['id']!=payload.get('replace_id')]
            locks.append({'id':str(uuid4()),'low_frame':a,'reference_frame':b})
            locks.sort(key=lambda p:p['low_frame'])
            if any(q['reference_frame']<=p['reference_frame'] for p,q in zip(locks,locks[1:])):
                raise ValueError('This pair crosses or repeats a saved anchor; edit the conflicting pair first')
        else: locks=[p for p in locks if (p['low_frame'],p['reference_frame'])!=(a,b)]
        value['locks']=locks
        for r in value['ranges']:
            if r['low_range'][0]<=a<r['low_range'][1] or (old and r['low_range'][0]<=old['low_frame']<r['low_range'][1]):
                r.update(status='proposed',pairs_file=None); r.pop('counts',None)
    elif operation=='create_range':
        a,b=payload.get('first'),payload.get('last')
        first=next((p for p in value['locks'] if p['id']==a),None)
        last=next((p for p in value['locks'] if p['id']==b),None)
        if not first or not last or first['low_frame']>=last['low_frame']:
            raise ValueError('Choose two saved pairs in chronological order')
        r={'id':str(uuid4()),'low_range':[first['low_frame'],last['low_frame']+1],
           'reference_range':[first['reference_frame'],last['reference_frame']+1], 'origin':'manual','status':'proposed','pairs_file':None}
        value['ranges'].append(r)
        validate_ranges(value['ranges'],idx)
    elif operation=='settings':
        framing=payload.get('framing','fit')
        if framing not in ('fit','fill'): raise ValueError('Unknown framing option')
        crops=payload.get('crops',{})
        for s in ('low','reference'):
            crop=crops.get(s)
            if crop is not None:
                if not isinstance(crop,list) or len(crop)!=4 or not all(type(v) is int for v in crop): raise ValueError('Crop requires four integer edges')
                l,t,r,b=crop
                if not 0<=l<r<=idx[s].meta['width'] or not 0<=t<b<=idx[s].meta['height']: raise ValueError('Crop is outside the source')
        value['geometry']={'framing':framing,'crops':crops}
        for r in value['ranges']: r.update(status='proposed',pairs_file=None)
    elif operation in ('inspect_range','approve_range','remove_range'):
        r=next((r for r in value['ranges'] if r['id']==payload.get('range_id')),None)
        if r is None: raise ValueError('Range not found')
        if operation=='remove_range': value['ranges'].remove(r)
        elif operation=='inspect_range':
            write_pairs(job_dir,r,correspondence(r,idx,value['locks']))
        else:
            if not r.get('pairs_file'): raise ValueError('Inspect this range before approving it')
            if payload.get('accept_exclusions') is not True: raise ValueError('Acknowledge excluded training frames and reference output coverage')
            r['status']='approved'
    else: raise ValueError('Unknown frame-review operation')
    value['history']=(history+[snapshot])[-30:]
    validate_ranges(value['ranges'],idx)
    return value


def edit_manifest(review, idx, allow_proposed=False):
    clips=[]; cursor=0; ends={'low':0,'reference':0}
    def add(source,start,end,role):
        nonlocal cursor
        if end<=start:return
        begin,finish=idx[source].interval(start,end)
        duration=finish-begin
        clips.append({'source':source,'source_range':{'start_frame':start,'end_frame':end},
                      'role':role,'source_start_seconds':float(begin),'source_end_seconds':float(finish),
                      'output_start_seconds':float(cursor),'output_duration_seconds':float(duration),
                      'output_end_seconds':float(cursor+duration),'frame_count':end-start})
        cursor+=duration
    for r in sorted(review['ranges'],key=lambda r:r['low_range'][0]):
        if r['status']!='approved':
            if allow_proposed:continue
            raise ValueError('Approve each proposed range or choose Keep both sections')
        for s in ('reference','low'):add(s,ends[s],r[f'{s}_range'][0],f'{s}_only')
        add('reference',*r['reference_range'],'shared')
        ends={s:r[f'{s}_range'][1] for s in ends}
    for s in ('reference','low'):add(s,ends[s],len(idx[s]),f'{s}_only')
    return clips


def training_pairs(review, job_dir):
    rows=[]
    for r in review['ranges']:
        if r['status']=='approved':
            rows.extend({**p,'segment_id':r['id']} for p in read_pairs(job_dir,r) if p['eligible'])
    return rows
