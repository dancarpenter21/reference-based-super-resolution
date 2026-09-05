from fractions import Fraction

import av
import cv2
import numpy as np
import pytest
import torch

from ml_engine.indexed_media import build_index, FrameReader, fit_frame
from ml_engine.review_v3 import edit_review, edit_manifest, indexes, read_pairs, training_pairs
from ml_engine.restoration_v3 import render, restore, stabilize, prepare_examples


def video(path, timestamps=(0,33,83,100,180,213), size=(80,48), bframes=0, start=0):
    with av.open(str(path),'w') as c:
        s=c.add_stream('libx264',rate=30);s.width,s.height=size;s.pix_fmt='yuv420p'
        s.time_base=Fraction(1,1000);s.codec_context.time_base=Fraction(1,1000)
        s.options={'bf':str(bframes),'crf':'12'}
        for i,pts in enumerate(timestamps):
            rng=np.random.default_rng(i)
            frame=rng.integers(20,230,(*size[::-1],3),np.uint8)
            frame[:,0:8]=(0,0,255);frame[:,-8:]=(0,255,0)
            f=av.VideoFrame.from_ndarray(frame,format='bgr24');f.pts=start+pts;f.time_base=Fraction(1,1000)
            for p in s.encode(f):c.mux(p)
        for p in s.encode():c.mux(p)


@pytest.fixture
def indexed(tmp_path):
    for source in ('low','reference'):
        path=tmp_path/f'{source}.mp4';video(path)
        build_index(path,tmp_path/'indexes'/source)
    review={'schema_version':3,'revision':1,'locks':[],'ranges':[],'history':[],'geometry':{'framing':'fit','crops':{}}}
    return tmp_path,indexes(tmp_path),review


@pytest.mark.parametrize('bframes',[0,2])
def test_exact_identity_and_seek_with_variable_timestamps(tmp_path,bframes):
    path=tmp_path/'video.mp4';video(path,bframes=bframes,start=700)
    idx=build_index(path,tmp_path/'index')
    assert idx.meta['variable_frame_rate']
    assert idx.frame(2)['time_seconds']==pytest.approx(.083)
    assert idx.frame(2)['frame_number']==3
    assert idx.frame(2)['pts']!=2
    with av.open(str(path)) as c:expected=[f.to_ndarray(format='bgr24') for f in c.decode(video=0)]
    with FrameReader(idx) as reader:
        for i in [4,1,2,0,5]: np.testing.assert_array_equal(reader.read(i),expected[i])


def test_timestamp_export_keeps_every_selected_duration(indexed):
    root,idx,review=indexed
    clips=edit_manifest(review,idx)
    output=root/'result.mp4'
    render(clips,idx,review,(80,48),output,method='lanczos')
    result=build_index(output,root/'output-index')
    expected=[r[2]*idx[s].time_base for s in ('reference','low') for r in idx[s].rows]
    actual=[r[2]*result.time_base for r in result.rows]
    assert actual==expected
    assert result.duration==sum(expected)


def test_pair_lock_does_not_approve_or_create_range_and_undo(indexed):
    root,idx,review=indexed
    locked=edit_review(review,{'operation':'lock_pair','low_frame':0,'reference_frame':1},root)
    assert review['locks']==[]
    assert locked['ranges']==[]
    assert len(locked['locks'])==1
    assert len(edit_manifest(locked,idx))==2
    undone=edit_review(locked,{'operation':'undo'},root)
    assert undone['locks']==[]


def test_explicit_replacement_removes_old_lock(indexed):
    root,idx,review=indexed
    a=edit_review(review,{'operation':'lock_pair','low_frame':0,'reference_frame':0},root)
    b=edit_review(a,{'operation':'lock_pair','low_frame':1,'reference_frame':2,'replace_id':a['locks'][0]['id']},root)
    assert [(p['low_frame'],p['reference_frame']) for p in b['locks']]==[(1,2)]


def test_range_approval_is_separate_and_frame_ids_survive(indexed):
    root,idx,r=indexed
    for i in [0,5]:r=edit_review(r,{'operation':'lock_pair','low_frame':i,'reference_frame':i},root)
    r=edit_review(r,{'operation':'create_range','first':r['locks'][0]['id'],'last':r['locks'][1]['id']},root)
    range_id=r['ranges'][0]['id']
    with pytest.raises(ValueError,match='Approve'):edit_manifest(r,idx)
    r=edit_review(r,{'operation':'inspect_range','range_id':range_id},root)
    pairs=read_pairs(root,r['ranges'][0]);assert len(pairs)==6
    assert pairs[0]['status']=='manual' and pairs[-1]['status']=='manual'
    assert training_pairs(r,root)==[]
    r=edit_review(r,{'operation':'approve_range','range_id':range_id,'accept_exclusions':True},root)
    assert [p['low_frame'] for p in training_pairs(r,root)]==list(range(6))
    assert len(edit_manifest(r,idx))==1
    changed=edit_review(r,{'operation':'remove_pair','low_frame':0,'reference_frame':0},root)
    assert changed['ranges'][0]['status']=='proposed'
    assert not changed['ranges'][0]['pairs_file']


def test_conflicting_lock_rejected(indexed):
    root,idx,r=indexed
    r=edit_review(r,{'operation':'lock_pair','low_frame':0,'reference_frame':4},root)
    with pytest.raises(ValueError,match='crosses'):
        edit_review(r,{'operation':'lock_pair','low_frame':3,'reference_frame':2},root)


def test_crop_change_invalidates_coverage(indexed):
    root,idx,r=indexed
    r['ranges']=[{'id':'range','low_range':[0,6],'reference_range':[0,6],'status':'approved','pairs_file':'old.json'}]
    r=edit_review(r,{'operation':'settings','framing':'fit','crops':{'low':[0,0,72,48]}},root)
    assert r['ranges'][0]['status']=='proposed' and r['ranges'][0]['pairs_file'] is None


@pytest.mark.parametrize('size',[(192,108),(80,60),(60,100)])
def test_full_picture_fit_retains_colored_edges(size):
    frame=np.full((48,80,3),120,np.uint8);frame[:,:8]=(0,0,255);frame[:,-8:]=(0,255,0)
    result=fit_frame(frame,size)
    assert result.shape==(size[1],size[0],3)
    assert ((result[...,2]>200)&(result[...,1]<40)).any()
    assert ((result[...,1]>200)&(result[...,2]<40)).any()


def test_tiled_inference_preserves_geometry_and_has_no_core_seams():
    class Nearest(torch.nn.Module):
        def forward(self,x):return torch.nn.functional.interpolate(x,scale_factor=2,mode='nearest')
    frame=np.random.default_rng(1).integers(0,256,(99,151,3),np.uint8)
    out=restore(Nearest(),frame,torch.device('cpu'),tile=64)
    np.testing.assert_array_equal(out,cv2.resize(frame,(302,198),interpolation=cv2.INTER_NEAREST))


def test_stabilization_rejects_cut_and_long_gap():
    current=np.full((48,80,3),200,np.uint8);previous=np.zeros_like(current)
    np.testing.assert_array_equal(stabilize(current,previous,current,previous,.033),current)
    np.testing.assert_array_equal(stabilize(current,previous,current,previous,.5),current)


def test_small_training_set_keeps_pretrained(indexed):
    root,idx,r=indexed
    train,val,test,report=prepare_examples([{'low_frame':0,'reference_frame':0}],idx,r)
    assert train==val==test==[]
    assert 'independent' in report['reason']


def test_index_cancellation_leaves_no_valid_partial(tmp_path):
    path=tmp_path/'v.mp4';video(path)
    with pytest.raises(InterruptedError):build_index(path,tmp_path/'index',cancel=lambda:True)
    assert not (tmp_path/'index'/'frames.sqlite3').exists()
