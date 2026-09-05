from pathlib import Path
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api import endpoints
from app.services.job_store import JobStore
from ml_engine.indexed_media import build_index
from ml_engine.tests.test_indexed_workflow import video


@pytest.fixture
def setup(tmp_path,monkeypatch):
    jobs=JobStore(tmp_path/'jobs.sqlite3');root=tmp_path/'job';root.mkdir()
    for source in ('low','reference'):
        path=root/f'{source}.mp4';video(path);build_index(path,root/'indexes'/source)
    job=jobs.create(str(root/'low.mp4'),str(root/'reference.mp4'),str(root),'quick')
    from ml_engine.review_v3 import indexes
    idx=indexes(root)
    review={'schema_version':3,'revision':1,'locks':[],'ranges':[],'history':[],'geometry':{'framing':'fit','crops':{}},'media':{s:{**i.meta,'duration':float(i.duration)} for s,i in idx.items()}}
    jobs.update(job['id'],state='awaiting_match_review',review_data=review,review_revision=1)
    monkeypatch.setattr(endpoints,'store',jobs)
    monkeypatch.setattr(endpoints.worker,'notify',lambda:None)
    return TestClient(app),jobs,job['id'],root


def test_exact_frame_and_index_are_authoritative(setup):
    client,store,id,root=setup
    row=client.get(f'/api/v1/jobs/{id}/frame-index/low?start=2&limit=1').json()['frames'][0]
    response=client.get(f'/api/v1/jobs/{id}/frames/low/2')
    assert response.status_code==200
    assert response.headers['content-type']=='image/png'
    assert int(response.headers['x-frame-pts'])==row['pts']
    assert float(response.headers['x-frame-time'])==row['time_seconds']
    assert client.get(f'/api/v1/jobs/{id}/frames/low/999').status_code==416


def test_lock_is_atomic_and_stale_edit_cannot_overwrite(setup):
    c,store,id,root=setup;url=f'/api/v1/jobs/{id}/match-review'
    a=c.patch(url,json={'revision':1,'operation':'lock_pair','low_frame':0,'reference_frame':1})
    assert a.status_code==200
    assert a.json()['locks'][0]['reference']['frame_number']==2
    assert a.json()['ranges']==[] and a.json()['can_undo']
    b=c.patch(url,json={'revision':1,'operation':'lock_pair','low_frame':2,'reference_frame':3})
    assert b.status_code==409
    assert len(store.get(id)['review_data']['locks'])==1
    undone=c.patch(url,json={'revision':2,'operation':'undo'})
    assert undone.status_code==200 and undone.json()['locks']==[]


def test_preview_state_is_durable_and_not_a_queued_job(setup):
    c,store,id,root=setup
    store.update(id,state='awaiting_quality_preview',phase='preview')
    reopened=JobStore(store.path)
    assert reopened.get(id)['state']=='awaiting_quality_preview'
    assert reopened.next_queued() is None
    assert reopened.cancel(id)['state']=='cancelled'


def test_export_requires_current_preview_and_known_method(setup):
    c,store,id,root=setup
    url=f'/api/v1/jobs/{id}/render'
    assert c.post(url,json={'revision':1,'method':'lanczos'}).status_code==409
    (root/'preparation-v3.json').write_text(json.dumps({'revision':1,'previews':{'lanczos':'preview-lanczos.mp4'}}))
    store.update(id,state='awaiting_quality_preview',phase='preview')
    assert c.post(url,json={'revision':0,'method':'lanczos'}).status_code==409
    assert c.post(url,json={'revision':1,'method':'unknown'}).status_code==422
    result=c.post(url,json={'revision':1,'method':'lanczos'})
    assert result.status_code==202
    assert store.get(id)['phase']=='render'


def test_pagination_limits_are_validated(setup):
    c,store,id,root=setup
    assert c.get(f'/api/v1/jobs/{id}/frame-index/low?limit=10000').status_code==422
