import cv2
import numpy as np

from ml_engine import pipeline


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (80, 60))
    for index in range(20):
        writer.write(np.full((60, 80, 3), index * 5, np.uint8))
    writer.release()


def test_reference_only_processing_writes_complete_reports_without_review_artifacts(tmp_path, monkeypatch):
    low, reference = tmp_path / "low.mp4", tmp_path / "reference.mp4"
    make_video(low)
    make_video(reference)
    pretrained = tmp_path / "pretrained.pth"
    checkpoint = tmp_path / "checkpoint.pth"
    monkeypatch.setattr(pipeline, "ensure_pretrained", lambda: pretrained)
    monkeypatch.setattr(
        pipeline, "train_model",
        lambda *args, **kwargs: (checkpoint, {"selected": "fine_tuned", "verified_pair_count": 0}),
    )
    monkeypatch.setattr(pipeline, "upscale_video", lambda *args, **kwargs: {"frames": 20})
    review = {
        "revision": 0, "approved_mode": "unpaired", "matching_mode": "reference_only",
        "segments": [],
    }

    report = pipeline.process_pipeline(low, reference, tmp_path / "job", "quick", review)

    assert report["matching_mode"] == "reference_only"
    assert report["alignment"]["mode"] == "unpaired"
    assert report["media"]["low"]["frame_count"] == 20
    assert not (tmp_path / "job" / "match-review.json").exists()
    assert not (tmp_path / "job" / "low-proxy.mp4").exists()
    assert (tmp_path / "job" / "media.json").is_file()
    assert (tmp_path / "job" / "report.json").is_file()
