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


def test_v2_processing_passes_every_confirmed_low_frame_to_training(tmp_path, monkeypatch):
    low, reference = tmp_path / "low.mp4", tmp_path / "reference.mp4"
    make_video(low)
    make_video(reference)
    captured = {}
    monkeypatch.setattr(pipeline, "ensure_pretrained", lambda: tmp_path / "pretrained.pth")

    def train(*args, **kwargs):
        captured["pairs"] = kwargs["paired_manifest"]
        return tmp_path / "checkpoint.pth", {"selected": "fine_tuned"}

    monkeypatch.setattr(pipeline, "train_model", train)
    monkeypatch.setattr(
        pipeline, "compose_timeline",
        lambda *args, **kwargs: {"frames": 25, "clips": [], "width": 80, "height": 60},
    )
    review = {
        "schema_version": 2, "revision": 3, "approved_mode": "paired", "matching_mode": "guided",
        "spans": [
            {"id": "intro", "kind": "difference", "low_range": None, "reference_range": {"start_frame": 0, "end_frame": 5}, "status": None},
            {"id": "shared", "kind": "match", "low_range": {"start_frame": 0, "end_frame": 15}, "reference_range": {"start_frame": 5, "end_frame": 20}, "status": "confirmed", "confidence": .9},
            {"id": "tail", "kind": "difference", "low_range": {"start_frame": 15, "end_frame": 20}, "reference_range": None, "status": None},
        ],
    }

    report = pipeline.process_pipeline(low, reference, tmp_path / "v2-job", "quick", review)

    assert report["alignment"]["mode"] == "paired"
    assert report["alignment"]["pair_count"] == 15
    assert [pair["low_frame"] for pair in captured["pairs"]] == list(range(15))
    assert captured["pairs"][0]["reference_frame"] == 5
    assert captured["pairs"][-1]["reference_frame"] == 19
    assert report["composition"]["coverage_mode"] == "combined_timeline"
    assert (tmp_path / "v2-job" / "edit-manifest.json").is_file()
