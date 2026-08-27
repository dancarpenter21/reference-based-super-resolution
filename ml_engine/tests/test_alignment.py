import cv2
import numpy as np
import pytest

from ml_engine.alignment import (
    Anchor, _add_audio_evidence, align_videos, analyze_alignment, build_dense_pair_manifest,
    build_edit_manifest,
    refine_anchors, validate_alignment_spans,
)
from ml_engine.media import MediaInfo, normalize_reference_frame, probe


def write_video(path, seed, frames=90, fps=10):
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    base = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    for index in range(frames):
        frame = np.roll(base, index // 2, axis=1)
        cv2.putText(frame, str(index), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def test_refines_coarse_anchors_to_exact_shifted_frames(tmp_path):
    low, ref = tmp_path / "low-shifted.mp4", tmp_path / "ref-shifted.mp4"
    rng = np.random.default_rng(91)
    frames = [rng.integers(0, 255, (120, 160, 3), dtype=np.uint8) for _ in range(12)]
    prefix = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(3)]
    for path, content in ((low, frames), (ref, prefix + frames)):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
        for frame in content:
            writer.write(frame)
        writer.release()
    low_info, ref_info = probe(low), probe(ref)
    capture = cv2.VideoCapture(str(low))
    low_samples = []
    for index in (2, 7):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        assert ok
        low_samples.append((index / low_info.fps, normalize_reference_frame(frame, low_info)))
    capture.release()

    refined = refine_anchors(
        ref,
        [Anchor(.2, .0, .8), Anchor(.7, 1.0, .8)],
        low_samples,
        low_info,
        ref_info,
    )

    assert [round(anchor.reference_time * ref_info.fps) for anchor in refined] == [5, 10]


def test_unrelated_videos_fall_back(tmp_path):
    low, ref = tmp_path / "low.mp4", tmp_path / "ref.mp4"
    write_video(low, 1)
    write_video(ref, 2)
    report = align_videos(low, ref, probe(low), probe(ref), sample_seconds=1)
    assert report.mode == "unpaired"
    assert report.warning


def test_alignment_normalizes_both_streams(monkeypatch):
    low_info = MediaInfo("low.mp4", 160, 90, 1, 2, 2, "h264", False, "1:1", False)
    ref_info = MediaInfo("ref.mp4", 320, 180, 1, 2, 2, "h264", False, "1:1", False)
    frames = [
        np.random.default_rng(seed).integers(0, 255, (90, 160, 3), dtype=np.uint8)
        for seed in (1, 2)
    ]
    normalized_infos = []

    def samples(path, _sample_seconds):
        scale = 2 if str(path) == ref_info.path else 1
        return iter(
            (float(index), cv2.resize(frame, None, fx=scale, fy=scale))
            for index, frame in enumerate(frames)
        )

    def normalize(frame, info):
        normalized_infos.append(info.path)
        return cv2.resize(frame, (640, 480))

    monkeypatch.setattr("ml_engine.alignment.sampled_frames", samples)
    monkeypatch.setattr("ml_engine.alignment.normalize_reference_frame", normalize)

    review = analyze_alignment(low_info.path, ref_info.path, low_info, ref_info)

    assert normalized_infos == [low_info.path, low_info.path, ref_info.path, ref_info.path]
    assert any(span["kind"] == "match" for span in review["spans"])


def test_audio_evidence_uses_available_overlap():
    visual = np.full((3, 3), 0.5, dtype=np.float32)
    low_audio = np.eye(2, dtype=np.float32)
    ref_audio = np.eye(2, dtype=np.float32)

    combined = _add_audio_evidence(visual, low_audio, ref_audio)

    assert combined[:2, :2] == pytest.approx(np.array([[0.6, 0.4], [0.4, 0.6]]))
    assert combined[2] == pytest.approx(visual[2])


def test_audio_evidence_rejects_repetitive_ambiguous_matches():
    visual = np.full((3, 3), 0.5, dtype=np.float32)
    repetitive_audio = np.array([
        [1.0, 0.0],
        [0.999, 0.045],
        [0.998, 0.063],
    ], dtype=np.float32)

    combined = _add_audio_evidence(visual, repetitive_audio, repetitive_audio)

    assert combined == pytest.approx(visual)


def write_timeline(path, source_seconds, fps):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    textures = {
        second: np.random.default_rng(second + 100).integers(0, 255, (120, 160, 3), dtype=np.uint8)
        for second in source_seconds
    }
    for second in source_seconds:
        for subframe in range(fps):
            frame = np.roll(textures[second], subframe // max(1, fps // 4), axis=1)
            cv2.putText(frame, f"S{second}", (12, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            writer.write(frame)
    writer.release()


def test_different_frame_rates_and_intermittent_gap_form_segments(tmp_path):
    low, ref = tmp_path / "low.mp4", tmp_path / "ref.mp4"
    write_timeline(low, list(range(12)), 10)
    write_timeline(ref, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11], 8)
    low_info, ref_info = probe(low), probe(ref)

    review = analyze_alignment(low, ref, low_info, ref_info, sample_seconds=1)

    assert review["schema_version"] == 2
    matches = [span for span in review["spans"] if span["kind"] == "match"]
    differences = [span for span in review["spans"] if span["kind"] == "difference"]
    assert len(matches) >= 2
    assert differences
    all_confirmed = [
        {**span, "status": "confirmed"} if span["kind"] == "match" else span
        for span in review["spans"]
    ]
    with pytest.raises(ValueError, match="missing or extra frames"):
        validate_alignment_spans(all_confirmed, low_info, ref_info)

    # Correct the first automatic draft to an exact-duration pair. The one
    # reference frame removed from Match Out becomes explicit unpaired footage.
    confirmed = [{**span} for span in review["spans"]]
    first_match = next(span for span in confirmed if span["kind"] == "match")
    first_match["reference_range"] = {**first_match["reference_range"], "end_frame": 32}
    first_match["status"] = "confirmed"
    following_difference = confirmed[confirmed.index(first_match) + 1]
    following_difference["reference_range"] = {"start_frame": 32, "end_frame": 33}
    confirmed = validate_alignment_spans(confirmed, low_info, ref_info)
    manifest = build_dense_pair_manifest(confirmed, low_info, ref_info)
    assert manifest
    assert all(0 <= pair["low_frame"] < low_info.frame_count for pair in manifest)
    assert all(0 <= pair["reference_frame"] < ref_info.frame_count for pair in manifest)
    expected_low_frames = sum(
        span["low_range"]["end_frame"] - span["low_range"]["start_frame"]
        for span in confirmed if span["kind"] == "match" and span["status"] == "confirmed"
    )
    assert len(manifest) == expected_low_frames
    assert [pair["low_frame"] for pair in manifest] == sorted({pair["low_frame"] for pair in manifest})


def test_alignment_partition_exposes_intro_tail_and_divergent_gap(tmp_path):
    low, ref = tmp_path / "low.mp4", tmp_path / "ref.mp4"
    write_timeline(low, [0, 1, 2, 3, 12, 13, 6, 7, 8, 14], 10)
    write_timeline(ref, [20, 21, 0, 1, 2, 3, 9, 10, 11, 15, 6, 7, 8], 8)
    low_info, ref_info = probe(low), probe(ref)

    review = analyze_alignment(low, ref, low_info, ref_info, sample_seconds=1)
    spans = validate_alignment_spans(review["spans"], low_info, ref_info)

    assert spans[0]["kind"] == "difference"
    assert spans[0]["reference_range"] is not None
    assert any(
        span["kind"] == "difference" and span["low_range"] and span["reference_range"]
        for span in spans
    )
    assert spans[-1]["kind"] == "difference"
    assert spans[-1]["low_range"] is not None
    assert sum(span.get("kind") == "match" for span in spans) >= 2


def test_edit_manifest_prefers_reference_and_keeps_both_sides_of_differences(tmp_path):
    low, ref = tmp_path / "edit-low.mp4", tmp_path / "edit-ref.mp4"
    write_timeline(low, list(range(10)), 10)
    write_timeline(ref, list(range(12)), 8)
    low_info, ref_info = probe(low), probe(ref)
    spans = [
        {"id": "intro", "kind": "difference", "low_range": None,
         "reference_range": {"start_frame": 0, "end_frame": 2}, "status": None},
        {"id": "shared", "kind": "match", "low_range": {"start_frame": 0, "end_frame": 5},
         "reference_range": {"start_frame": 2, "end_frame": 6}, "status": "confirmed", "confidence": .9},
        {"id": "different", "kind": "difference", "low_range": {"start_frame": 5, "end_frame": 8},
         "reference_range": {"start_frame": 6, "end_frame": 9}, "status": None},
        {"id": "tails", "kind": "difference", "low_range": {"start_frame": 8, "end_frame": low_info.frame_count},
         "reference_range": {"start_frame": 9, "end_frame": ref_info.frame_count}, "status": None},
    ]

    clips = build_edit_manifest(spans, low_info, ref_info)

    assert [(clip["source"], clip["role"]) for clip in clips] == [
        ("reference", "reference_only"), ("reference", "shared"),
        ("reference", "reference_only"), ("low", "supplemental_only"),
        ("reference", "reference_only"), ("low", "supplemental_only"),
    ]
    assert sum(
        clip["source_range"]["end_frame"] - clip["source_range"]["start_frame"]
        for clip in clips if clip["source"] == "reference"
    ) == ref_info.frame_count
    assert [clip["output_start_frame"] for clip in clips] == [
        0, *[clip["output_end_frame"] for clip in clips[:-1]]
    ]


def test_edit_manifest_rejects_unresolved_shared_ranges(tmp_path):
    low, ref = tmp_path / "draft-low.mp4", tmp_path / "draft-ref.mp4"
    write_timeline(low, list(range(2)), 10)
    write_timeline(ref, list(range(2)), 10)
    low_info, ref_info = probe(low), probe(ref)
    spans = [{
        "id": "draft", "kind": "match", "low_range": {"start_frame": 0, "end_frame": 20},
        "reference_range": {"start_frame": 0, "end_frame": 20}, "status": "proposed",
    }]
    with pytest.raises(ValueError, match="Resolve every proposed match"):
        build_edit_manifest(spans, low_info, ref_info)
