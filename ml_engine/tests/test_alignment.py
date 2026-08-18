import cv2
import numpy as np

from ml_engine.alignment import analyze_alignment, build_pair_manifest, segment_from_dict, align_videos
from ml_engine.media import probe


def write_video(path, seed, frames=90, fps=10):
    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    base = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    for index in range(frames):
        frame = np.roll(base, index // 2, axis=1)
        cv2.putText(frame, str(index), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def test_unrelated_videos_fall_back(tmp_path):
    low, ref = tmp_path / "low.mp4", tmp_path / "ref.mp4"
    write_video(low, 1)
    write_video(ref, 2)
    report = align_videos(low, ref, probe(low), probe(ref), sample_seconds=1)
    assert report.mode == "unpaired"
    assert report.warning


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

    assert len(review["segments"]) >= 2
    confirmed = [segment_from_dict({**item, "status": "confirmed"}) for item in review["segments"]]
    manifest = build_pair_manifest(confirmed, low_info, ref_info)
    assert manifest
    assert all(0 <= pair["low_frame"] < low_info.frame_count for pair in manifest)
    assert all(0 <= pair["reference_frame"] < ref_info.frame_count for pair in manifest)
