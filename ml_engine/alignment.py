from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .media import MediaInfo, audio_fingerprints, normalize_reference_frame, sampled_frames


@dataclass(frozen=True)
class FrameRef:
    frame_index: int
    pts: int
    time_seconds: float


@dataclass(frozen=True)
class Anchor:
    low_time: float
    reference_time: float
    score: float


@dataclass(frozen=True)
class MatchSegment:
    id: str
    low_start: FrameRef
    low_end: FrameRef
    reference_start: FrameRef
    reference_end: FrameRef
    confidence: float
    origin: str = "automatic"
    status: str = "proposed"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AlignmentReport:
    mode: str
    confidence: float
    verified_seconds: float
    anchors: tuple[Anchor, ...]
    warning: str | None
    segments: tuple[MatchSegment, ...] = ()
    pair_count: int = 0
    review_revision: int | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["anchors"] = [asdict(a) for a in self.anchors]
        result["segments"] = [s.to_dict() for s in self.segments]
        return result


def frame_ref(index: int, info: MediaInfo) -> FrameRef:
    index = max(0, min(int(index), info.frame_count - 1))
    numerator = info.fps_numerator or round(info.fps * 1000)
    denominator = info.fps_denominator or 1000
    return FrameRef(index, index, index * denominator / numerator)


def _descriptor(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    gray = (gray - gray.mean()) / (gray.std() + 1e-6)
    return gray.ravel() / (np.linalg.norm(gray) + 1e-6)


def _orb_verify(low: np.ndarray, ref: np.ndarray) -> float:
    low = cv2.resize(low, (640, 480), interpolation=cv2.INTER_CUBIC)
    orb = cv2.ORB_create(nfeatures=1200)
    k1, d1 = orb.detectAndCompute(cv2.cvtColor(low, cv2.COLOR_BGR2GRAY), None)
    k2, d2 = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return 0.0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [a for a, b in pairs if a.distance < 0.72 * b.distance]
    if len(good) < 12:
        return 0.0
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    return float(mask.mean()) if mask is not None else 0.0


def _monotonic(candidates: list[Anchor]) -> list[Anchor]:
    """Maximum-score chronological subsequence; edits may create arbitrary time offsets."""
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda a: (a.low_time, a.reference_time))
    score = [a.score for a in candidates]
    previous = [-1] * len(candidates)
    for i, current in enumerate(candidates):
        for j in range(i):
            prior = candidates[j]
            if prior.low_time < current.low_time and prior.reference_time < current.reference_time:
                candidate_score = score[j] + current.score
                if candidate_score > score[i]:
                    score[i] = candidate_score
                    previous[i] = j
    cursor = int(np.argmax(score))
    result: list[Anchor] = []
    while cursor >= 0:
        result.append(candidates[cursor])
        cursor = previous[cursor]
    return list(reversed(result))


def _segments_from_anchors(
    anchors: list[Anchor], low_info: MediaInfo, reference_info: MediaInfo, sample_seconds: float
) -> list[MatchSegment]:
    groups: list[list[Anchor]] = []
    current: list[Anchor] = []
    for anchor in anchors:
        if current:
            prior = current[-1]
            low_delta = anchor.low_time - prior.low_time
            ref_delta = anchor.reference_time - prior.reference_time
            initial_offset = current[0].reference_time - current[0].low_time
            current_offset = anchor.reference_time - anchor.low_time
            # Within one retained section, elapsed playback time must advance together.
            if (
                low_delta > sample_seconds * 4 or ref_delta > sample_seconds * 4
                or abs(low_delta - ref_delta) > sample_seconds * 1.5
                or abs(current_offset - initial_offset) > 0.25
            ):
                if len(current) >= 2:
                    groups.append(current)
                current = []
        current.append(anchor)
    if len(current) >= 2:
        groups.append(current)

    segments: list[MatchSegment] = []
    for group in groups:
        low_start = round(group[0].low_time * low_info.fps)
        low_end = round(group[-1].low_time * low_info.fps)
        ref_start = round(group[0].reference_time * reference_info.fps)
        ref_end = round(group[-1].reference_time * reference_info.fps)
        if low_end <= low_start or ref_end <= ref_start:
            continue
        segments.append(MatchSegment(
            id=str(uuid4()),
            low_start=frame_ref(low_start, low_info),
            low_end=frame_ref(low_end, low_info),
            reference_start=frame_ref(ref_start, reference_info),
            reference_end=frame_ref(ref_end, reference_info),
            confidence=float(np.mean([a.score for a in group])),
        ))
    return segments


def analyze_alignment(
    low_path: str | Path,
    reference_path: str | Path,
    low_info: MediaInfo,
    reference_info: MediaInfo,
    sample_seconds: float = 1.0,
) -> dict:
    """Build reviewable, chronological segment proposals without trusting them as training truth."""
    low_samples = list(sampled_frames(low_path, sample_seconds))
    ref_samples = [(t, normalize_reference_frame(f, reference_info)) for t, f in sampled_frames(reference_path, sample_seconds)]
    if not low_samples or not ref_samples:
        return {"revision": 1, "mode": "review", "segments": [], "summary": {
            "proposed_segments": 0, "matched_seconds": 0.0,
            "warning": "No decodable samples were available. Continue unpaired or replace the inputs.",
        }}
    low_desc = np.stack([_descriptor(f) for _, f in low_samples])
    ref_desc = np.stack([_descriptor(f) for _, f in ref_samples])
    similarities = low_desc @ ref_desc.T
    if low_info.has_audio and reference_info.has_audio:
        low_audio = audio_fingerprints(low_path, sample_seconds)
        ref_audio = audio_fingerprints(reference_path, sample_seconds)
        if len(low_audio) >= len(low_samples) and len(ref_audio) >= len(ref_samples):
            audio_similarity = low_audio[:len(low_samples)] @ ref_audio[:len(ref_samples)].T
            # Only trust audio when it contains repeatable evidence; alternate tracks remain visual-only.
            if float(np.median(audio_similarity.max(axis=1))) >= 0.35:
                similarities = similarities * 0.8 + audio_similarity * 0.2
    low_best = similarities.argmax(axis=1)
    ref_best = similarities.argmax(axis=0)
    candidates: list[Anchor] = []
    for li, ri_value in enumerate(low_best):
        ri = int(ri_value)
        visual = float(similarities[li, ri])
        # Mutual nearest neighbors suppress repeated titles and recurring shots.
        if visual < 0.72 or int(ref_best[ri]) != li:
            continue
        geometric = _orb_verify(low_samples[li][1], ref_samples[ri][1])
        if geometric >= 0.45:
            candidates.append(Anchor(low_samples[li][0], ref_samples[ri][0], geometric))
    anchors = _monotonic(candidates)
    segments = _segments_from_anchors(anchors, low_info, reference_info, sample_seconds)
    matched = sum(max(0.0, s.low_end.time_seconds - s.low_start.time_seconds) for s in segments)
    warning = None if segments else "No reliable shared section was proposed. Add one manually or continue unpaired."
    return {
        "revision": 1,
        "mode": "review",
        "segments": [s.to_dict() for s in segments],
        "summary": {"proposed_segments": len(segments), "matched_seconds": matched, "warning": warning},
    }


def segment_from_dict(value: dict) -> MatchSegment:
    def ref(name: str) -> FrameRef:
        item = value[name]
        return FrameRef(int(item["frame_index"]), int(item.get("pts", item["frame_index"])), float(item["time_seconds"]))
    return MatchSegment(
        id=str(value["id"]), low_start=ref("low_start"), low_end=ref("low_end"),
        reference_start=ref("reference_start"), reference_end=ref("reference_end"),
        confidence=float(value.get("confidence", 1.0)), origin=str(value.get("origin", "manual")),
        status=str(value.get("status", "confirmed")),
    )


def validate_segments(segments: list[dict], low_info: MediaInfo, reference_info: MediaInfo) -> list[MatchSegment]:
    confirmed = [segment_from_dict(s) for s in segments if s.get("status") == "confirmed"]
    confirmed.sort(key=lambda s: s.low_start.frame_index)
    prior: MatchSegment | None = None
    for segment in confirmed:
        if not (0 <= segment.low_start.frame_index < segment.low_end.frame_index < low_info.frame_count):
            raise ValueError(f"Segment {segment.id} has invalid complete-source boundaries")
        if not (0 <= segment.reference_start.frame_index < segment.reference_end.frame_index < reference_info.frame_count):
            raise ValueError(f"Segment {segment.id} has invalid reference boundaries")
        if prior and (segment.low_start.frame_index <= prior.low_end.frame_index or segment.reference_start.frame_index <= prior.reference_end.frame_index):
            raise ValueError("Confirmed segments must be chronological and non-overlapping on both timelines")
        low_duration = segment.low_end.time_seconds - segment.low_start.time_seconds
        ref_duration = segment.reference_end.time_seconds - segment.reference_start.time_seconds
        if abs(low_duration - ref_duration) > max(0.25, min(low_duration, ref_duration) * 0.03):
            raise ValueError(f"Segment {segment.id} durations disagree; adjust or split its boundaries")
        prior = segment
    return confirmed


def build_pair_manifest(segments: list[MatchSegment], low_info: MediaInfo, reference_info: MediaInfo) -> list[dict]:
    """Map confirmed ranges by relative PTS. Decoding performs final local verification in the dataset."""
    pairs: list[dict] = []
    for segment in segments:
        duration = segment.reference_end.time_seconds - segment.reference_start.time_seconds
        sample_count = max(2, int(duration) + 1)
        for position in np.linspace(0.0, 1.0, sample_count):
            ref_index = round(segment.reference_start.frame_index + position * (
                segment.reference_end.frame_index - segment.reference_start.frame_index
            ))
            low_index = round(segment.low_start.frame_index + position * (
                segment.low_end.frame_index - segment.low_start.frame_index
            ))
            pairs.append({
                "segment_id": segment.id,
                "low_frame": min(low_info.frame_count - 1, low_index),
                "reference_frame": min(reference_info.frame_count - 1, ref_index),
                "confidence": segment.confidence,
            })
    unique = {(p["low_frame"], p["reference_frame"]): p for p in pairs}
    return list(unique.values())


def align_videos(
    low_path: str | Path,
    reference_path: str | Path,
    low_info: MediaInfo,
    reference_info: MediaInfo,
    sample_seconds: float = 2.0,
) -> AlignmentReport:
    """Compatibility inspection API: proposals are never silently accepted for training."""
    review = analyze_alignment(low_path, reference_path, low_info, reference_info, sample_seconds)
    segments = tuple(segment_from_dict({**s, "status": "confirmed"}) for s in review["segments"])
    anchors = tuple(
        Anchor(s.low_start.time_seconds, s.reference_start.time_seconds, s.confidence)
        for s in segments
    )
    verified = sum(s.low_end.time_seconds - s.low_start.time_seconds for s in segments)
    if not segments:
        return AlignmentReport("unpaired", 0.0, 0.0, (), review["summary"]["warning"])
    return AlignmentReport(
        "review_required", float(np.mean([s.confidence for s in segments])), verified,
        anchors, "Proposed matches require user confirmation before paired training.", segments,
    )
