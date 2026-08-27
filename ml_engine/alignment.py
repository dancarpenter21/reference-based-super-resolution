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
class FrameRange:
    """A half-open CFR frame range: start_frame is included, end_frame is not."""

    start_frame: int
    end_frame: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AlignmentSpan:
    id: str
    kind: str
    low_range: FrameRange | None
    reference_range: FrameRange | None
    confidence: float | None = None
    origin: str = "automatic"
    status: str | None = None

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


def _sequence_pairs(similarities: np.ndarray, gap_penalty: float = 0.08) -> list[tuple[int, int]]:
    """Global ordered alignment with explicit gaps on either source timeline."""
    low_count, ref_count = similarities.shape
    scores = np.empty((low_count + 1, ref_count + 1), dtype=np.float32)
    trace = np.zeros((low_count + 1, ref_count + 1), dtype=np.uint8)
    scores[:, 0] = -gap_penalty * np.arange(low_count + 1)
    scores[0, :] = -gap_penalty * np.arange(ref_count + 1)
    trace[1:, 0] = 1  # skip low sample
    trace[0, 1:] = 2  # skip reference sample
    for low_index in range(1, low_count + 1):
        for ref_index in range(1, ref_count + 1):
            # Scores above ~0.68 are useful evidence; weaker comparisons are
            # cheaper to skip than to force into a correspondence.
            match_reward = (float(similarities[low_index - 1, ref_index - 1]) - 0.68) * 4.0
            choices = (
                scores[low_index - 1, ref_index - 1] + match_reward,
                scores[low_index - 1, ref_index] - gap_penalty,
                scores[low_index, ref_index - 1] - gap_penalty,
            )
            direction = int(np.argmax(choices))
            scores[low_index, ref_index] = choices[direction]
            trace[low_index, ref_index] = direction
    low_index, ref_index = low_count, ref_count
    pairs: list[tuple[int, int]] = []
    while low_index or ref_index:
        direction = int(trace[low_index, ref_index])
        if direction == 0:
            low_index -= 1
            ref_index -= 1
            if similarities[low_index, ref_index] >= 0.68:
                pairs.append((low_index, ref_index))
        elif direction == 1:
            low_index -= 1
        else:
            ref_index -= 1
    return list(reversed(pairs))


def _add_audio_evidence(
    similarities: np.ndarray, low_audio: np.ndarray, ref_audio: np.ndarray,
) -> np.ndarray:
    """Blend audio over the samples covered by both audio and video descriptors."""
    low_count = min(similarities.shape[0], len(low_audio))
    ref_count = min(similarities.shape[1], len(ref_audio))
    if low_count == 0 or ref_count == 0:
        return similarities
    audio_similarity = low_audio[:low_count] @ ref_audio[:ref_count].T
    # Repetitive music and ambience can resemble many different moments. Require
    # both a strong best match and meaningful separation from the runner-up.
    if ref_count < 2:
        return similarities
    ordered = np.partition(audio_similarity, -2, axis=1)
    best = ordered[:, -1]
    distinctness = best - ordered[:, -2]
    if float(np.median(best)) < 0.35 or float(np.median(distinctness)) < 0.02:
        return similarities
    combined = similarities.copy()
    combined[:low_count, :ref_count] = (
        combined[:low_count, :ref_count] * 0.8 + audio_similarity * 0.2
    )
    return combined


def _segments_from_anchors(
    anchors: list[Anchor], low_info: MediaInfo, reference_info: MediaInfo, sample_seconds: float
) -> list[MatchSegment]:
    groups: list[list[Anchor]] = []
    current: list[Anchor] = []
    frame_tolerance = max(2 / low_info.fps, 2 / reference_info.fps)
    for anchor in anchors:
        if current:
            prior = current[-1]
            low_delta = anchor.low_time - prior.low_time
            ref_delta = anchor.reference_time - prior.reference_time
            offsets = [item.reference_time - item.low_time for item in current]
            expected_offset = float(np.median(offsets))
            current_offset = anchor.reference_time - anchor.low_time
            # Missing confidence samples do not imply an edit. Split only when the
            # two source clocks actually diverge or their persistent offset changes.
            if (
                abs(low_delta - ref_delta) > frame_tolerance
                or abs(current_offset - expected_offset) > frame_tolerance
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
        low_end = round((group[-1].low_time + sample_seconds) * low_info.fps) - 1
        ref_start = round(group[0].reference_time * reference_info.fps)
        ref_end = round((group[-1].reference_time + sample_seconds) * reference_info.fps) - 1
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


def _read_selected_descriptors(
    path: str | Path, info: MediaInfo, indices: set[int],
) -> dict[int, np.ndarray]:
    """Decode selected CFR frames in batches without retaining full-size images."""
    wanted = sorted(index for index in indices if 0 <= index < info.frame_count)
    if not wanted:
        return {}
    groups: list[tuple[int, int]] = []
    start = end = wanted[0]
    for index in wanted[1:]:
        if index == end + 1:
            end = index
        else:
            groups.append((start, end))
            start = end = index
    groups.append((start, end))
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {}
    result: dict[int, np.ndarray] = {}
    wanted_set = set(wanted)
    try:
        for start, end in groups:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for index in range(start, end + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                if index in wanted_set:
                    result[index] = _descriptor(normalize_reference_frame(frame, info))
    finally:
        cap.release()
    return result


def refine_anchors(
    reference_path: str | Path,
    anchors: list[Anchor],
    low_samples: list[tuple[float, np.ndarray]],
    low_info: MediaInfo,
    reference_info: MediaInfo,
    search_seconds: float = 1.0,
) -> list[Anchor]:
    """Refine coarse reference times to exact frames near each sampled pair."""
    if not anchors or not Path(reference_path).is_file():
        return anchors
    samples_by_frame = {
        round(time * low_info.fps): frame for time, frame in low_samples
    }
    radius = max(2, round(reference_info.fps * search_seconds))
    candidates_by_anchor: list[tuple[Anchor, int, range]] = []
    candidate_indices: set[int] = set()
    for anchor in anchors:
        low_index = round(anchor.low_time * low_info.fps)
        center = round(anchor.reference_time * reference_info.fps)
        candidates = range(
            max(0, center - radius),
            min(reference_info.frame_count, center + radius + 1),
        )
        candidates_by_anchor.append((anchor, low_index, candidates))
        candidate_indices.update(candidates)
    reference_descriptors = _read_selected_descriptors(reference_path, reference_info, candidate_indices)
    refined: list[Anchor] = []
    for anchor, low_index, candidates in candidates_by_anchor:
        low_frame = samples_by_frame.get(low_index)
        if low_frame is None:
            refined.append(anchor)
            continue
        low_descriptor = _descriptor(low_frame)
        scores = [
            (float(low_descriptor @ reference_descriptors[index]), index)
            for index in candidates if index in reference_descriptors
        ]
        if not scores:
            refined.append(anchor)
            continue
        score, reference_index = max(scores)
        refined.append(Anchor(
            low_index / low_info.fps,
            reference_index / reference_info.fps,
            max(0.0, min(1.0, score)),
        ))
    return refined


def _frame_range(start: int, end: int) -> FrameRange | None:
    return FrameRange(int(start), int(end)) if end > start else None


def _range_seconds(value: FrameRange | None, info: MediaInfo) -> float:
    return 0.0 if value is None else (value.end_frame - value.start_frame) / info.fps


def _decorate_timeline(spans: list[dict], low_info: MediaInfo, reference_info: MediaInfo) -> list[dict]:
    cursor = 0.0
    result: list[dict] = []
    for span in spans:
        low = FrameRange(**span["low_range"]) if span.get("low_range") else None
        reference = FrameRange(**span["reference_range"]) if span.get("reference_range") else None
        duration = max(_range_seconds(low, low_info), _range_seconds(reference, reference_info))
        result.append({**span, "sequence_start_seconds": cursor, "sequence_duration_seconds": duration})
        cursor += duration
    return result


def spans_from_segments(
    segments: list[MatchSegment], low_info: MediaInfo, reference_info: MediaInfo,
) -> list[dict]:
    """Turn sparse match islands into a complete ordered partition of both sources."""
    spans: list[dict] = []
    low_cursor = reference_cursor = 0
    for segment in sorted(segments, key=lambda item: item.low_start.frame_index):
        low_start = segment.low_start.frame_index
        reference_start = segment.reference_start.frame_index
        gap_low = _frame_range(low_cursor, low_start)
        gap_reference = _frame_range(reference_cursor, reference_start)
        if gap_low or gap_reference:
            spans.append(AlignmentSpan(
                str(uuid4()), "difference", gap_low, gap_reference, origin="automatic",
            ).to_dict())
        # Anchor endpoints are matching frames, so make the stored range half-open.
        low_end = min(low_info.frame_count, segment.low_end.frame_index + 1)
        reference_end = min(reference_info.frame_count, segment.reference_end.frame_index + 1)
        spans.append(AlignmentSpan(
            segment.id, "match", FrameRange(low_start, low_end),
            FrameRange(reference_start, reference_end), segment.confidence,
            segment.origin, segment.status,
        ).to_dict())
        low_cursor, reference_cursor = low_end, reference_end
    tail_low = _frame_range(low_cursor, low_info.frame_count)
    tail_reference = _frame_range(reference_cursor, reference_info.frame_count)
    if tail_low or tail_reference:
        spans.append(AlignmentSpan(
            str(uuid4()), "difference", tail_low, tail_reference, origin="automatic",
        ).to_dict())
    if not spans:
        spans.append(AlignmentSpan(
            str(uuid4()), "difference", FrameRange(0, low_info.frame_count),
            FrameRange(0, reference_info.frame_count), origin="automatic",
        ).to_dict())
    return _decorate_timeline(spans, low_info, reference_info)


def alignment_summary(spans: list[dict], low_info: MediaInfo) -> dict:
    matches = [span for span in spans if span.get("kind") == "match"]
    confirmed = [span for span in matches if span.get("status") == "confirmed"]
    matched_frames = sum(
        span["low_range"]["end_frame"] - span["low_range"]["start_frame"] for span in confirmed
    )
    return {
        "proposed_blocks": sum(span.get("status") == "proposed" for span in matches),
        "confirmed_blocks": len(confirmed),
        "difference_blocks": sum(span.get("kind") == "difference" for span in spans),
        "matched_frames": matched_frames,
        "matched_seconds": matched_frames / low_info.fps,
        "warning": None if matches else "No reliable shared section was proposed. Add one manually or continue unpaired.",
    }


def validate_alignment_spans(
    spans: list[dict], low_info: MediaInfo, reference_info: MediaInfo,
) -> list[dict]:
    if not isinstance(spans, list) or not spans:
        raise ValueError("Alignment must contain at least one span")
    cursors = {"low": 0, "reference": 0}
    counts = {"low": low_info.frame_count, "reference": reference_info.frame_count}
    seen: set[str] = set()
    for span in spans:
        span_id = str(span.get("id", ""))
        if not span_id or span_id in seen:
            raise ValueError("Every alignment span must have a unique id")
        seen.add(span_id)
        kind = span.get("kind")
        if kind not in {"match", "difference"}:
            raise ValueError(f"Span {span_id} has an unknown kind")
        present = 0
        for stream in ("low", "reference"):
            value = span.get(f"{stream}_range")
            if value is None:
                continue
            present += 1
            try:
                start, end = int(value["start_frame"]), int(value["end_frame"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"Span {span_id} has an invalid {stream} range")
            if start != cursors[stream] or not (0 <= start < end <= counts[stream]):
                raise ValueError(f"Span {span_id} breaks the {stream} timeline partition")
            cursors[stream] = end
        if kind == "match":
            if present != 2 or span.get("status") not in {"proposed", "confirmed"}:
                raise ValueError(f"Matched span {span_id} requires both ranges and a valid status")
            if span.get("status") == "confirmed":
                low_range, ref_range = span["low_range"], span["reference_range"]
                low_duration = (low_range["end_frame"] - low_range["start_frame"]) / low_info.fps
                ref_duration = (ref_range["end_frame"] - ref_range["start_frame"]) / reference_info.fps
                # Half a frame accounts for unavoidable rounding when the source
                # rates differ, while still rejecting a full missing frame.
                tolerance = max(0.5 / low_info.fps, 0.5 / reference_info.fps) + 1e-6
                if abs(low_duration - ref_duration) > tolerance:
                    raise ValueError(
                        f"Matched span {span_id} contains missing or extra frames; "
                        "mark out before the discontinuity and create a new segment after it"
                    )
        elif not present:
            raise ValueError(f"Difference span {span_id} does not contain footage")
    for stream in ("low", "reference"):
        if cursors[stream] != counts[stream]:
            raise ValueError(f"Alignment does not cover the complete {stream} video")
    return _decorate_timeline(spans, low_info, reference_info)


def build_dense_pair_manifest(spans: list[dict], low_info: MediaInfo, reference_info: MediaInfo) -> list[dict]:
    pairs: list[dict] = []
    for span in spans:
        if span.get("kind") != "match" or span.get("status") != "confirmed":
            continue
        low, reference = span["low_range"], span["reference_range"]
        low_count = low["end_frame"] - low["start_frame"]
        ref_count = reference["end_frame"] - reference["start_frame"]
        for offset in range(low_count):
            position = offset / max(1, low_count - 1)
            ref_offset = round(position * max(0, ref_count - 1))
            pairs.append({
                "span_id": span["id"], "segment_id": span["id"],
                "low_frame": low["start_frame"] + offset,
                "reference_frame": reference["start_frame"] + ref_offset,
                "confidence": float(span.get("confidence", 1.0)),
            })
    return pairs


def build_edit_manifest(
    spans: list[dict], low_info: MediaInfo, reference_info: MediaInfo,
) -> list[dict]:
    """Build the final, de-duplicated edit from a complete reviewed partition.

    Confirmed shared footage comes from the reference. Difference blocks retain
    every source-only range; when both sources differ, reference footage is
    emitted before the supplemental footage.
    """
    spans = validate_alignment_spans(spans, low_info, reference_info)
    if any(span.get("kind") == "match" and span.get("status") != "confirmed" for span in spans):
        raise ValueError("Resolve every proposed match before rendering the combined timeline")

    clips: list[dict] = []
    output_cursor = 0

    def append_clip(span: dict, source: str, role: str) -> None:
        nonlocal output_cursor
        info = reference_info if source == "reference" else low_info
        source_range = span[f"{source}_range"]
        source_count = source_range["end_frame"] - source_range["start_frame"]
        output_count = max(1, round(source_count / info.fps * reference_info.fps))
        clips.append({
            "span_id": span["id"], "source": source, "role": role,
            "source_range": dict(source_range),
            "source_duration_seconds": source_count / info.fps,
            "output_start_frame": output_cursor,
            "output_end_frame": output_cursor + output_count,
            "output_duration_seconds": output_count / reference_info.fps,
            "audio_source": source,
        })
        output_cursor += output_count

    for span in spans:
        if span["kind"] == "match":
            append_clip(span, "reference", "shared")
            continue
        if span.get("reference_range"):
            append_clip(span, "reference", "reference_only")
        if span.get("low_range"):
            append_clip(span, "low", "supplemental_only")
    return clips


def analyze_alignment(
    low_path: str | Path,
    reference_path: str | Path,
    low_info: MediaInfo,
    reference_info: MediaInfo,
    sample_seconds: float = 1.0,
    use_audio: bool = False,
) -> dict:
    """Build reviewable, chronological segment proposals without trusting them as training truth."""
    low_samples = [
        (t, normalize_reference_frame(f, low_info))
        for t, f in sampled_frames(low_path, sample_seconds)
    ]
    ref_samples = [
        (t, normalize_reference_frame(f, reference_info))
        for t, f in sampled_frames(reference_path, sample_seconds)
    ]
    if not low_samples or not ref_samples:
        spans = spans_from_segments([], low_info, reference_info)
        return {"schema_version": 2, "revision": 1, "mode": "review", "spans": spans, "summary": {
            **alignment_summary(spans, low_info),
            "warning": "No decodable samples were available. Continue unpaired or replace the inputs.",
        }}
    low_desc = np.stack([_descriptor(f) for _, f in low_samples])
    ref_desc = np.stack([_descriptor(f) for _, f in ref_samples])
    similarities = low_desc @ ref_desc.T
    if use_audio and low_info.has_audio and reference_info.has_audio:
        low_audio = audio_fingerprints(low_path, sample_seconds)
        ref_audio = audio_fingerprints(reference_path, sample_seconds)
        similarities = _add_audio_evidence(similarities, low_audio, ref_audio)
    candidates: list[Anchor] = []
    for li, ri in _sequence_pairs(similarities):
        visual = float(similarities[li, ri])
        if visual < 0.68:
            continue
        geometric = _orb_verify(low_samples[li][1], ref_samples[ri][1])
        if geometric >= 0.45:
            candidates.append(Anchor(low_samples[li][0], ref_samples[ri][0], geometric))
        else:
            # Ordered, highly similar frames bridge compression-heavy stretches
            # where ORB has too few stable keypoints. Their lower score keeps the
            # resulting block visibly less trustworthy during review.
            candidates.append(Anchor(low_samples[li][0], ref_samples[ri][0], visual * 0.45))
    anchors = refine_anchors(reference_path, candidates, low_samples, low_info, reference_info)
    segments = _segments_from_anchors(anchors, low_info, reference_info, sample_seconds)
    matched = sum(max(0.0, s.low_end.time_seconds - s.low_start.time_seconds) for s in segments)
    warning = None if segments else "No reliable shared section was proposed. Add one manually or continue unpaired."
    spans = spans_from_segments(segments, low_info, reference_info)
    return {
        "schema_version": 2, "revision": 1, "mode": "review", "spans": spans,
        "summary": {**alignment_summary(spans, low_info), "warning": warning},
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
            raise ValueError(f"Segment {segment.id} has invalid supplemental-video boundaries")
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
    if review.get("schema_version") == 2:
        segments = tuple(
            MatchSegment(
                id=span["id"],
                low_start=frame_ref(span["low_range"]["start_frame"], low_info),
                low_end=frame_ref(span["low_range"]["end_frame"] - 1, low_info),
                reference_start=frame_ref(span["reference_range"]["start_frame"], reference_info),
                reference_end=frame_ref(span["reference_range"]["end_frame"] - 1, reference_info),
                confidence=float(span.get("confidence", 1.0)), status="confirmed",
            )
            for span in review["spans"] if span.get("kind") == "match"
        )
    else:
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
