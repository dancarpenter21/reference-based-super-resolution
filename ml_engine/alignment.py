from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .media import MediaInfo, normalize_reference_frame, sampled_frames


@dataclass(frozen=True)
class Anchor:
    low_time: float
    reference_time: float
    score: float


@dataclass(frozen=True)
class AlignmentReport:
    mode: str
    confidence: float
    verified_seconds: float
    anchors: tuple[Anchor, ...]
    warning: str | None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["anchors"] = [asdict(a) for a in self.anchors]
        return result


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


def align_videos(
    low_path: str | Path,
    reference_path: str | Path,
    low_info: MediaInfo,
    reference_info: MediaInfo,
    sample_seconds: float = 2.0,
) -> AlignmentReport:
    """Conservatively accept only repeated, geometrically verified timeline matches."""
    low_samples = list(sampled_frames(low_path, sample_seconds))
    ref_samples = [
        (t, normalize_reference_frame(frame, reference_info))
        for t, frame in sampled_frames(reference_path, sample_seconds)
    ]
    if not low_samples or not ref_samples:
        return AlignmentReport("unpaired", 0.0, 0.0, (), "No decodable samples")
    low_desc = np.stack([_descriptor(f) for _, f in low_samples])
    ref_desc = np.stack([_descriptor(f) for _, f in ref_samples])
    similarities = low_desc @ ref_desc.T
    candidates: list[Anchor] = []
    for li in range(len(low_samples)):
        ri = int(similarities[li].argmax())
        visual = float(similarities[li, ri])
        if visual < 0.78:
            continue
        geometric = _orb_verify(low_samples[li][1], ref_samples[ri][1])
        if geometric >= 0.55:
            candidates.append(Anchor(low_samples[li][0], ref_samples[ri][0], geometric))
    if not candidates:
        return AlignmentReport(
            "unpaired", 0.0, 0.0, (),
            "No geometrically trustworthy overlap was found; using synthetic reference adaptation.",
        )
    # A true edited subset creates locally consistent offsets. Retain anchors supported by neighbors.
    kept: list[Anchor] = []
    for anchor in candidates:
        offset = anchor.reference_time - anchor.low_time
        nearby = [
            other for other in candidates
            if abs(other.low_time - anchor.low_time) <= 30
            and abs((other.reference_time - other.low_time) - offset) <= 2 * sample_seconds
        ]
        if len(nearby) >= 3:
            kept.append(anchor)
    spans = {int(a.low_time // 60) for a in kept}
    verified = len(kept) * sample_seconds
    if verified < 30 or len(spans) < 3:
        return AlignmentReport(
            "unpaired", min(0.49, verified / 60), verified, tuple(kept[:100]),
            "Potential matches were too sparse for paired supervision; using synthetic reference adaptation.",
        )
    confidence = min(1.0, float(np.mean([a.score for a in kept])) * min(1.0, verified / 90))
    return AlignmentReport("paired", confidence, verified, tuple(kept[:500]), None)
