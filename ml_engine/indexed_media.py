"""Presentation-order frame identities. Average FPS is never used for seeking."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np


def cancelled(check):
    if check and check():
        raise InterruptedError('Operation cancelled')


def identity(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def descriptor(frame):
    # An equal normalized field of view tolerates small aspect differences without cropping.
    return cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 24), interpolation=cv2.INTER_AREA).ravel()


class FrameIndex:
    def __init__(self, directory):
        self.directory = Path(directory)
        with sqlite3.connect(f'file:{self.directory / "frames.sqlite3"}?mode=ro', uri=True) as db:
            self.meta = json.loads(db.execute('select value from metadata').fetchone()[0])
            self.rows = db.execute('select ordinal,pts,duration,keyframe from frames order by ordinal').fetchall()
        self.time_base = Fraction(self.meta['time_base'])
        self.origin = self.rows[0][1]
        self.times = np.array([float((r[1] - self.origin) * self.time_base) for r in self.rows])
        self.features = np.load(self.directory / 'descriptors.npy', mmap_mode='r')

    def __len__(self):
        return len(self.rows)

    def frame(self, ordinal):
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or not 0 <= ordinal < len(self):
            raise ValueError('Frame number is outside the source')
        _, pts, duration, keyframe = self.rows[ordinal]
        return {'frame_index': ordinal, 'frame_number': ordinal + 1, 'pts': pts,
                'time_base': str(self.time_base), 'time_seconds': float((pts - self.origin) * self.time_base),
                'duration_seconds': float(duration * self.time_base), 'source_id': self.meta['source_id']}

    def interval(self, start, end):
        if not 0 <= start < end <= len(self):
            raise ValueError('Invalid half-open source range')
        first = self.rows[start][1]
        last = self.rows[end][1] if end < len(self) else self.rows[-1][1] + self.rows[-1][2]
        return first * self.time_base, last * self.time_base

    @property
    def duration(self):
        a, b = self.interval(0, len(self))
        return b - a

    def nearest(self, seconds):
        i = int(np.searchsorted(self.times, seconds))
        choices = {max(0, min(len(self)-1, i)), max(0, min(len(self)-1, i-1))}
        return min(choices, key=lambda j: abs(self.times[j] - seconds))


def build_index(path, directory, progress=None, cancel=None):
    path, directory = Path(path).resolve(), Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    source_id = identity(path)
    if (directory / 'frames.sqlite3').exists():
        old = FrameIndex(directory)
        if old.meta['source_id'] == source_id:
            return old
    rows, features, issues = [], [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        tb = stream.time_base
        for ordinal, frame in enumerate(container.decode(stream)):
            cancelled(cancel)
            if frame.pts is None:
                raise ValueError('Source has missing presentation timestamps; repair its timing before importing')
            pts = int(Fraction(frame.pts) * frame.time_base / tb)
            if rows and pts <= rows[-1][1]:
                raise ValueError('Source has duplicate or nonmonotonic timestamps; explicit timing repair is required')
            duration = int(Fraction(frame.duration or 0) * frame.time_base / tb)
            rows.append((ordinal, pts, duration, int(frame.key_frame)))
            features.append(descriptor(frame.to_ndarray(format='bgr24')))
            if progress and ordinal % 120 == 0:
                progress(ordinal, stream.frames or None)
        if not rows:
            raise ValueError('No decodable video frames')
        for i in range(len(rows)-1):
            rows[i] = (*rows[i][:2], rows[i+1][1]-rows[i][1], rows[i][3])
        if rows[-1][2] <= 0:
            tail = rows[-2][2] if len(rows) > 1 else 0
            if tail <= 0:
                raise ValueError('Final frame duration is missing and cannot be estimated')
            rows[-1] = (*rows[-1][:2], tail, rows[-1][3])
            issues.append('Final frame duration estimated from preceding display interval')
        sar = stream.sample_aspect_ratio or Fraction(1)
        meta = {'source_id': source_id, 'path': str(path), 'time_base': str(tb),
                'width': stream.codec_context.width, 'height': stream.codec_context.height,
                'sample_aspect_ratio': f'{sar.numerator}:{sar.denominator}',
                'has_audio': bool(container.streams.audio), 'issues': issues,
                'variable_frame_rate': len({r[2] for r in rows[:-1]}) > 1, 'frame_count': len(rows)}
    temp = directory / 'frames.partial.sqlite3'
    temp.unlink(missing_ok=True)
    with sqlite3.connect(temp) as db:
        db.execute('create table frames(ordinal integer primary key, pts integer, duration integer, keyframe integer)')
        db.executemany('insert into frames values(?,?,?,?)', rows)
        db.execute('create table metadata(value text)')
        db.execute('insert into metadata values(?)', (json.dumps(meta),))
    np.save(directory / 'descriptors.npy', np.stack(features))
    temp.replace(directory / 'frames.sqlite3')
    return FrameIndex(directory)


class FrameReader:
    def __init__(self, index):
        self.index = index
        self.container = av.open(index.meta['path'])
        self.stream = self.container.streams.video[0]
        self.iterator = None
        self.position = -1

    def read(self, ordinal):
        self.index.frame(ordinal)
        if self.iterator is None or ordinal <= self.position or ordinal > self.position + 120:
            key = ordinal
            while key > 0 and not self.index.rows[key][3]:
                key -= 1
            self.container.seek(self.index.rows[key][1], stream=self.stream, backward=True)
            self.iterator = iter(self.container.decode(self.stream))
        target_pts = self.index.rows[ordinal][1]
        for frame in self.iterator:
            pts = int(Fraction(frame.pts) * frame.time_base / self.index.time_base)
            if pts == target_pts:
                self.position = ordinal
                return frame.to_ndarray(format='bgr24')
            if pts > target_pts:
                break
        raise ValueError(f'Could not decode indexed frame {ordinal+1}')

    def close(self):
        self.container.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def fit_frame(frame, size, mode='fit'):
    w, h = size
    scale = (max if mode == 'fill' else min)(w/frame.shape[1], h/frame.shape[0])
    resized = cv2.resize(frame, (max(1, round(frame.shape[1]*scale)), max(1, round(frame.shape[0]*scale))), interpolation=cv2.INTER_LANCZOS4)
    if mode == 'fill':
        y, x = (resized.shape[0]-h)//2, (resized.shape[1]-w)//2
        return resized[y:y+h, x:x+w]
    output = np.zeros((h,w,3), np.uint8)
    y, x = (h-resized.shape[0])//2, (w-resized.shape[1])//2
    output[y:y+resized.shape[0],x:x+resized.shape[1]] = resized
    return output


def picture(frame, index, crop=None):
    if crop:
        left, top, right, bottom = crop
        frame = frame[top:bottom, left:right]
    sar = Fraction(index.meta['sample_aspect_ratio'].replace(':','/'))
    if sar != 1:
        frame = cv2.resize(frame, (round(frame.shape[1]*sar), frame.shape[0]))
    return frame
