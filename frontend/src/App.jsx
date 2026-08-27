import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react'
import axios from 'axios'
import './App.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'
const terminal = new Set(['completed', 'failed', 'cancelled'])
const selectedJobKey = 'refsr-selected-job'
const ORIGIN = API.replace('/api/v1', '')
const streamLabel = {
  low: 'Supplemental · low resolution',
  reference: 'Reference · high resolution',
}
const trackLabel = { low: 'Low res', reference: 'High res' }

function Icon({ name }) {
  const paths = {
    play: <path d="m8 5 11 7-11 7Z" />,
    pause: <><path d="M8 5v14" /><path d="M16 5v14" /></>,
    earlier10: <><path d="M6 5v14" /><path d="m17 7-7 5 7 5" /></>,
    earlier: <path d="m15 6-6 6 6 6" />,
    later: <path d="m9 6 6 6-6 6" />,
    later10: <><path d="M18 5v14" /><path d="m7 7 7 5-7 5" /></>,
    closest: <><circle cx="10" cy="10" r="5" /><path d="m14 14 5 5M18 4v4M16 6h4" /></>,
    markIn: <><path d="M5 4v16" /><path d="m18 7-7 5 7 5" /></>,
    markOut: <><path d="M19 4v16" /><path d="m6 7 7 5-7 5" /></>,
    locate: <><circle cx="12" cy="12" r="5" /><path d="M12 2v3M12 19v3M2 12h3M19 12h3" /></>,
    save: <><path d="M5 4h11l3 3v13H5Z" /><path d="M8 4v6h8V5M8 20v-6h8v6" /></>,
    apply: <><path d="M5 5h14v14H5Z" /><path d="m8 12 3 3 5-6" /></>,
    confirm: <path d="m5 12 4 4L19 6" />,
    discard: <><path d="M9 7H5v-4" /><path d="M5 7a8 8 0 1 1-1 8" /></>,
    unpaired: <><path d="m8 8-2-2a3 3 0 0 0-4 4l3 3a3 3 0 0 0 4 0l1-1" /><path d="m16 16 2 2a3 3 0 0 0 4-4l-3-3a3 3 0 0 0-4 0l-1 1M4 20 20 4" /></>,
    refresh: <><path d="M20 7v5h-5" /><path d="M4 17v-5h5M18.5 9A7 7 0 0 0 6 6.5L4 9M5.5 15A7 7 0 0 0 18 17.5l2-2.5" /></>,
  }
  return <svg className="button-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}

function matchingLabel(mode) {
  return mode === 'reference_only' ? 'reference only' : 'guided matching'
}

function formatTime(seconds = 0) {
  const whole = Math.max(0, seconds)
  const hours = Math.floor(whole / 3600)
  const minutes = Math.floor((whole % 3600) / 60)
  const secs = (whole % 60).toFixed(3).padStart(6, '0')
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${secs}`
}

function durationBetween(start, end) {
  return Math.max(0, end.time_seconds - start.time_seconds)
}

const boundaryKeys = ['low_start', 'low_end', 'reference_start', 'reference_end']

function withAdjustmentBaseline(segment) {
  if (segment.adjustment_baseline) return segment
  return {
    ...segment,
    adjustment_baseline: Object.fromEntries(boundaryKeys.map((key) => [key, segment[key].frame_index])),
  }
}

function withFreshAdjustmentBaseline(segment) {
  const rest = { ...segment }
  delete rest.adjustment_baseline
  return withAdjustmentBaseline(rest)
}

function frameShift(segment, key, frameIndex = segment[key].frame_index) {
  const baseline = segment.adjustment_baseline?.[key] ?? segment[key].frame_index
  const delta = frameIndex - baseline
  if (delta === 0) return { className: 'unchanged', label: 'No shift from proposal' }
  const count = Math.abs(delta)
  return {
    className: delta < 0 ? 'earlier' : 'later',
    label: `${delta > 0 ? '+' : '−'}${count} frame${count === 1 ? '' : 's'} ${delta < 0 ? 'earlier' : 'later'}`,
  }
}

function TooltipButton({ tooltip, wrapperClassName = '', children, ...buttonProps }) {
  const tooltipId = useId()
  const describedBy = [buttonProps['aria-describedby'], tooltipId].filter(Boolean).join(' ')

  return (
    <span
      className={`tooltip-button ${wrapperClassName}`.trim()}
      tabIndex={buttonProps.disabled ? 0 : undefined}
      aria-label={buttonProps.disabled && typeof children === 'string' ? `${children}. ${tooltip}` : undefined}
    >
      <button {...buttonProps} aria-describedby={describedBy}>{children}</button>
      <span className="button-tooltip" id={tooltipId} role="tooltip">{tooltip}</span>
    </span>
  )
}

function TooltipCheckbox({ tooltip, children, ...inputProps }) {
  const tooltipId = useId()
  return (
    <label className="tooltip-checkbox">
      <input {...inputProps} type="checkbox" aria-describedby={tooltipId} />
      <span>{children}</span>
      <i aria-hidden="true">?</i>
      <span className="button-tooltip" id={tooltipId} role="tooltip">{tooltip}</span>
    </label>
  )
}

function CancelJobsDialog({ activeCount, busy, error, onClose, onConfirm }) {
  const titleId = useId()
  const descriptionId = useId()
  const dialogRef = useRef(null)
  const cancelRef = useRef(null)

  useEffect(() => {
    cancelRef.current?.focus()
  }, [])

  useEffect(() => {
    function keydown(event) {
      if (event.key === 'Escape' && !busy) {
        event.preventDefault()
        onClose()
      }
      if (event.key !== 'Tab') return
      const controls = [...(dialogRef.current?.querySelectorAll('button:not(:disabled)') || [])]
      if (!controls.length) return
      const first = controls[0]
      const last = controls[controls.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    window.addEventListener('keydown', keydown)
    return () => window.removeEventListener('keydown', keydown)
  }, [busy, onClose])

  return (
    <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) onClose() }}>
      <section ref={dialogRef} className="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby={titleId} aria-describedby={descriptionId}>
        <div className="dialog-mark" aria-hidden="true">!</div>
        <p className="eyebrow">CANCEL ACTIVE JOBS</p>
        <h2 id={titleId}>Stop all active work?</h2>
        <p id={descriptionId} className="dialog-copy">
          This will cancel {activeCount} queued, processing, or review {activeCount === 1 ? 'job' : 'jobs'}. Completed jobs and their files will remain available.
        </p>
        <p className="dialog-consequence">A running GPU step may take a moment to stop safely.</p>
        {error && <p className="alert error" role="alert">{error}</p>}
        <div className="dialog-actions">
          <button ref={cancelRef} type="button" onClick={onClose} disabled={busy}>Keep jobs running</button>
          <button className="destructive" type="button" onClick={onConfirm} disabled={busy}>{busy ? 'Cancelling…' : `Cancel ${activeCount} active ${activeCount === 1 ? 'job' : 'jobs'}`}</button>
        </div>
      </section>
    </div>
  )
}

function streamWindow(span, spanIndex, stream, media) {
  const range = span?.[`${stream}_range`]
  if (!range) return null
  const sourceDuration = (range.end_frame - range.start_frame) / media[stream].fps
  if (span.kind === 'match') {
    return { start: span.sequence_start_seconds, end: span.sequence_start_seconds + span.sequence_duration_seconds }
  }
  // A leading source-only section is right-aligned against the first shared
  // frame. Later differences start at the preceding shared cut.
  const start = spanIndex === 0
    ? span.sequence_start_seconds + span.sequence_duration_seconds - sourceDuration
    : span.sequence_start_seconds
  return { start, end: start + sourceDuration }
}

function locationAt(spans, time, total) {
  if (!spans.length) return null
  const clamped = Math.max(0, Math.min(total, Number(time)))
  return spans.find((span) => clamped >= span.sequence_start_seconds
    && clamped < span.sequence_start_seconds + span.sequence_duration_seconds)
    || spans[spans.length - 1]
}

function matchDraftForSpan(span) {
  if (span?.match_draft) {
    return {
      lowIn: span.match_draft.low_in, lowOut: span.match_draft.low_out,
      referenceIn: span.match_draft.reference_in, referenceOut: span.match_draft.reference_out,
      inTime: span.match_draft.in_time, outTime: span.match_draft.out_time,
    }
  }
  if (span?.kind === 'match') {
    return {
      lowIn: span.low_range.start_frame, lowOut: span.low_range.end_frame - 1,
      referenceIn: span.reference_range.start_frame, referenceOut: span.reference_range.end_frame - 1,
      inTime: span.sequence_start_seconds,
      outTime: span.sequence_start_seconds + span.sequence_duration_seconds,
    }
  }
  return { lowIn: null, lowOut: null, referenceIn: null, referenceOut: null, inTime: null, outTime: null }
}

function matchPresentation(span) {
  if (span.kind !== 'match') return { className: 'difference', label: 'Unpaired' }
  if (span.match_draft) return { className: 'draft', label: 'Saved adjustment' }
  if (span.origin === 'manual' && span.status === 'confirmed') return { className: 'confirmed adjusted', label: 'Adjusted · confirmed' }
  if (span.origin === 'manual') return { className: 'adjusted', label: 'Adjusted · review' }
  if (span.status === 'confirmed') return { className: 'confirmed', label: 'Confirmed' }
  return { className: 'proposed', label: 'Automatic proposal' }
}

function adjustmentDelta(value, baseline) {
  if (value == null || baseline == null || value === baseline) return null
  const delta = value - baseline
  return `${delta > 0 ? '+' : '−'}${Math.abs(delta)}`
}

function UnifiedTracks({ spans, media, total, selectedId, onSelect, onSeek, playhead = null, marks = [] }) {
  const width = Math.max(.001, total)
  return (
    <div className="unified-tracks" aria-label="Unified video timeline">
      {['reference', 'low'].map((stream) => <div className="unified-track-row" key={stream}>
        <span>{trackLabel[stream]}</span>
        <div className="unified-track">
          {spans.map((span) => {
            const range = span[`${stream}_range`]
            if (!range) return null
            const spanIndex = spans.findIndex((item) => item.id === span.id)
            const sourceWindow = streamWindow(span, spanIndex, stream, media)
            const presentation = matchPresentation(span)
            const className = `${span.kind} ${presentation.className}${span.id === selectedId ? ' selected' : ''}`
            return <button
              type="button"
              key={`${stream}-${span.id}`}
              className={className}
              style={{ left: `${sourceWindow.start / width * 100}%`, width: `${Math.max(.2, (sourceWindow.end - sourceWindow.start) / width * 100)}%` }}
              aria-label={`${span.kind === 'match' ? presentation.label : 'unpaired'} ${stream} block, frames ${range.start_frame} through ${range.end_frame - 1}`}
              title={`${span.kind === 'match' ? presentation.label : 'Unpaired footage'} · frames ${range.start_frame}–${range.end_frame - 1}`}
              onClick={() => onSelect(span.id)}
            ><span>{presentation.label}</span></button>
          })}
          {marks.map((mark) => <i
            className={`match-mark ${mark.edge}`}
            key={`${stream}-${mark.edge}`}
            style={{ left: `${mark.time / width * 100}%` }}
          ><span>{mark.edge}</span></i>)}
          {playhead != null && <i className="timeline-playhead" style={{ left: `${playhead / width * 100}%` }} />}
        </div>
      </div>)}
      {onSeek && <input
        className="timeline-direct-scrubber"
        aria-label="Unified timeline position"
        aria-valuetext={`${formatTime(playhead)} of ${formatTime(total)}`}
        title="Click or drag across the tracks to scrub both videos"
        type="range" min="0" max={total} step="0.001" value={playhead ?? 0}
        onChange={(event) => onSeek(event.target.value)}
      />}
    </div>
  )
}

function UnifiedMatchReview({ job, initialReview, onQueued }) {
  const [review, setReview] = useState(initialReview)
  const firstMatch = review.spans.find((span) => span.kind === 'match') || review.spans[0]
  const [selectedId, setSelectedId] = useState(firstMatch?.id || null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [playheads, setPlayheads] = useState({
    low: firstMatch?.low_range?.start_frame ?? 0,
    reference: firstMatch?.reference_range?.start_frame ?? 0,
  })
  const [sequenceTime, setSequenceTime] = useState(firstMatch?.sequence_start_seconds || 0)
  const [playing, setPlaying] = useState(false)
  const [matchDraft, setMatchDraft] = useState(null)
  const [draftDirty, setDraftDirty] = useState(false)
  const [draftSaveState, setDraftSaveState] = useState(firstMatch?.match_draft ? 'saved' : 'idle')
  const lowVideo = useRef(null)
  const referenceVideo = useRef(null)
  const sequenceTimeRef = useRef(sequenceTime)
  const syncedSequenceRef = useRef(null)
  const playingSpanRef = useRef({ low: null, reference: null })
  const total = Math.max(.001, review.spans.reduce((value, span) => Math.max(value, span.sequence_start_seconds + span.sequence_duration_seconds), 0))
  const selected = review.spans.find((span) => span.id === selectedId) || review.spans[0]

  const sourcePosition = useCallback((time, stream) => {
    const span = locationAt(review.spans, time, total)
    if (!span) return null
    const spanIndex = review.spans.findIndex((item) => item.id === span.id)
    const range = span[`${stream}_range`]
    const window = streamWindow(span, spanIndex, stream, review.media)
    if (!range || !window || time < window.start || time > window.end + .0001) return null
    const anchorFrame = matchDraft?.[`${stream}In`]
    if (span.id === selected.id && matchDraft?.inTime != null && anchorFrame != null && time >= matchDraft.inTime) {
      const frame = anchorFrame + Math.round((time - matchDraft.inTime) * review.media[stream].fps)
      if (frame >= range.start_frame && frame < range.end_frame) {
        return { span, range, window, frame, time: frame / review.media[stream].fps }
      }
    }
    const progress = Math.max(0, Math.min(1, (time - window.start) / Math.max(.001, window.end - window.start)))
    const frame = Math.min(range.end_frame - 1, range.start_frame + Math.round(progress * Math.max(0, range.end_frame - range.start_frame - 1)))
    return { span, range, window, frame, time: frame / review.media[stream].fps }
  }, [matchDraft, review.media, review.spans, selected.id, total])

  useEffect(() => {
    sequenceTimeRef.current = sequenceTime
    const positions = { low: sourcePosition(sequenceTime, 'low'), reference: sourcePosition(sequenceTime, 'reference') }
    if (syncedSequenceRef.current !== sequenceTime) {
      syncedSequenceRef.current = sequenceTime
      setPlayheads((current) => ({
        low: positions.low?.frame ?? current.low,
        reference: positions.reference?.frame ?? current.reference,
      }))
    }
    for (const stream of ['low', 'reference']) {
      const video = stream === 'low' ? lowVideo.current : referenceVideo.current
      const position = positions[stream]
      if (!video) continue
      if (!position) {
        video.pause()
        playingSpanRef.current[stream] = null
        continue
      }
      const target = position.time
      if (!playing || Math.abs(video.currentTime - target) > .15) video.currentTime = target
      const sourceDuration = (position.range.end_frame - position.range.start_frame) / review.media[stream].fps
      const timelineDuration = position.window.end - position.window.start
      video.playbackRate = Math.max(.25, Math.min(4, sourceDuration / Math.max(.001, timelineDuration)))
      if (playing && playingSpanRef.current[stream] !== position.span.id) {
        playingSpanRef.current[stream] = position.span.id
        const attempt = video.play()
        if (attempt) attempt.catch(() => {
          setPlaying(false)
          setError('Playback could not start. Allow video playback in the browser and try again.')
        })
      }
    }
  }, [playing, review.media, sequenceTime, sourcePosition])

  useEffect(() => {
    if (!playing) {
      ;[lowVideo.current, referenceVideo.current].forEach((video) => video?.pause())
      playingSpanRef.current = { low: null, reference: null }
      return undefined
    }
    let animationFrame
    let prior = performance.now()
    function advance(now) {
      const next = Math.min(total, sequenceTimeRef.current + (now - prior) / 1000)
      prior = now
      sequenceTimeRef.current = next
      setSequenceTime(next)
      const span = locationAt(review.spans, next, total)
      if (span) setSelectedId(span.id)
      if (next >= total) setPlaying(false)
      else animationFrame = requestAnimationFrame(advance)
    }
    animationFrame = requestAnimationFrame(advance)
    return () => cancelAnimationFrame(animationFrame)
  }, [playing, review.spans, total])

  useEffect(() => {
    setMatchDraft(matchDraftForSpan(selected))
    setDraftDirty(false)
    setDraftSaveState(selected.match_draft ? 'saved' : 'idle')
  }, [review.revision, selected.id]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!draftDirty) return undefined
    function beforeUnload(event) {
      event.preventDefault()
      event.returnValue = ''
    }
    window.addEventListener('beforeunload', beforeUnload)
    return () => window.removeEventListener('beforeunload', beforeUnload)
  }, [draftDirty])

  async function edit(operation) {
    if (!selected || saving) return null
    setSaving(true)
    try {
      const response = await axios.patch(`${API}/jobs/${job.id}/match-review`, {
        revision: review.revision, span_id: selected.id, ...operation,
      })
      setReview(response.data)
      const stillSelected = response.data.spans.some((span) => span.id === selected.id)
      if (!stillSelected) {
        setSelectedId(response.data.spans.find((span) => span.kind === 'match' && span.status === 'proposed')?.id || response.data.spans[0]?.id)
      }
      setError('')
      return response.data
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
      return null
    } finally { setSaving(false) }
  }

  async function approve(mode) {
    setSaving(true)
    try {
      const response = await axios.post(`${API}/jobs/${job.id}/match-review/approve`, { revision: review.revision, mode })
      onQueued(response.data)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    } finally { setSaving(false) }
  }

  async function reanalyze() {
    if (!window.confirm('Rebuild the automatic alignment? This discards all current match decisions.')) return
    setSaving(true)
    try {
      const response = await axios.post(`${API}/jobs/${job.id}/match-review/reanalyze`, { revision: review.revision, confirm_discard: true })
      onQueued(response.data)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    } finally { setSaving(false) }
  }

  function seekSequence(time) {
    const next = Math.max(0, Math.min(total, Number(time)))
    const span = locationAt(review.spans, next, total)
    if (span && span.id !== selected.id && draftDirty && !window.confirm('Discard the unsaved Match In/Out changes?')) return
    syncedSequenceRef.current = null
    sequenceTimeRef.current = next
    setSequenceTime(next)
    if (span) setSelectedId(span.id)
  }

  function selectSpan(id) {
    const span = review.spans.find((item) => item.id === id)
    if (id !== selected.id && draftDirty && !window.confirm('Discard the unsaved Match In/Out changes?')) return
    setSelectedId(id)
    if (span) {
      const next = span.sequence_start_seconds + Math.min(.001, span.sequence_duration_seconds / 2)
      syncedSequenceRef.current = null
      sequenceTimeRef.current = next
      setSequenceTime(next)
    }
  }

  function togglePlayback() {
    if (playing) {
      setPlaying(false)
      return
    }
    if (sequenceTime >= total - .001) seekSequence(0)
    playingSpanRef.current = { low: null, reference: null }
    setPlaying(true)
  }

  function jogPlayhead(stream, nextFrame) {
    const range = editableRange(stream)
    if (!range) return
    setPlaying(false)
    const frame = Math.max(range.start_frame, Math.min(range.end_frame - 1, Number(nextFrame)))
    const video = stream === 'low' ? lowVideo.current : referenceVideo.current
    if (video) video.currentTime = frame / review.media[stream].fps
    setPlayheads((current) => ({ ...current, [stream]: frame }))
  }

  function stepPlayhead(stream, amount) { jogPlayhead(stream, playheads[stream] + amount) }

  async function markPair(edge) {
    if (!sourcePosition(sequenceTime, 'low') || !sourcePosition(sequenceTime, 'reference')) {
      setError('Both videos need footage at the shared playhead before you can mark a pair.')
      return
    }
    const nextDraft = {
      ...matchDraft,
      lowIn: edge === 'in' ? playheads.low : matchDraft.lowIn,
      lowOut: edge === 'out' ? playheads.low : matchDraft.lowOut,
      referenceIn: edge === 'in' ? playheads.reference : matchDraft.referenceIn,
      referenceOut: edge === 'out' ? playheads.reference : matchDraft.referenceOut,
      inTime: edge === 'in' ? sequenceTime : matchDraft.inTime,
      outTime: edge === 'out' ? sequenceTime : matchDraft.outTime,
    }
    setMatchDraft(nextDraft)
    setDraftDirty(true)
    setDraftSaveState('saving')
    setError('')
    const saved = await edit({
      operation: 'set_match_draft',
      draft: {
        low_in: nextDraft.lowIn, low_out: nextDraft.lowOut,
        reference_in: nextDraft.referenceIn, reference_out: nextDraft.referenceOut,
        in_time: nextDraft.inTime, out_time: nextDraft.outTime,
      },
    })
    if (saved) {
      setDraftDirty(false)
      setDraftSaveState('saved')
    } else {
      setDraftSaveState('failed')
    }
  }

  function goToMark(edge) {
    const low = matchDraft?.[`low${edge === 'in' ? 'In' : 'Out'}`]
    const reference = matchDraft?.[`reference${edge === 'in' ? 'In' : 'Out'}`]
    const time = matchDraft?.[`${edge}Time`]
    if (low == null || reference == null || time == null) return
    seekSequence(time)
    jogPlayhead('low', low)
    jogPlayhead('reference', reference)
  }

  async function snapPlayhead(targetStream) {
    if (saving) return
    const fixedStream = targetStream === 'low' ? 'reference' : 'low'
    setSaving(true)
    try {
      const response = await axios.post(`${API}/jobs/${job.id}/match-review/snap`, {
        fixed_stream: fixedStream,
        fixed_frame_index: playheads[fixedStream],
        target_frame_index: playheads[targetStream],
      })
      jogPlayhead(targetStream, response.data.frame.frame_index)
      setError('')
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    } finally { setSaving(false) }
  }

  async function applyMatchDraft() {
    if (!draftValidation.valid || draftDirty || !selected.match_draft || saving) return
    const appliedDraft = { ...matchDraft }
    const saved = await edit({ operation: 'apply_match_draft' })
    if (!saved) return
    if (selected.kind === 'difference') {
      const created = saved.spans.find((span) => span.kind === 'match'
        && span.low_range.start_frame === appliedDraft.lowIn
        && span.reference_range.start_frame === appliedDraft.referenceIn)
      if (created) setSelectedId(created.id)
    }
    setDraftDirty(false)
    setDraftSaveState('idle')
  }

  async function retryMatchDraft() {
    if (!draftDirty || saving) return
    setDraftSaveState('saving')
    const saved = await edit({
      operation: 'set_match_draft',
      draft: {
        low_in: matchDraft.lowIn, low_out: matchDraft.lowOut,
        reference_in: matchDraft.referenceIn, reference_out: matchDraft.referenceOut,
        in_time: matchDraft.inTime, out_time: matchDraft.outTime,
      },
    })
    if (saved) {
      setDraftDirty(false)
      setDraftSaveState('saved')
    } else {
      setDraftSaveState('failed')
    }
  }

  async function discardMatchDraft() {
    if (saving) return
    if (selected.match_draft) {
      const saved = await edit({ operation: 'set_match_draft', draft: null })
      if (!saved) return
    } else {
      setMatchDraft(matchDraftForSpan(selected))
    }
    setDraftDirty(false)
    setDraftSaveState('idle')
  }

  useEffect(() => {
    function keydown(event) {
      if (!selected || event.target.matches('input, button, select')) return
      const amount = event.shiftKey ? 10 : 1
      if (event.key === 'ArrowLeft') { event.preventDefault(); stepPlayhead('low', -amount) }
      if (event.key === 'ArrowRight') { event.preventDefault(); stepPlayhead('low', amount) }
      if (event.key === 'ArrowDown') { event.preventDefault(); stepPlayhead('reference', -amount) }
      if (event.key === 'ArrowUp') { event.preventDefault(); stepPlayhead('reference', amount) }
    }
    window.addEventListener('keydown', keydown)
    return () => window.removeEventListener('keydown', keydown)
  })

  const proposed = review.summary?.proposed_blocks || 0
  const confirmed = review.summary?.confirmed_blocks || 0
  const adjusted = review.spans.filter((span) => span.kind === 'match' && span.origin === 'manual').length
  const savedDrafts = review.spans.filter((span) => span.match_draft).length
  const differenceLabel = selected?.kind === 'difference'
    ? selected.low_range && selected.reference_range ? 'Both tracks contain different, unpaired footage here.'
      : selected.low_range ? 'Only the supplemental video has footage here.' : 'Only the reference video has footage here.'
    : null

  function editableRange(stream) {
    const own = selected?.[`${stream}_range`]
    if (!own || selected.kind === 'difference') return own
    const index = review.spans.findIndex((span) => span.id === selected.id)
    const prior = [...review.spans.slice(0, index)].reverse().find((span) => span.kind === 'match')
    const following = review.spans.slice(index + 1).find((span) => span.kind === 'match')
    return {
      start_frame: prior?.[`${stream}_range`]?.end_frame ?? 0,
      end_frame: following?.[`${stream}_range`]?.start_frame ?? review.media[stream].frame_count,
    }
  }

  const draftComplete = matchDraft && ['lowIn', 'lowOut', 'referenceIn', 'referenceOut']
    .every((key) => matchDraft[key] != null)
  let draftMessage = draftComplete ? '' : 'Set both Match In and Match Out frame pairs.'
  if (draftComplete && (matchDraft.lowIn > matchDraft.lowOut || matchDraft.referenceIn > matchDraft.referenceOut)) {
    draftMessage = 'Match Out cannot be earlier than Match In in either video.'
  }
  const lowDuration = draftComplete ? (matchDraft.lowOut - matchDraft.lowIn + 1) / review.media.low.fps : 0
  const referenceDuration = draftComplete ? (matchDraft.referenceOut - matchDraft.referenceIn + 1) / review.media.reference.fps : 0
  const durationTolerance = Math.max(.5 / review.media.low.fps, .5 / review.media.reference.fps) + 1e-6
  if (!draftMessage && Math.abs(lowDuration - referenceDuration) > durationTolerance) {
    const frames = Math.round(Math.abs(lowDuration - referenceDuration) * Math.max(review.media.low.fps, review.media.reference.fps))
    draftMessage = `The segment differs by about ${frames} frame${frames === 1 ? '' : 's'}. Mark Out before the missing footage and start a new segment after it.`
  }
  const draftValidation = { complete: Boolean(draftComplete), valid: Boolean(draftComplete && !draftMessage), message: draftMessage }
  const hasDraftChanges = draftDirty || Boolean(selected?.match_draft)
  const draftMarks = [
    matchDraft?.inTime != null ? { edge: 'in', time: matchDraft.inTime } : null,
    matchDraft?.outTime != null ? { edge: 'out', time: matchDraft.outTime } : null,
  ].filter(Boolean)

  function timelineFeed(stream) {
    const position = sourcePosition(sequenceTime, stream)
    const jogRange = editableRange(stream)
    if (!position) return <figure className={`timeline-video-feed no-footage ${stream}`} aria-label={`${streamLabel[stream]} playback feed`}>
      <div className="timeline-video-empty"><b>No footage</b><span>{streamLabel[stream]} has no source video at {formatTime(sequenceTime)}.</span></div>
      <figcaption><b>{streamLabel[stream]}</b><span>empty at unified playhead</span></figcaption>
    </figure>
    return <figure className={`timeline-video-feed ${stream}`} aria-label={`${streamLabel[stream]} playback feed`}>
      <video
        ref={stream === 'low' ? lowVideo : referenceVideo}
        muted playsInline preload="metadata" src={`${ORIGIN}${review.proxy_urls[stream]}`}
        poster={`${API}/jobs/${job.id}/frames/${stream}/${playheads[stream]}`}
      ><img className="video-fallback-frame" src={`${API}/jobs/${job.id}/frames/${stream}/${playheads[stream]}`} alt={`${streamLabel[stream]} frame ${playheads[stream]}`} /></video>
      <figcaption>
        <b>{streamLabel[stream]}</b>
        <span>frame {playheads[stream]} · source {formatTime(playheads[stream] / review.media[stream].fps)}</span>
        <span className="feed-step-controls">
          {[-10, -1, 1, 10].map((amount) => {
            const direction = amount < 0 ? 'Earlier' : 'Later'
            const icon = amount === -10 ? 'earlier10' : amount === -1 ? 'earlier' : amount === 1 ? 'later' : 'later10'
            const label = `${direction} ${Math.abs(amount)} ${streamLabel[stream]} frame${Math.abs(amount) === 1 ? '' : 's'}`
            return <TooltipButton key={amount} type="button" className="icon-button" onClick={() => stepPlayhead(stream, amount)} aria-label={label} tooltip={label}><Icon name={icon} /></TooltipButton>
          })}
          <TooltipButton type="button" className="icon-button" disabled={saving} onClick={() => snapPlayhead(stream)} aria-label={`Find closest ${streamLabel[stream]} frame`} tooltip="Search nearby frames for the closest visual match."><Icon name="closest" /></TooltipButton>
        </span>
        {jogRange && <label className="feed-jog"><span>Independent source jog</span><input aria-label={`Jog ${streamLabel[stream]} frame`} type="range" min={jogRange.start_frame} max={jogRange.end_frame - 1} step="1" value={playheads[stream]} onChange={(event) => jogPlayhead(stream, event.target.value)} /></label>}
      </figcaption>
    </figure>
  }

  return <div className="match-review unified-review">
    <div className="review-intro">
      <p className="eyebrow">UNIFIED FRAME ALIGNMENT</p>
      <h3>Align the shared edit, then approve training blocks.</h3>
      <p>Both tracks use one sequence axis. Solid blocks correspond frame-by-frame; hatched blocks are visible but excluded from paired training.</p>
    </div>
    <div className="review-summary">
      <div><b>{confirmed}</b><span>confirmed blocks</span></div>
      <div><b>{proposed}</b><span>needs review</span></div>
      <div><b>{adjusted}</b><span>user adjusted</span></div>
      <div><b>{formatTime(review.summary?.matched_seconds || 0)}</b><span>approved pairs</span></div>
    </div>

    <section className="review-stage unified-timeline-stage" aria-labelledby="alignment-map-heading">
      <div className="stage-heading-row">
        <div className="stage-heading"><span>1</span><div><p className="eyebrow">ALIGNMENT MAP</p><h3 id="alignment-map-heading">One sequence, two source tracks</h3></div></div>
        <TooltipButton type="button" className="icon-button" onClick={reanalyze} disabled={saving} aria-label="Rebuild alignment" tooltip="Rebuild the automatic alignment and discard current match decisions."><Icon name="refresh" /></TooltipButton>
      </div>
      <p className="stage-help">Jog either source to matching frames, then mark In or Out. Each mark saves automatically; Apply changes the highlighted range, and Confirm approves it for training.</p>
      <div className="coverage-key"><span><i className="confirmed" />Confirmed</span><span><i className="proposed" />Automatic proposal</span><span><i className="draft" />Saved adjustment</span><span><i className="adjusted" />Adjusted · review</span><span><i className="difference" />Unpaired</span></div>
      {timelineFeed('reference')}
      <UnifiedTracks spans={review.spans} media={review.media} total={total} selectedId={selected?.id} onSelect={selectSpan} onSeek={seekSequence} playhead={sequenceTime} marks={draftMarks} />
      <div className="unified-transport">
        <TooltipButton type="button" className="icon-button" onClick={togglePlayback} aria-label={playing ? 'Pause both timeline videos' : 'Play both timeline videos'} tooltip={playing ? 'Pause both timeline videos.' : 'Play both timeline videos.'}><Icon name={playing ? 'pause' : 'play'} /></TooltipButton>
        <output>Playhead {formatTime(sequenceTime)} / {formatTime(total)}</output>
      </div>
      {(selected?.kind === 'match' || (selected?.low_range && selected?.reference_range)) && <div className="timeline-match-toolbar" role="group" aria-label="Frame match controls">
        <div className="timeline-mark-pairs">
          {['in', 'out'].map((edge) => {
            const title = edge === 'in' ? 'Match In' : 'Match Out'
            const shortTitle = edge === 'in' ? 'In' : 'Out'
            const lowFrame = matchDraft?.[`low${edge === 'in' ? 'In' : 'Out'}`]
            const referenceFrame = matchDraft?.[`reference${edge === 'in' ? 'In' : 'Out'}`]
            const baselineKey = edge === 'in' ? 'in' : 'out'
            const lowDelta = adjustmentDelta(lowFrame, selected.adjustment_baseline?.[`low_${baselineKey}`])
            const referenceDelta = adjustmentDelta(referenceFrame, selected.adjustment_baseline?.[`reference_${baselineKey}`])
            return <div className={`timeline-mark-pair ${edge}`} key={edge}>
              <span className="timeline-mark-value"><b>{shortTitle}</b><span>{lowFrame == null ? 'Not set' : `L${lowFrame} ↔ H${referenceFrame}`}</span>{(lowDelta || referenceDelta) && <small>{lowDelta ? `L ${lowDelta}` : 'L —'} · {referenceDelta ? `H ${referenceDelta}` : 'H —'}</small>}</span>
              <div className="match-mark-actions">
                <TooltipButton type="button" className="icon-button" disabled={lowFrame == null || referenceFrame == null} onClick={() => goToMark(edge)} aria-label={`Go to ${title}`} tooltip={`Move both feeds to the saved ${title} pair.`}><Icon name="locate" /></TooltipButton>
                <TooltipButton type="button" className="icon-button primary" disabled={saving} aria-label={`Mark ${title} from playheads`} onClick={() => markPair(edge)} tooltip={`Set ${title} from the two current frames.`}><Icon name={edge === 'in' ? 'markIn' : 'markOut'} /></TooltipButton>
              </div>
            </div>
          })}
          <div className="timeline-match-actions">
            {selected.kind === 'match' && <TooltipButton type="button" className="icon-button" disabled={saving} onClick={() => edit({ operation: 'mark_unpaired' })} aria-label="Mark block unpaired" tooltip="Exclude this block from paired training."><Icon name="unpaired" /></TooltipButton>}
            <TooltipButton type="button" className="icon-button" disabled={saving || (!draftDirty && !selected.match_draft)} onClick={discardMatchDraft} aria-label="Discard frame draft" tooltip="Discard saved and unsaved In/Out changes for this block."><Icon name="discard" /></TooltipButton>
            {draftSaveState === 'failed' && <TooltipButton type="button" className="icon-button" disabled={saving} onClick={retryMatchDraft} aria-label="Retry saving frame draft" tooltip="Retry saving these In/Out marks without discarding them."><Icon name="save" /></TooltipButton>}
            <TooltipButton type="button" className="icon-button primary" disabled={saving || draftDirty || !selected.match_draft || !hasDraftChanges || !draftValidation.valid} onClick={applyMatchDraft} aria-label={selected.kind === 'match' ? 'Apply In / Out' : 'Create proposed match'} tooltip={selected.kind === 'match' ? 'Apply the saved In and Out pairs to this match.' : 'Create a proposed match from the saved frame pairs.'}><Icon name="apply" /></TooltipButton>
            {selected.kind === 'match' && <TooltipButton type="button" className="icon-button primary" disabled={saving || hasDraftChanges || !draftValidation.valid || selected.status === 'confirmed'} onClick={() => edit({ operation: 'set_status', status: 'confirmed' })} aria-label={selected.status === 'confirmed' ? 'Match confirmed' : 'Confirm match'} tooltip={selected.status === 'confirmed' ? 'This match is confirmed for paired training.' : 'Approve this applied match for paired training.'}><Icon name="confirm" /></TooltipButton>}
          </div>
        </div>
        {draftValidation.message && <p className="match-validation" role="status">{draftValidation.message}</p>}
        {draftSaveState === 'saving' && <p className="saving-draft" role="status">Saving adjustment…</p>}
        {draftSaveState === 'failed' && <p className="unsaved-draft" role="alert">Adjustment not saved · retry or discard before leaving</p>}
        {draftSaveState === 'saved' && selected.match_draft && <p className="saved-draft">Adjustment saved · Apply when ready</p>}
      </div>}
      {timelineFeed('low')}
      {selected && <p className={`selected-span-label ${selected.kind}`}>
        <b>{selected.kind === 'match' ? matchPresentation(selected).label : 'unpaired difference'}</b>
        <span>{formatTime(selected.sequence_duration_seconds)}</span>
        {selected.confidence != null && <span>{Math.round(selected.confidence * 100)}% confidence</span>}
      </p>}
      {differenceLabel && <p className="difference-notice">{differenceLabel}</p>}
      <p className="keyboard-help">Keyboard: ←/→ steps the supplemental frame; ↓/↑ steps the reference frame. Hold Shift for ten frames.</p>
    </section>
    {error && <p className="alert error" role="alert">{error}</p>}
    <div className="approval-actions">
      <p>Resolve each proposed match and apply or discard saved drafts. Difference blocks remain visible but never become training pairs.</p>
      <button type="button" onClick={() => approve('unpaired')} disabled={saving || proposed > 0 || savedDrafts > 0 || draftDirty}>Train without confirmed pairs</button>
      <button type="button" className="primary" onClick={() => approve('paired')} disabled={saving || proposed > 0 || confirmed === 0 || savedDrafts > 0 || draftDirty}>Use dense confirmed pairs and start processing</button>
    </div>
  </div>
}

function MatchReview({ job, onQueued }) {
  const [review, setReview] = useState(null)
  const [selected, setSelected] = useState(0)
  const [boundary, setBoundary] = useState('start')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [comparisonMode, setComparisonMode] = useState('side-by-side')
  const [overlayOpacity, setOverlayOpacity] = useState(50)
  const [playing, setPlaying] = useState(false)
  const [playbackProgress, setPlaybackProgress] = useState(0)
  const [playheads, setPlayheads] = useState({ low: 0, reference: 0 })
  const [boundaryDrafts, setBoundaryDrafts] = useState({ low: null, reference: null })
  const lowVideo = useRef(null)
  const referenceVideo = useRef(null)
  const sourceLowVideo = useRef(null)
  const sourceReferenceVideo = useRef(null)
  const sliderCommits = useRef({})

  const load = useCallback(async () => {
    try {
      const response = await axios.get(`${API}/jobs/${job.id}/match-review`)
      setReview(response.data)
      setError('')
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    }
  }, [job.id])

  useEffect(() => { load() }, [load])
  const segment = review?.segments?.[selected] || null
  const segmentId = segment?.id
  const segmentLowStart = segment?.low_start.time_seconds
  const segmentReferenceStart = segment?.reference_start.time_seconds
  const lowBoundaryFrame = segment?.[`low_${boundary}`]?.frame_index
  const referenceBoundaryFrame = segment?.[`reference_${boundary}`]?.frame_index

  useEffect(() => {
    ;[lowVideo.current, referenceVideo.current].forEach((video) => video?.pause())
    setPlaying(false)
    setPlaybackProgress(0)
    if (segmentId) {
      setPlayheads({ low: segmentLowStart, reference: segmentReferenceStart })
      if (lowVideo.current) lowVideo.current.currentTime = segmentLowStart
      if (referenceVideo.current) referenceVideo.current.currentTime = segmentReferenceStart
    }
  }, [selected, segmentId, segmentLowStart, segmentReferenceStart])

  useEffect(() => {
    setBoundaryDrafts({ low: lowBoundaryFrame ?? null, reference: referenceBoundaryFrame ?? null })
    sliderCommits.current = {}
  }, [selected, boundary, lowBoundaryFrame, referenceBoundaryFrame])

  useEffect(() => {
    if (!segment) return undefined
    const videos = { low: lowVideo.current, reference: referenceVideo.current }
    function update(stream) {
      const video = videos[stream]
      if (!video) return
      setPlayheads((current) => ({ ...current, [stream]: video.currentTime }))
      if (stream === 'low') {
        const duration = durationBetween(segment.low_start, segment.low_end) || 1
        setPlaybackProgress(Math.max(0, Math.min(1, (video.currentTime - segment.low_start.time_seconds) / duration)))
      }
      const end = segment[`${stream}_end`].time_seconds
      if (playing && video.currentTime >= end - .002) {
        Object.values(videos).forEach((item) => item?.pause())
        setPlaying(false)
      }
    }
    const lowUpdate = () => update('low')
    const referenceUpdate = () => update('reference')
    videos.low?.addEventListener('timeupdate', lowUpdate)
    videos.reference?.addEventListener('timeupdate', referenceUpdate)
    return () => {
      videos.low?.removeEventListener('timeupdate', lowUpdate)
      videos.reference?.removeEventListener('timeupdate', referenceUpdate)
    }
  }, [playing, segment])

  async function persist(segments) {
    setSaving(true)
    try {
      const response = await axios.put(`${API}/jobs/${job.id}/match-review`, {
        revision: review.revision, segments,
      })
      setReview((current) => ({ ...current, ...response.data }))
      setError('')
      return response.data
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
      return null
    } finally {
      setSaving(false)
    }
  }

  function changedSegment(mutator) {
    const segments = review.segments.map((item, index) => index === selected ? mutator(item) : item)
    return persist(segments)
  }

  async function resolveSelected(status) {
    const segments = review.segments.map((item, index) => index === selected ? { ...item, status } : item)
    const saved = await persist(segments)
    if (!saved) return
    const next = saved.segments.findIndex((item, index) => index > selected && item.status === 'proposed')
    if (next >= 0) setSelected(next)
    else {
      const first = saved.segments.findIndex((item) => item.status === 'proposed')
      if (first >= 0) setSelected(first)
    }
  }

  function step(stream, amount) {
    if (!segment || saving) return
    const key = `${stream}_${boundary}`
    const info = review.media[stream]
    changedSegment((item) => {
      const based = withAdjustmentBaseline(item)
      const lower = boundary === 'start' ? 0 : based[`${stream}_start`].frame_index + 1
      const upper = boundary === 'start' ? based[`${stream}_end`].frame_index - 1 : info.frame_count - 1
      const next = Math.max(lower, Math.min(upper, based[key].frame_index + amount))
      return { ...based, [key]: { frame_index: next, pts: next, time_seconds: next / info.fps }, status: 'proposed', origin: 'manual' }
    })
  }

  function sliderRange(stream) {
    const info = review.media[stream]
    const start = segment[`${stream}_start`].frame_index
    const end = segment[`${stream}_end`].frame_index
    const baselineStart = segment.adjustment_baseline?.[`${stream}_start`] ?? start
    const baselineEnd = segment.adjustment_baseline?.[`${stream}_end`] ?? end
    const outsideFrames = Math.max(10, Math.round(info.fps))
    return boundary === 'start'
      ? { min: Math.max(0, Math.min(start, baselineStart - outsideFrames)), max: Math.max(start, end - 1) }
      : { min: Math.min(end, start + 1), max: Math.min(info.frame_count - 1, Math.max(end, baselineEnd + outsideFrames)) }
  }

  function updateBoundarySliders(stream, nextFrame) {
    const otherStream = stream === 'low' ? 'reference' : 'low'
    const key = `${stream}_${boundary}`
    const otherKey = `${otherStream}_${boundary}`
    const deltaSeconds = (nextFrame - segment[key].frame_index) / review.media[stream].fps
    const projectedOtherFrame = segment[otherKey].frame_index + Math.round(deltaSeconds * review.media[otherStream].fps)
    const otherRange = sliderRange(otherStream)
    const otherFrame = Math.max(otherRange.min, Math.min(otherRange.max, projectedOtherFrame))
    setBoundaryDrafts((current) => ({ ...current, [stream]: nextFrame, [otherStream]: otherFrame }))
  }

  async function commitBoundarySliders() {
    if (!segment || saving) return
    const changes = ['low', 'reference'].filter((stream) => {
      const next = boundaryDrafts[stream]
      return next !== null && next !== segment[`${stream}_${boundary}`].frame_index
    })
    if (!changes.length) return
    const commitKey = `${boundary}:${boundaryDrafts.low}:${boundaryDrafts.reference}`
    if (sliderCommits.current[commitKey]) return
    sliderCommits.current[commitKey] = true
    const saved = await changedSegment((item) => {
      let based = withAdjustmentBaseline(item)
      for (const stream of changes) {
        const key = `${stream}_${boundary}`
        const next = boundaryDrafts[stream]
        based = { ...based, [key]: { frame_index: next, pts: next, time_seconds: next / review.media[stream].fps } }
      }
      return { ...based, status: 'proposed', origin: 'manual' }
    })
    if (!saved) {
      delete sliderCommits.current[commitKey]
      setBoundaryDrafts({
        low: segment[`low_${boundary}`].frame_index,
        reference: segment[`reference_${boundary}`].frame_index,
      })
    }
  }

  useEffect(() => {
    function keydown(event) {
      if (!segment || event.target.matches('input, button, select')) return
      const amount = event.shiftKey ? 10 : 1
      if (event.key === 'ArrowLeft') { event.preventDefault(); step('low', -amount) }
      if (event.key === 'ArrowRight') { event.preventDefault(); step('low', amount) }
      if (event.key === 'ArrowDown') { event.preventDefault(); step('reference', -amount) }
      if (event.key === 'ArrowUp') { event.preventDefault(); step('reference', amount) }
    }
    window.addEventListener('keydown', keydown)
    return () => window.removeEventListener('keydown', keydown)
  })

  async function addFromPlayheads() {
    const lowFrame = Math.round((sourceLowVideo.current?.currentTime || 0) * review.media.low.fps)
    const referenceFrame = Math.round((sourceReferenceVideo.current?.currentTime || 0) * review.media.reference.fps)
    setSaving(true)
    try {
      const response = await axios.post(`${API}/jobs/${job.id}/match-review/expand`, {
        low_frame_index: lowFrame, reference_frame_index: referenceFrame,
      })
      const saved = await persist([...review.segments, response.data])
      if (saved) setSelected(saved.segments.length - 1)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    } finally { setSaving(false) }
  }

  function splitAtPlayheads() {
    if (!segment) return
    const lowIndex = Math.round((lowVideo.current?.currentTime || segment.low_start.time_seconds) * review.media.low.fps)
    const refIndex = Math.round((referenceVideo.current?.currentTime || segment.reference_start.time_seconds) * review.media.reference.fps)
    if (lowIndex <= segment.low_start.frame_index || lowIndex >= segment.low_end.frame_index || refIndex <= segment.reference_start.frame_index || refIndex >= segment.reference_end.frame_index) {
      setError('Move both playheads inside the selected segment before splitting.')
      return
    }
    const lowRef = { frame_index: lowIndex, pts: lowIndex, time_seconds: lowIndex / review.media.low.fps }
    const highRef = { frame_index: refIndex, pts: refIndex, time_seconds: refIndex / review.media.reference.fps }
    const left = withFreshAdjustmentBaseline({ ...segment, id: `${segment.id}-a`, low_end: lowRef, reference_end: highRef, status: 'proposed', origin: 'manual' })
    const right = withFreshAdjustmentBaseline({ ...segment, id: `${segment.id}-b`, low_start: lowRef, reference_start: highRef, status: 'proposed', origin: 'manual' })
    persist(review.segments.flatMap((item, index) => index === selected ? [left, right] : [item]))
  }

  function mergeNext() {
    if (!segment || selected >= review.segments.length - 1) return
    const next = review.segments[selected + 1]
    const merged = withFreshAdjustmentBaseline({ ...segment, id: `${segment.id}-merged`, low_end: next.low_end, reference_end: next.reference_end, status: 'proposed', origin: 'manual' })
    persist(review.segments.filter((_, index) => index !== selected + 1).map((item, index) => index === selected ? merged : item))
  }

  async function snap(targetStream) {
    if (!segment || saving) return
    const fixedStream = targetStream === 'low' ? 'reference' : 'low'
    setSaving(true)
    try {
      const response = await axios.post(`${API}/jobs/${job.id}/match-review/snap`, {
        fixed_stream: fixedStream,
        fixed_frame_index: segment[`${fixedStream}_${boundary}`].frame_index,
        target_frame_index: segment[`${targetStream}_${boundary}`].frame_index,
      })
      const segments = review.segments.map((item, index) => {
        if (index !== selected) return item
        const based = withAdjustmentBaseline(item)
        const lower = boundary === 'start' ? 0 : based[`${targetStream}_start`].frame_index + 1
        const upper = boundary === 'start' ? based[`${targetStream}_end`].frame_index - 1 : review.media[targetStream].frame_count - 1
        const next = Math.max(lower, Math.min(upper, response.data.frame.frame_index))
        return { ...based, [`${targetStream}_${boundary}`]: { frame_index: next, pts: next, time_seconds: next / review.media[targetStream].fps }, status: 'proposed', origin: 'manual' }
      })
      await persist(segments)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    } finally { setSaving(false) }
  }

  async function approve(mode) {
    setSaving(true)
    try {
      const response = await axios.post(`${API}/jobs/${job.id}/match-review/approve`, { revision: review.revision, mode })
      onQueued(response.data)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    } finally { setSaving(false) }
  }

  async function rebuildLegacyAlignment() {
    if (!window.confirm('Rebuild this alignment with the unified matcher? This discards all current segment edits.')) return
    setSaving(true)
    try {
      const response = await axios.post(`${API}/jobs/${job.id}/match-review/reanalyze`, {
        revision: review.revision, confirm_discard: true,
      })
      onQueued(response.data)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    } finally { setSaving(false) }
  }

  function seekComparison(progress) {
    if (!segment) return
    const next = Math.max(0, Math.min(1, Number(progress)))
    const times = {
      low: segment.low_start.time_seconds + durationBetween(segment.low_start, segment.low_end) * next,
      reference: segment.reference_start.time_seconds + durationBetween(segment.reference_start, segment.reference_end) * next,
    }
    if (lowVideo.current) lowVideo.current.currentTime = times.low
    if (referenceVideo.current) referenceVideo.current.currentTime = times.reference
    setPlaybackProgress(next)
    setPlayheads(times)
  }

  function showBoundary(nextBoundary) {
    setBoundary(nextBoundary)
    seekComparison(nextBoundary === 'start' ? 0 : 1)
  }

  function togglePlayback() {
    if (!segment) return
    const videos = [lowVideo.current, referenceVideo.current]
    if (playing) {
      videos.forEach((video) => video?.pause())
      setPlaying(false)
      return
    }
    const nextProgress = playbackProgress >= .999 ? 0 : playbackProgress
    seekComparison(nextProgress)
    const attempts = videos.map((video) => video?.play()).filter(Boolean)
    Promise.all(attempts).then(() => setPlaying(true)).catch(() => {
      videos.forEach((video) => video?.pause())
      setError('Playback could not start. Allow video playback in the browser and try again.')
    })
  }

  if (!review) return <p className="message">Loading frame-match workspace…</p>
  if (review.schema_version === 2) return <UnifiedMatchReview job={job} initialReview={review} onQueued={onQueued} />
  const unresolved = review.segments.filter((item) => item.status === 'proposed').length
  const confirmed = review.segments.filter((item) => item.status === 'confirmed').length
  const lowDuration = review.media.low.duration || 1
  const refDuration = review.media.reference.duration || 1
  const imageUrl = (stream, ref) => `${API}/jobs/${job.id}/frames/${stream}/${ref.frame_index}`
  const playbackFrame = (stream) => Math.round(playheads[stream] * review.media[stream].fps)

  return (
    <div className="match-review">
      <div className="review-intro">
        <p className="eyebrow">SHARED-FOOTAGE REVIEW</p>
        <h3>Teach the model which sections show the same footage.</h3>
        <p>Each match connects a range in one video to the same range in the other. Confirmed matches become high/low training examples. Unmatched and rejected footage is not deleted; it simply is not used as a paired example.</p>
        <button type="button" onClick={rebuildLegacyAlignment} disabled={saving}>Rebuild in unified timeline</button>
      </div>
      <div className="review-summary">
        <div><b>{confirmed}</b><span>confirmed</span></div>
        <div><b>{unresolved}</b><span>needs review</span></div>
        <div><b>{formatTime(review.summary?.matched_seconds || 0)}</b><span>matched</span></div>
      </div>
      <section className="review-stage match-navigation" aria-labelledby="choose-match-heading">
        <div className="stage-heading"><span>1</span><div><p className="eyebrow">CHOOSE A MATCH</p><h3 id="choose-match-heading">Select a shared section to review</h3></div></div>
        <div className="coverage-key"><span><i className="confirmed" />Confirmed match</span><span><i className="proposed" />Needs review</span><span><i className="unmatched" />Not in a match</span></div>
        <div className="timeline" aria-label="Shared footage coverage">
          <span>{streamLabel.low}</span><div>{review.segments.map((item, index) => <button title={`Open segment ${index + 1}: ${item.status}.`} aria-label={`Inspect segment ${index + 1} in supplemental video`} type="button" key={item.id} className={`range ${item.status}${selected === index ? ' selected' : ''}`} onClick={() => setSelected(index)} style={{ left: `${item.low_start.time_seconds / lowDuration * 100}%`, width: `${Math.max(.4, (item.low_end.time_seconds - item.low_start.time_seconds) / lowDuration * 100)}%` }} />)}</div>
          <span>{streamLabel.reference}</span><div>{review.segments.map((item, index) => <button title={`Open segment ${index + 1}: ${item.status}.`} aria-label={`Inspect segment ${index + 1} in reference video`} type="button" key={item.id} className={`range ${item.status}${selected === index ? ' selected' : ''}`} onClick={() => setSelected(index)} style={{ left: `${item.reference_start.time_seconds / refDuration * 100}%`, width: `${Math.max(.4, (item.reference_end.time_seconds - item.reference_start.time_seconds) / refDuration * 100)}%` }} />)}</div>
          <span /><div className="timeline-scale"><small>00:00</small><small>Uncolored areas are currently unmatched</small><small>End</small></div>
        </div>
        <div className="segment-navigator">
          <TooltipButton type="button" onClick={() => setSelected((current) => Math.max(0, current - 1))} disabled={selected === 0} tooltip="Open the previous proposed or confirmed match without changing this one.">← Previous</TooltipButton>
          <label>Current match<select aria-label="Review segment" value={selected} onChange={(event) => setSelected(Number(event.target.value))}>{review.segments.map((item, index) => <option key={item.id} value={index}>Segment {index + 1} · {item.status}</option>)}</select></label>
          <TooltipButton type="button" onClick={() => setSelected((current) => Math.min(review.segments.length - 1, current + 1))} disabled={selected >= review.segments.length - 1} tooltip="Open the next proposed or confirmed match without changing this one.">Next →</TooltipButton>
        </div>
        {segment && <div className="selected-match-summary"><b>Segment {selected + 1} · {segment.status}</b><span>{formatTime(durationBetween(segment.low_start, segment.low_end))} long</span><span>{Math.round((segment.confidence || 0) * 100)}% automatic confidence</span></div>}
      </section>
      {segment && <>
        <section className="review-stage playback-stage" aria-labelledby="compare-segment-heading">
          <div className="stage-heading-row">
            <div className="stage-heading"><span>2</span><div><p className="eyebrow">COMPARE THE SEGMENT</p><h3 id="compare-segment-heading">Check that both clips show the same motion</h3></div></div>
            <div className="view-toggle" aria-label="Comparison view">
              <TooltipButton type="button" className={comparisonMode === 'side-by-side' ? 'selected' : ''} aria-pressed={comparisonMode === 'side-by-side'} onClick={() => setComparisonMode('side-by-side')} tooltip="Show both videos next to each other while keeping their linked playheads.">Side by side</TooltipButton>
              <TooltipButton type="button" className={comparisonMode === 'overlay' ? 'selected' : ''} aria-pressed={comparisonMode === 'overlay'} onClick={() => setComparisonMode('overlay')} tooltip="Stack the playing reference over the supplemental video so motion differences are easier to spot.">Overlay</TooltipButton>
            </div>
          </div>
          <p className="stage-help">Playback starts from each clip’s matched boundary. Controls are linked, but the app does not silently correct drift while the clips play.</p>
          <div className={`comparison-visual ${comparisonMode}`}>
            <figure className="comparison-video comparison-low">
              <video ref={lowVideo} muted playsInline preload="metadata" src={`${ORIGIN}${review.proxy_urls.low}`} onLoadedMetadata={() => seekComparison(playbackProgress)} />
              <figcaption><b>{streamLabel.low}</b><span>frame {playbackFrame('low')} · {formatTime(playheads.low)}</span></figcaption>
            </figure>
            <figure className="comparison-video comparison-reference" style={comparisonMode === 'overlay' ? { opacity: overlayOpacity / 100 } : undefined}>
              <video ref={referenceVideo} muted playsInline preload="metadata" src={`${ORIGIN}${review.proxy_urls.reference}`} onLoadedMetadata={() => seekComparison(playbackProgress)} />
              <figcaption><b>{streamLabel.reference}</b><span>frame {playbackFrame('reference')} · {formatTime(playheads.reference)}</span></figcaption>
            </figure>
          </div>
          {comparisonMode === 'overlay' && <label className="overlay-control"><span>Reference opacity</span><input aria-label="Reference video opacity" type="range" min="0" max="100" value={overlayOpacity} onChange={(event) => setOverlayOpacity(Number(event.target.value))} /><output>{overlayOpacity}%</output></label>}
          <div className="playback-controls">
            <TooltipButton type="button" className="primary" onClick={togglePlayback} tooltip={playing ? 'Pause both videos at their current positions.' : 'Start both videos from the linked position shown on the segment scrubber.'}>{playing ? 'Pause both clips' : 'Play both clips'}</TooltipButton>
            <TooltipButton type="button" onClick={() => { ;[lowVideo.current, referenceVideo.current].forEach((video) => video?.pause()); setPlaying(false); seekComparison(0) }} tooltip="Pause both videos and return each one to this segment’s start boundary.">Restart segment</TooltipButton>
            <label className="segment-scrubber"><span>Matched position</span><input aria-label="Matched segment position" type="range" min="0" max="1000" value={Math.round(playbackProgress * 1000)} onChange={(event) => seekComparison(Number(event.target.value) / 1000)} /><output>{Math.round(playbackProgress * 100)}%</output></label>
          </div>
        </section>
        <section className="review-stage boundary-stage" aria-labelledby="match-frames-heading">
          <div className="stage-heading"><span>3</span><div><p className="eyebrow">MATCH EXACT FRAMES</p><h3 id="match-frames-heading">Align the first and last moments</h3></div></div>
          <p className="stage-help">Choose a boundary, then scrub either clip until both images show the same instant. The sliders move together by elapsed time while respecting each clip’s frame rate, and extend up to one second beyond the segment’s outer edge. Adjustments are saved when you release a slider.</p>
          <div className="boundary-tabs">
            <TooltipButton type="button" className={boundary === 'start' ? 'selected' : ''} aria-pressed={boundary === 'start'} onClick={() => showBoundary('start')} tooltip="Compare and adjust the first paired frame used by this match.">Start frames</TooltipButton>
            <TooltipButton type="button" className={boundary === 'end' ? 'selected' : ''} aria-pressed={boundary === 'end'} onClick={() => showBoundary('end')} tooltip="Compare and adjust the last paired frame used by this match.">End frames</TooltipButton>
          </div>
          <div className="frame-compare">
          {['low', 'reference'].map((stream) => {
            const key = `${stream}_${boundary}`
            const frameIndex = boundaryDrafts[stream] ?? segment[key].frame_index
            const ref = { frame_index: frameIndex, time_seconds: frameIndex / review.media[stream].fps }
            const shift = frameShift(segment, key, frameIndex)
            const range = sliderRange(stream)
            return <div className={`frame-pane pane-${stream}`} key={stream}>
              <div className="frame-pane-heading">
                <span><b>{streamLabel[stream]}</b> · frame {ref.frame_index} · {formatTime(ref.time_seconds)}</span>
                <strong className={`frame-shift ${shift.className}`} aria-live="polite">{shift.label}</strong>
              </div>
              <img src={imageUrl(stream, ref)} alt={`${stream} ${boundary} frame`} />
              <label className="boundary-slider">
                <span><b>Linked time</b> · Scrub {stream === 'low' ? 'supplemental' : 'reference'} {boundary} frame</span>
                <input
                  type="range"
                  min={range.min}
                  max={range.max}
                  value={frameIndex}
                  disabled={saving}
                  aria-label={`Scrub ${stream === 'low' ? 'supplemental' : 'reference'} ${boundary} boundary frame`}
                  onChange={(event) => updateBoundarySliders(stream, Number(event.target.value))}
                  onPointerUp={commitBoundarySliders}
                  onKeyUp={commitBoundarySliders}
                  onBlur={commitBoundarySliders}
                />
                <small><span>Frame {range.min}</span><b>Frame {frameIndex}</b><span>Frame {range.max}</span></small>
              </label>
              <div className="step-controls">{[-10, -1, 1, 10].map((amount) => <TooltipButton key={amount} type="button" disabled={saving} onClick={() => step(stream, amount)} tooltip={`Move the ${stream === 'low' ? 'supplemental' : 'reference'} ${boundary} boundary ${Math.abs(amount)} frame${Math.abs(amount) === 1 ? '' : 's'} ${amount < 0 ? 'earlier' : 'later'}.`}>{amount < 0 ? '←' : '→'} {Math.abs(amount)} frame{Math.abs(amount) === 1 ? '' : 's'}</TooltipButton>)}</div>
              <TooltipButton wrapperClassName="snap-tooltip" type="button" className="snap" disabled={saving} onClick={() => snap(stream)} tooltip={`Keep the other frame fixed and search nearby ${stream === 'low' ? 'supplemental' : 'reference'} frames for the closest visual match.`}>Find closest {stream === 'low' ? 'supplemental' : 'reference'} frame</TooltipButton>
            </div>
          })}
          </div>
          <p className="keyboard-help">Keyboard: ←/→ steps the supplemental frame; ↓/↑ steps the reference frame. Hold Shift to move 10 frames.</p>
          <div className="review-actions">
            <TooltipButton type="button" onClick={() => resolveSelected('rejected')} disabled={saving} tooltip="Exclude this proposed relationship from paired training. Neither source clip is deleted.">Reject this match</TooltipButton>
            <TooltipButton type="button" className="primary" onClick={() => resolveSelected('confirmed')} disabled={saving} tooltip="Save these start and end frame pairs as approved training footage, then open the next unresolved match.">Confirm frame match</TooltipButton>
          </div>
        </section>
      </>}
      <details className="advanced-editing">
        <summary>Advanced segment editing</summary>
        <p>Browse both full sources independently to add missing shared footage. Splitting uses the linked position in the selected segment above.</p>
        <div className="source-videos">
          <label>{streamLabel.low}<video ref={sourceLowVideo} controls muted playsInline preload="metadata" src={`${ORIGIN}${review.proxy_urls.low}`} /></label>
          <label>{streamLabel.reference}<video ref={sourceReferenceVideo} controls muted playsInline preload="metadata" src={`${ORIGIN}${review.proxy_urls.reference}`} /></label>
        </div>
        <div className="advanced-actions">
          <TooltipButton type="button" onClick={addFromPlayheads} disabled={saving} tooltip="Create a new 10-second proposed match centered on the two independently positioned source playheads.">Add match from source playheads</TooltipButton>
          <TooltipButton type="button" onClick={splitAtPlayheads} disabled={!segment || saving} tooltip="Divide the selected match at the linked playback position; both resulting matches will need confirmation.">Split selected match here</TooltipButton>
          <TooltipButton type="button" onClick={mergeNext} disabled={!segment || selected >= review.segments.length - 1 || saving} tooltip="Join this match to the next match, using this start boundary and the next match’s end boundary.">Merge with next match</TooltipButton>
        </div>
      </details>
      {error && <p className="alert error" role="alert">{error}</p>}
      <div className="approval-actions">
        <p>Resolve every proposal, then choose whether confirmed matches should supervise training.</p>
        <TooltipButton type="button" onClick={() => approve('unpaired')} disabled={saving || unresolved > 0} tooltip="Start adaptation without using any confirmed high/low frame pairs. Every proposal must be resolved first.">Train without confirmed pairs</TooltipButton>
        <TooltipButton type="button" className="primary" onClick={() => approve('paired')} disabled={saving || unresolved > 0 || confirmed === 0} tooltip="Use all confirmed matches as paired supervision and queue this job for processing.">Use confirmed pairs and start processing</TooltipButton>
      </div>
    </div>
  )
}

function FileField({ label, description, file, onChange }) {
  return (
    <label className="file-field">
      <span className="file-title">{label}</span>
      <span className="file-help">{description}</span>
      <input type="file" accept="video/mp4,video/quicktime,.m4v" onChange={(event) => onChange(event.target.files?.[0] || null)} />
      <span className={file ? 'file-name selected' : 'file-name'}>{file ? file.name : 'Choose MP4 or MOV'}</span>
    </label>
  )
}

function SystemStatus({ status, online, onRecheck }) {
  const worker = status?.worker
  const gpu = status?.gpu
  const queue = status?.queue
  const workerText = worker?.state === 'busy'
    ? `Processing job ${worker.current_job_id.slice(0, 8)}`
    : worker?.state === 'idle' ? 'Idle and ready' : 'Stopped'
  const gpuText = gpu?.state === 'available'
    ? gpu.name
    : gpu?.state === 'unavailable' ? 'Unavailable' : 'Checking availability'

  return (
    <section className="panel system-panel" aria-label="System status">
      <div className="system-heading">
        <div><p className="eyebrow">SYSTEM STATUS</p><h2>Processing capacity</h2></div>
        <button onClick={onRecheck} disabled={online !== true || gpu?.state === 'checking'} type="button">Recheck GPU</button>
      </div>
      <div className="status-grid">
        <div><span>Backend</span><b className={online === false ? 'status-bad' : online === true ? 'status-good' : ''}>{online === false ? 'Unreachable' : online === true ? 'Online' : 'Connecting'}</b></div>
        <div><span>Worker</span><b className={worker?.state === 'stopped' ? 'status-bad' : ''}>{online === false ? 'Unknown' : workerText}</b></div>
        <div><span>GPU</span><b className={gpu?.state === 'available' ? 'status-good' : gpu?.state === 'unavailable' ? 'status-bad' : ''}>{online === false ? 'Unknown' : gpuText}</b></div>
        <div><span>Outstanding</span><b>{online === false ? 'Unknown' : `${queue?.outstanding ?? 0} (${queue?.queued ?? 0} queued · ${queue?.needs_review ?? 0} review)`}</b></div>
      </div>
      {online === false && <p className="alert error">The backend is unreachable. Existing jobs remain visible, but no worker can claim queued work until the backend is running.</p>}
      {worker?.fatal_error && <p className="alert error">Worker stopped: {worker.fatal_error}</p>}
      {gpu?.state === 'available' && <p className="status-detail">{gpu.device_count} device{gpu.device_count === 1 ? '' : 's'} · {gpu.device} · HIP {gpu.hip_version} · PyTorch {gpu.torch_version}</p>}
      {gpu?.state === 'unavailable' && <p className="alert error">GPU check failed: {gpu.error}</p>}
    </section>
  )
}

function JobList({ jobs, selectedId, onSelect, onCancelAll }) {
  const activeCount = jobs.filter((job) => !terminal.has(job.state)).length

  return (
    <section className="panel jobs-panel" aria-label="Jobs">
      <div className="jobs-heading">
        <div><p className="eyebrow">DURABLE QUEUE</p><h2>Jobs</h2></div>
        {activeCount > 0 && <button className="danger" onClick={onCancelAll}>Cancel all active jobs</button>}
      </div>
      {jobs.length === 0 ? (
        <p className="empty-jobs">No jobs yet. Submitted work will remain available here after a restart.</p>
      ) : (
        <div className="job-list">
          {jobs.map((job) => {
            const progress = Math.round((job.progress || 0) * 100)
            return (
              <button
                className={`job-row${job.id === selectedId ? ' selected' : ''}`}
                key={job.id}
                onClick={() => onSelect(job.id)}
                type="button"
              >
                <span><b>JOB {job.id.slice(0, 8)}</b><small>{job.preset} · {matchingLabel(job.matching_mode)}</small></span>
                <span><b className={`state state-${job.state}`}>{job.stage.replace('_', ' ')}</b><small>{new Date(job.created_at).toLocaleString()}</small></span>
                <strong>{progress}%</strong>
              </button>
            )
          })}
        </div>
      )}
    </section>
  )
}

function App() {
  const [low, setLow] = useState(null)
  const [reference, setReference] = useState(null)
  const [preset, setPreset] = useState('balanced')
  const [matchingMode, setMatchingMode] = useState('guided')
  const [useAudioMatching, setUseAudioMatching] = useState(false)
  const [jobs, setJobs] = useState([])
  const [jobsLoaded, setJobsLoaded] = useState(false)
  const [selectedId, setSelectedId] = useState(() => window.localStorage.getItem(selectedJobKey))
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState('')
  const [jobsError, setJobsError] = useState('')
  const [system, setSystem] = useState(null)
  const [backendOnline, setBackendOnline] = useState(null)
  const [cancelAllOpen, setCancelAllOpen] = useState(false)
  const [cancellingAll, setCancellingAll] = useState(false)
  const [cancelAllError, setCancelAllError] = useState('')

  const refreshJobs = useCallback(async () => {
    try {
      const response = await axios.get(`${API}/jobs`)
      setJobs(response.data.jobs)
      setJobsLoaded(true)
      setJobsError('')
    } catch (requestError) {
      setJobsError(requestError.response?.data?.detail || requestError.message)
    }
  }, [])

  const refreshSystem = useCallback(async () => {
    try {
      const response = await axios.get(`${API}/system/status`)
      setSystem(response.data)
      setBackendOnline(true)
    } catch {
      setBackendOnline(false)
    }
  }, [])

  useEffect(() => {
    const initial = window.setTimeout(refreshJobs, 0)
    const initialSystem = window.setTimeout(refreshSystem, 0)
    const timer = window.setInterval(refreshJobs, 1500)
    const systemTimer = window.setInterval(refreshSystem, 1500)
    return () => {
      window.clearTimeout(initial)
      window.clearTimeout(initialSystem)
      window.clearInterval(timer)
      window.clearInterval(systemTimer)
    }
  }, [refreshJobs, refreshSystem])

  const effectiveSelectedId = useMemo(() => {
    if (selectedId && jobs.some((job) => job.id === selectedId)) return selectedId
    return (jobs.find((job) => !terminal.has(job.state)) || jobs[0])?.id || null
  }, [jobs, selectedId])

  useEffect(() => {
    if (!jobsLoaded) return
    if (effectiveSelectedId) window.localStorage.setItem(selectedJobKey, effectiveSelectedId)
    else window.localStorage.removeItem(selectedJobKey)
  }, [effectiveSelectedId, jobsLoaded])

  const job = useMemo(() => jobs.find((item) => item.id === effectiveSelectedId) || null, [effectiveSelectedId, jobs])
  const busy = jobs.some((item) => !terminal.has(item.state) && item.state !== 'awaiting_match_review')
  const activeCount = jobs.filter((item) => !terminal.has(item.state)).length
  const progress = Math.round((job?.progress || 0) * 100)
  const eta = job?.eta_seconds ? `${Math.max(1, Math.round(job.eta_seconds / 60))} min remaining` : null

  async function submit(event) {
    event.preventDefault()
    if (!low || !reference) {
      setError('Select both the low-resolution supplement and the high-resolution reference.')
      return
    }
    setError('')
    setUploadProgress(0)
    const data = new FormData()
    data.append('low_video', low)
    data.append('reference_video', reference)
    data.append('preset', preset)
    data.append('matching_mode', matchingMode)
    data.append('use_audio_matching', String(useAudioMatching))
    try {
      const response = await axios.post(`${API}/jobs`, data, {
        onUploadProgress: ({ loaded, total }) => setUploadProgress(total ? Math.round(loaded * 100 / total) : 0),
      })
      setJobs((current) => [response.data, ...current.filter((item) => item.id !== response.data.id)])
      setSelectedId(response.data.id)
      setUploadProgress(0)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    }
  }

  async function cancel() {
    try {
      const response = await axios.post(`${API}/jobs/${job.id}/cancel`)
      setJobs((current) => current.map((item) => item.id === response.data.id ? response.data : item))
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    }
  }

  async function cancelAll() {
    setCancellingAll(true)
    setCancelAllError('')
    try {
      const response = await axios.post(`${API}/jobs/cancel-all`)
      setJobs(response.data.jobs)
      setCancelAllOpen(false)
    } catch (requestError) {
      setCancelAllError(requestError.response?.data?.detail || requestError.message)
    } finally {
      setCancellingAll(false)
    }
  }

  function requestCancelAll() {
    setCancelAllError('')
    setCancelAllOpen(true)
  }

  async function recheckGpu() {
    try {
      const response = await axios.post(`${API}/system/gpu/recheck`)
      setSystem((current) => current ? { ...current, gpu: response.data } : current)
      setBackendOnline(true)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
      setBackendOnline(false)
    }
  }

  async function remove() {
    try {
      await axios.delete(`${API}/jobs/${job.id}`)
      setJobs((current) => current.filter((item) => item.id !== job.id))
      setUploadProgress(0)
    } catch (requestError) {
      setError(requestError.response?.data?.detail || requestError.message)
    }
  }

  function reviewQueued(updated) {
    setJobs((current) => current.map((item) => item.id === updated.id ? updated : item))
  }

  return (
    <main>
      <header className="hero">
        <p className="eyebrow">LOCAL GPU RESTORATION</p>
        <h1>Two partial sources.<br /><span>One restoration model.</span></h1>
        <p className="lede">Train a video-specific upscaler from two overlapping versions of the same video: a high-resolution reference and a low-resolution supplement. Either video may contain footage the other does not.</p>
      </header>

      <SystemStatus status={system} online={backendOnline} onRecheck={recheckGpu} />

      <section className="panel">
        <form onSubmit={submit}>
          <div className="workflow-explainer" aria-label="How the workflow works">
            <div><b>1 · Find overlap</b><span>Locate sections that appear in both videos, even when their edits or frame rates differ.</span></div>
            <div><b>2 · Learn detail</b><span>Use confirmed high/low frame pairs plus the full high-resolution reference to adapt the upscaler.</span></div>
            <div><b>3 · Restore footage</b><span>Upscale low-resolution sections using detail learned from the high-resolution version.</span></div>
          </div>
          <p className="input-note"><b>Neither upload has to be complete.</b> “Reference” describes the source of visual detail; “supplemental” describes the footage that needs upscaling.</p>
          <p className="coverage-caveat"><b>Output coverage in this build:</b> the rendered result still follows the supplemental video's timeline. High-resolution-only ranges are exposed during review, but are not automatically inserted into the result because their ordering is ambiguous without an edit plan.</p>
          <div className="files">
            <FileField label="01 · Low-resolution supplement" description="Lower-quality footage that fills gaps or extends the high-resolution version." file={low} onChange={setLow} />
            <FileField label="02 · High-resolution reference" description="Higher-quality footage used as the detail target. It may also contain unique sections." file={reference} onChange={setReference} />
          </div>
          <fieldset className="mode-options">
            <legend>How should the reference be used?</legend>
            <label className={matchingMode === 'guided' ? 'selected' : ''}>
              <input type="radio" name="matching-mode" value="guided" checked={matchingMode === 'guided'} onChange={(event) => setMatchingMode(event.target.value)} disabled={busy} />
              <span><b>Find and match shared frames</b><small>Recommended · Review exact matching sections before training for the strongest supervision.</small></span>
            </label>
            <label className={matchingMode === 'reference_only' ? 'selected' : ''}>
              <input type="radio" name="matching-mode" value="reference_only" checked={matchingMode === 'reference_only'} onChange={(event) => setMatchingMode(event.target.value)} disabled={busy} />
              <span><b>Skip matching · reference only</b><small>Start sooner using synthetic low-quality pairs when the videos do not share trustworthy frames.</small></span>
            </label>
          </fieldset>
          <TooltipCheckbox
            checked={useAudioMatching}
            onChange={(event) => setUseAudioMatching(event.target.checked)}
            disabled={busy || matchingMode !== 'guided'}
            tooltip="Repetitive audio, including video-game ambience or music, can cause false-positive or false-negative matches. Leave this off unless both clips share distinctive audio."
          >
            Use audio when matching
          </TooltipCheckbox>
          <div className="controls">
            <label>
              <span>Training preset</span>
              <select value={preset} onChange={(event) => setPreset(event.target.value)} disabled={busy}>
                <option value="quick">Quick · up to 15 min</option>
                <option value="balanced">Balanced · up to 1 hour</option>
                <option value="quality">Quality · up to 4 hours</option>
              </select>
            </label>
            <button className="primary" disabled={busy} type="submit">{busy ? 'Work already queued or processing' : matchingMode === 'guided' ? 'Analyze frames' : 'Start reference-only processing'}</button>
          </div>
        </form>
        {uploadProgress > 0 && <p className="upload">Uploading · {uploadProgress}%</p>}
        {jobsError && <p className="alert error" role="alert">Job list refresh failed: {jobsError}</p>}
        {error && <p className="alert error" role="alert">{error}</p>}
      </section>

      {job && (
        <section className="panel job" aria-live="polite">
          <div className="job-head">
            <div><p className="eyebrow">JOB {job.id.slice(0, 8)}</p><h2>{job.stage.replace('_', ' ')}</h2></div>
            <strong>{progress}%</strong>
          </div>
          <div className="progress"><span style={{ width: `${progress}%` }} /></div>
          <p className="message">{job.message}{eta ? ` · ${eta}` : ''}</p>
          <p className="mode">Workflow · <b>{matchingLabel(job.matching_mode)}</b></p>
          {job.alignment_mode && <p className="mode">Alignment mode · <b>{job.alignment_mode}</b></p>}
          {job.warning && <p className="alert warning">{job.warning}</p>}
          {job.error && <p className="alert error">{job.error}</p>}
          {job.metrics?.psnr && <div className="metrics"><span>Validation PSNR</span><b>{job.metrics.psnr.toFixed(2)} dB</b></div>}
          {job.state === 'awaiting_match_review' && <MatchReview job={job} onQueued={reviewQueued} />}
          {job.state === 'completed' && (
            <div className="result">
              <video controls src={`${API.replace('/api/v1', '')}${job.result_url}`} />
              <div className="actions">
                <a className="primary link" href={`${API.replace('/api/v1', '')}${job.result_url}`} download>Download restored video</a>
                <a href={`${API.replace('/api/v1', '')}${job.report_url}`} download>Download report</a>
              </div>
            </div>
          )}
          <div className="job-actions">
            {!terminal.has(job.state) && <button onClick={cancel}>Cancel job</button>}
            {terminal.has(job.state) && <button className="danger" onClick={remove}>Delete job and files</button>}
          </div>
        </section>
      )}

      <JobList jobs={jobs} selectedId={effectiveSelectedId} onSelect={setSelectedId} onCancelAll={requestCancelAll} />
      {cancelAllOpen && <CancelJobsDialog activeCount={activeCount} busy={cancellingAll} error={cancelAllError} onClose={() => setCancelAllOpen(false)} onConfirm={cancelAll} />}
    </main>
  )
}

export default App
