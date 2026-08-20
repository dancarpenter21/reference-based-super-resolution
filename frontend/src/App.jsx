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

function frameShift(segment, key) {
  const baseline = segment.adjustment_baseline?.[key] ?? segment[key].frame_index
  const delta = segment[key].frame_index - baseline
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
  const lowVideo = useRef(null)
  const referenceVideo = useRef(null)
  const sourceLowVideo = useRef(null)
  const sourceReferenceVideo = useRef(null)

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
    persist(segments)
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
      const next = Math.max(0, Math.min(info.frame_count - 1, based[key].frame_index + amount))
      return { ...based, [key]: { frame_index: next, pts: next, time_seconds: next / info.fps }, status: 'proposed', origin: 'manual' }
    })
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
        return { ...based, [`${targetStream}_${boundary}`]: response.data.frame, status: 'proposed', origin: 'manual' }
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
          <p className="stage-help">Choose a boundary, then move either clip until both images show the same instant. Every adjustment is saved to this segment.</p>
          <div className="boundary-tabs">
            <TooltipButton type="button" className={boundary === 'start' ? 'selected' : ''} aria-pressed={boundary === 'start'} onClick={() => showBoundary('start')} tooltip="Compare and adjust the first paired frame used by this match.">Start frames</TooltipButton>
            <TooltipButton type="button" className={boundary === 'end' ? 'selected' : ''} aria-pressed={boundary === 'end'} onClick={() => showBoundary('end')} tooltip="Compare and adjust the last paired frame used by this match.">End frames</TooltipButton>
          </div>
          <div className="frame-compare">
          {['low', 'reference'].map((stream) => {
            const key = `${stream}_${boundary}`
            const ref = segment[key]
            const shift = frameShift(segment, key)
            return <div className={`frame-pane pane-${stream}`} key={stream}>
              <div className="frame-pane-heading">
                <span><b>{streamLabel[stream]}</b> · frame {ref.frame_index} · {formatTime(ref.time_seconds)}</span>
                <strong className={`frame-shift ${shift.className}`} aria-live="polite">{shift.label}</strong>
              </div>
              <img src={imageUrl(stream, ref)} alt={`${stream} ${boundary} frame`} />
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

      <JobList jobs={jobs} selectedId={effectiveSelectedId} onSelect={setSelectedId} onCancelAll={requestCancelAll} />

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
      {cancelAllOpen && <CancelJobsDialog activeCount={activeCount} busy={cancellingAll} error={cancelAllError} onClose={() => setCancelAllOpen(false)} onConfirm={cancelAll} />}
    </main>
  )
}

export default App
