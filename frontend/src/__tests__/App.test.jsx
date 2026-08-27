import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import axios from 'axios'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import App from '../App'

vi.mock('axios')

const activeJob = {
  id: 'active-job-1234', state: 'training', stage: 'training', progress: 0.4,
  message: 'step 40', preset: 'balanced', metrics: null, warning: null, error: null,
  created_at: '2026-08-13T12:00:00+00:00', updated_at: '2026-08-13T12:01:00+00:00',
  eta_seconds: 120,
}

const completedJob = {
  id: 'done-job-5678', state: 'completed', stage: 'completed', progress: 1,
  message: 'Upscaled video is ready', preset: 'quick', metrics: null, warning: null, error: null,
  created_at: '2026-08-12T12:00:00+00:00', updated_at: '2026-08-12T13:00:00+00:00',
  result_url: '/api/v1/jobs/done-job-5678/result', report_url: '/api/v1/jobs/done-job-5678/report',
}

const systemStatus = {
  backend: { state: 'online' },
  worker: { state: 'idle', current_job_id: null, started_at: '2026-08-13T12:00:00Z', fatal_error: null },
  gpu: {
    state: 'available', device: 'cuda:0', device_count: 1, name: 'AMD Radeon RX 9070 XT',
    hip_version: '7.2', torch_version: '2.9.1', error: null, last_checked_at: '2026-08-13T12:00:01Z',
  },
  queue: { queued: 0, processing: 0, outstanding: 0 },
}

function mockGets(jobs = [], status = systemStatus) {
  axios.get.mockImplementation((url) => Promise.resolve({ data: url.endsWith('/system/status') ? status : { jobs } }))
}

describe('App', () => {
  beforeEach(() => {
    window.localStorage.clear()
    vi.restoreAllMocks()
    mockGets()
  })

  it('renders the two-video workflow and an empty durable queue', async () => {
    render(<App />)
    expect(screen.getByText('01 · Low-resolution supplement')).toBeInTheDocument()
    expect(screen.getByText('02 · High-resolution reference')).toBeInTheDocument()
    expect(screen.getByText(/Neither upload has to be complete/)).toBeInTheDocument()
    expect(screen.getByText(/rendered result still follows the supplemental video's timeline/i)).toBeInTheDocument()
    expect(screen.getByRole('radio', { name: /find and match shared frames/i })).toBeChecked()
    expect(screen.getByRole('checkbox', { name: /use audio when matching/i })).not.toBeChecked()
    expect(screen.getByRole('tooltip', { name: /video-game ambience or music/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /analyze frames/i })).toBeInTheDocument()
    expect(await screen.findByText(/No jobs yet/)).toBeInTheDocument()
  })

  it('restores jobs, selects the active job, and disables submission', async () => {
    mockGets([activeJob, completedJob], {
      ...systemStatus,
      worker: { ...systemStatus.worker, state: 'busy', current_job_id: activeJob.id },
      queue: { queued: 0, processing: 1, outstanding: 1 },
    })

    render(<App />)

    expect((await screen.findAllByText('JOB active-j')).length).toBeGreaterThan(0)
    expect(screen.getByText('JOB done-job')).toBeInTheDocument()
    const currentPanel = screen.getByText('step 40 · 2 min remaining').closest('section')
    const jobsPanel = screen.getByRole('region', { name: 'Jobs' })
    expect(currentPanel.compareDocumentPosition(jobsPanel) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(screen.getByRole('button', { name: /work already queued or processing/i })).toBeDisabled()
    expect(screen.getByText('Processing job active-j')).toBeInTheDocument()
    expect(screen.getByText('AMD Radeon RX 9070 XT')).toBeInTheDocument()
    expect(window.localStorage.getItem('refsr-selected-job')).toBe(activeJob.id)
  })

  it('submits reference-only mode without frame analysis', async () => {
    const queued = {
      ...activeJob, id: 'reference-only', state: 'queued', stage: 'queued', progress: 0,
      message: 'Queued for reference-only processing', matching_mode: 'reference_only',
    }
    axios.post.mockResolvedValue({ data: queued })
    const { container } = render(<App />)
    const [lowInput, referenceInput] = container.querySelectorAll('input[type="file"]')
    fireEvent.change(lowInput, { target: { files: [new File(['low'], 'low.mp4', { type: 'video/mp4' })] } })
    fireEvent.change(referenceInput, { target: { files: [new File(['reference'], 'reference.mp4', { type: 'video/mp4' })] } })
    fireEvent.click(screen.getByRole('radio', { name: /skip matching/i }))

    expect(screen.getByRole('button', { name: /start reference-only processing/i })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /start reference-only processing/i }))

    await waitFor(() => expect(axios.post).toHaveBeenCalled())
    const [, data] = axios.post.mock.calls[0]
    expect(data.get('matching_mode')).toBe('reference_only')
    expect(data.get('use_audio_matching')).toBe('false')
    expect(data.get('preset')).toBe('balanced')
  })

  it('submits the opt-in audio matching preference', async () => {
    axios.post.mockResolvedValue({ data: { ...activeJob, id: 'audio-job', state: 'queued' } })
    const { container } = render(<App />)
    const [lowInput, referenceInput] = container.querySelectorAll('input[type="file"]')
    fireEvent.change(lowInput, { target: { files: [new File(['low'], 'low.mp4', { type: 'video/mp4' })] } })
    fireEvent.change(referenceInput, { target: { files: [new File(['reference'], 'reference.mp4', { type: 'video/mp4' })] } })
    fireEvent.click(screen.getByRole('checkbox', { name: /use audio when matching/i }))
    fireEvent.click(screen.getByRole('button', { name: /analyze frames/i }))

    await waitFor(() => expect(axios.post).toHaveBeenCalled())
    const [, data] = axios.post.mock.calls[0]
    expect(data.get('use_audio_matching')).toBe('true')
  })

  it('restores the selected historical job from local storage', async () => {
    window.localStorage.setItem('refsr-selected-job', completedJob.id)
    mockGets([activeJob, completedJob])

    render(<App />)

    expect(await screen.findByText('Upscaled video is ready')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /delete job and files/i })).toBeInTheDocument()
  })

  it('bulk cancels active jobs after confirmation', async () => {
    const cancelled = { ...activeJob, state: 'cancelled', stage: 'cancelled', message: 'Job cancelled' }
    mockGets([activeJob])
    axios.post.mockResolvedValue({ data: { affected: 1, jobs: [cancelled] } })
    render(<App />)

    fireEvent.click(await screen.findByRole('button', { name: /cancel all active jobs/i }))
    const dialog = screen.getByRole('dialog', { name: /stop all active work/i })
    expect(dialog).toBeInTheDocument()
    expect(screen.getByText(/completed jobs and their files will remain available/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /keep jobs running/i })).toHaveFocus()
    expect(axios.post).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: /cancel 1 active job/i }))

    await waitFor(() => expect(axios.post).toHaveBeenCalledWith(expect.stringMatching(/jobs\/cancel-all$/)))
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
    expect(await screen.findByText(/Job cancelled/)).toBeInTheDocument()
  })

  it('keeps the last loaded list when polling fails', async () => {
    let jobRequests = 0
    axios.get.mockImplementation((url) => {
      if (url.endsWith('/system/status')) return Promise.resolve({ data: systemStatus })
      jobRequests += 1
      return jobRequests === 1 ? Promise.resolve({ data: { jobs: [activeJob] } }) : Promise.reject(new Error('offline'))
    })
    render(<App />)
    expect((await screen.findAllByText('JOB active-j')).length).toBeGreaterThan(0)

    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('offline'), { timeout: 2500 })
    expect(screen.getAllByText('JOB active-j').length).toBeGreaterThan(0)
  })

  it('explains when the backend is unreachable while retaining jobs', async () => {
    axios.get.mockImplementation((url) => url.endsWith('/system/status')
      ? Promise.reject(new Error('connection refused'))
      : Promise.resolve({ data: { jobs: [activeJob] } }))

    render(<App />)

    expect(await screen.findByText('Unreachable')).toBeInTheDocument()
    expect(screen.getByText(/no worker can claim queued work/i)).toBeInTheDocument()
    expect(screen.getAllByText('JOB active-j').length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: /recheck gpu/i })).toBeDisabled()
  })

  it('shows a GPU failure and starts a manual recheck', async () => {
    const unavailable = {
      ...systemStatus,
      gpu: { ...systemStatus.gpu, state: 'unavailable', name: null, error: 'RuntimeError: driver missing' },
    }
    mockGets([], unavailable)
    axios.post.mockResolvedValue({ data: { ...unavailable.gpu, state: 'checking', error: null } })
    render(<App />)

    expect(await screen.findByText(/GPU check failed: RuntimeError: driver missing/)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /recheck gpu/i }))

    await waitFor(() => expect(axios.post).toHaveBeenCalledWith(expect.stringMatching(/system\/gpu\/recheck$/)))
    expect(screen.getByText('Checking availability')).toBeInTheDocument()
  })

  it('requires boundary review before processing a proposed segment', async () => {
    const reviewJob = {
      ...activeJob, state: 'awaiting_match_review', stage: 'awaiting_match_review', progress: .07,
      message: 'Review proposed frame matches', needs_review: true,
    }
    const segment = {
      id: 'segment-1', confidence: .9, origin: 'automatic', status: 'proposed',
      low_start: { frame_index: 10, pts: 10, time_seconds: 1 },
      low_end: { frame_index: 50, pts: 50, time_seconds: 5 },
      reference_start: { frame_index: 8, pts: 8, time_seconds: 1 },
      reference_end: { frame_index: 40, pts: 40, time_seconds: 5 },
    }
    const review = {
      revision: 1, segments: [segment], summary: { proposed_segments: 1, matched_seconds: 4 },
      media: {
        low: { fps: 10, frame_count: 100, duration: 10 },
        reference: { fps: 8, frame_count: 80, duration: 10 },
      },
      proxy_urls: { low: '/api/v1/jobs/review/navigation/low', reference: '/api/v1/jobs/review/navigation/reference' },
    }
    axios.get.mockImplementation((url) => {
      if (url.endsWith('/system/status')) return Promise.resolve({ data: systemStatus })
      if (url.endsWith('/match-review')) return Promise.resolve({ data: review })
      return Promise.resolve({ data: { jobs: [reviewJob] } })
    })
    axios.put.mockImplementation((_, payload) => Promise.resolve({ data: {
      ...review,
      revision: payload.revision + 1,
      segments: payload.segments,
      summary: {
        confirmed_segments: payload.segments.filter((item) => item.status === 'confirmed').length,
        proposed_segments: payload.segments.filter((item) => item.status === 'proposed').length,
        matched_seconds: 4,
      },
    } }))

    const playSpy = vi.spyOn(window.HTMLMediaElement.prototype, 'play').mockResolvedValue()
    const { container } = render(<App />)

    expect(await screen.findByText('Start frames')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /use confirmed pairs/i })).toBeDisabled()
    expect(screen.queryByRole('region', { name: /segment 1 contents/i })).not.toBeInTheDocument()
    expect(screen.getAllByText(/no shift from proposal/i)).toHaveLength(2)
    expect(screen.getByRole('button', { name: /side by side/i })).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(screen.getByRole('button', { name: /play both clips/i }))
    await waitFor(() => expect(playSpy).toHaveBeenCalledTimes(2))
    const comparisonVideos = container.querySelectorAll('.comparison-video video')
    fireEvent.change(screen.getByRole('slider', { name: /matched segment position/i }), { target: { value: 500 } })
    expect(comparisonVideos[0].currentTime).toBe(3)
    expect(comparisonVideos[1].currentTime).toBe(3)
    fireEvent.click(screen.getByRole('button', { name: /pause both clips/i }))
    fireEvent.click(screen.getByRole('button', { name: /^overlay$/i }))
    expect(screen.getByRole('button', { name: /^overlay$/i })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('slider', { name: /reference video opacity/i })).toHaveValue('50')
    expect(screen.getByRole('tooltip', { name: /stack the playing reference/i })).toBeInTheDocument()
    expect(screen.getByText(/advanced segment editing/i).closest('details')).not.toHaveAttribute('open')
    expect(screen.getByText(/Unmatched and rejected footage is not deleted/i)).toBeInTheDocument()
    const boundarySlider = screen.getByRole('slider', { name: /scrub supplemental start boundary frame/i })
    expect(boundarySlider).toHaveAttribute('min', '0')
    expect(boundarySlider).toHaveAttribute('max', '49')
    expect(boundarySlider).toHaveValue('10')
    fireEvent.change(boundarySlider, { target: { value: 7 } })
    expect(screen.getByAltText(/low start frame/i).getAttribute('src')).toMatch(/\/frames\/low\/7$/)
    expect(screen.getByAltText(/reference start frame/i).getAttribute('src')).toMatch(/\/frames\/reference\/6$/)
    expect(screen.getByText('−3 frames earlier')).toBeInTheDocument()
    expect(screen.getByText('−2 frames earlier')).toBeInTheDocument()
    expect(axios.put).not.toHaveBeenCalled()
    fireEvent.pointerUp(boundarySlider)
    await waitFor(() => expect(axios.put).toHaveBeenCalledWith(
      expect.stringMatching(/match-review$/),
      expect.objectContaining({
        revision: 1,
        segments: [expect.objectContaining({
          adjustment_baseline: expect.objectContaining({ low_start: 10, reference_start: 8 }),
          low_start: expect.objectContaining({ frame_index: 7 }),
          reference_start: expect.objectContaining({ frame_index: 6 }),
        })],
      }),
    ))
    fireEvent.click(screen.getByRole('button', { name: /confirm frame match/i }))

    await waitFor(() => expect(axios.put).toHaveBeenCalledTimes(2))
    expect(await screen.findByRole('button', { name: /use confirmed pairs/i })).toBeEnabled()
  })

  it('stacks synchronized feeds around one timeline and edits exact Match In and Out pairs', async () => {
    const reviewJob = {
      ...activeJob, id: 'unified-job', state: 'awaiting_match_review', stage: 'awaiting_match_review', progress: .07,
      message: 'Review proposed frame matches', needs_review: true, matching_mode: 'guided',
    }
    const spans = [
      { id: 'intro', kind: 'difference', low_range: null, reference_range: { start_frame: 0, end_frame: 20 }, status: null, confidence: null, origin: 'automatic', sequence_start_seconds: 0, sequence_duration_seconds: 2 },
      { id: 'shared', kind: 'match', low_range: { start_frame: 0, end_frame: 40 }, reference_range: { start_frame: 20, end_frame: 60 }, status: 'proposed', confidence: .96, origin: 'automatic', sequence_start_seconds: 2, sequence_duration_seconds: 4 },
      { id: 'gap', kind: 'difference', low_range: { start_frame: 40, end_frame: 50 }, reference_range: { start_frame: 60, end_frame: 70 }, status: null, confidence: null, origin: 'automatic', sequence_start_seconds: 6, sequence_duration_seconds: 1 },
      { id: 'tail', kind: 'difference', low_range: { start_frame: 50, end_frame: 70 }, reference_range: null, status: null, confidence: null, origin: 'automatic', sequence_start_seconds: 7, sequence_duration_seconds: 2 },
    ]
    const review = {
      schema_version: 2, revision: 1, spans,
      summary: { proposed_blocks: 1, confirmed_blocks: 0, difference_blocks: 3, matched_seconds: 0 },
      media: { low: { fps: 10, frame_count: 70, duration: 7 }, reference: { fps: 10, frame_count: 70, duration: 7 } },
      proxy_urls: { low: '/low.mp4', reference: '/reference.mp4' },
    }
    axios.get.mockImplementation((url) => {
      if (url.endsWith('/system/status')) return Promise.resolve({ data: systemStatus })
      if (url.endsWith('/match-review')) return Promise.resolve({ data: review })
      return Promise.resolve({ data: { jobs: [reviewJob] } })
    })
    let currentReview = review
    axios.patch.mockImplementation((_, payload) => {
      let nextSpans = currentReview.spans
      let summary = currentReview.summary
      if (payload.operation === 'set_match_range') {
        nextSpans = nextSpans.map((span) => {
          if (span.id !== payload.span_id) return span
          const rest = { ...span }
          delete rest.match_draft
          return {
            ...rest, status: 'proposed', origin: 'manual',
            low_range: { start_frame: payload.low_start, end_frame: payload.low_end },
            reference_range: { start_frame: payload.reference_start, end_frame: payload.reference_end },
          }
        })
      } else if (payload.operation === 'set_match_draft') {
        nextSpans = nextSpans.map((span) => {
          if (span.id !== payload.span_id) return span
          if (!payload.draft) {
            const rest = { ...span }
            delete rest.match_draft
            return rest
          }
          const adjustment_baseline = span.kind === 'match' ? span.adjustment_baseline || {
            low_in: span.low_range.start_frame, low_out: span.low_range.end_frame - 1,
            reference_in: span.reference_range.start_frame, reference_out: span.reference_range.end_frame - 1,
          } : undefined
          return { ...span, match_draft: payload.draft, ...(adjustment_baseline ? { adjustment_baseline } : {}) }
        })
      } else if (payload.operation === 'apply_match_draft') {
        const applyingDifference = nextSpans.find((span) => span.id === payload.span_id)?.kind === 'difference'
        nextSpans = nextSpans.map((span) => {
          if (span.id !== payload.span_id) return span
          const draft = span.match_draft
          const rest = { ...span }
          delete rest.match_draft
          if (span.kind === 'difference') return {
            ...rest, id: 'manual-match', kind: 'match', status: 'proposed', origin: 'manual',
            low_range: { start_frame: draft.low_in, end_frame: draft.low_out + 1 },
            reference_range: { start_frame: draft.reference_in, end_frame: draft.reference_out + 1 },
          }
          return {
            ...rest, status: 'proposed', origin: 'manual',
            low_range: { start_frame: draft.low_in, end_frame: draft.low_out + 1 },
            reference_range: { start_frame: draft.reference_in, end_frame: draft.reference_out + 1 },
          }
        })
        if (applyingDifference) summary = { ...summary, proposed_blocks: 1 }
      } else if (payload.operation === 'set_status') {
        nextSpans = nextSpans.map((span) => span.id === payload.span_id ? { ...span, status: payload.status } : span)
        summary = { ...summary, proposed_blocks: 0, confirmed_blocks: 1, matched_seconds: 2.1 }
      } else if (payload.operation === 'create_match') {
        nextSpans = nextSpans.map((span) => span.id === payload.span_id ? {
          ...span, id: 'manual-match', kind: 'match', status: 'proposed', origin: 'manual',
          low_range: { start_frame: payload.low_start, end_frame: payload.low_end },
          reference_range: { start_frame: payload.reference_start, end_frame: payload.reference_end },
        } : span)
        summary = { ...summary, proposed_blocks: 1, confirmed_blocks: 1 }
      }
      currentReview = { ...currentReview, revision: currentReview.revision + 1, spans: nextSpans, summary }
      return Promise.resolve({ data: currentReview })
    })
    const playSpy = vi.spyOn(window.HTMLMediaElement.prototype, 'play').mockResolvedValue()

    render(<App />)

    expect(await screen.findByText(/one sequence, two source tracks/i)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /zoom in/i })).not.toBeInTheDocument()
    expect(screen.queryByRole('slider', { name: /timeline viewport position/i })).not.toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: /unpaired reference block, frames 0 through 19/i }).length).toBeGreaterThan(0)
    expect(screen.getAllByRole('button', { name: /unpaired low block, frames 50 through 69/i }).length).toBeGreaterThan(0)
    expect(screen.getByAltText(/supplemental.*frame 0/i)).toBeInTheDocument()
    expect(screen.getByAltText(/reference.*frame 20/i)).toBeInTheDocument()
    const highFeed = screen.getByLabelText(/reference.*playback feed/i)
    const timeline = screen.getByLabelText(/unified video timeline/i)
    const matchControls = screen.getByRole('group', { name: /frame match controls/i })
    const lowFeed = screen.getByLabelText(/supplemental.*playback feed/i)
    expect(matchControls).toContainElement(screen.getByRole('button', { name: /mark match in from playheads/i }))
    expect(matchControls).toContainElement(screen.getByRole('button', { name: /mark match out from playheads/i }))
    expect(matchControls).toContainElement(screen.getByRole('button', { name: /apply in \/ out/i }))
    expect(matchControls).toContainElement(screen.getByRole('button', { name: /confirm match/i }))
    expect(screen.queryByRole('button', { name: /save frame draft/i })).not.toBeInTheDocument()
    expect(matchControls).toContainElement(screen.getByRole('button', { name: /discard frame draft/i }))
    expect(timeline).toContainElement(screen.getByRole('slider', { name: /^unified timeline position$/i }))
    expect(screen.queryByText(/^unified timeline position$/i)).not.toBeInTheDocument()
    expect(highFeed.compareDocumentPosition(timeline) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(timeline.compareDocumentPosition(matchControls) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(matchControls.compareDocumentPosition(lowFeed) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(timeline.compareDocumentPosition(lowFeed) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: /play both timeline videos/i }))
    await waitFor(() => expect(playSpy).toHaveBeenCalledTimes(2))
    fireEvent.click(screen.getByRole('button', { name: /pause both timeline videos/i }))

    fireEvent.change(screen.getByRole('slider', { name: /^unified timeline position$/i }), { target: { value: 4 } })
    expect(screen.getByAltText(/supplemental.*frame 20/i)).toBeInTheDocument()
    expect(screen.getByAltText(/reference.*frame 40/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /use dense confirmed pairs/i })).toBeDisabled()

    fireEvent.change(screen.getByRole('slider', { name: /jog reference.*frame/i }), { target: { value: 41 } })
    fireEvent.click(screen.getByRole('button', { name: /mark match out from playheads/i }))
    await waitFor(() => expect(axios.patch).toHaveBeenCalledWith(
      expect.stringMatching(/match-review$/),
      expect.objectContaining({
        revision: 1, span_id: 'shared', operation: 'set_match_draft',
        draft: expect.objectContaining({ low_out: 20, reference_out: 41 }),
      }),
    ))
    expect(await screen.findByText(/adjustment saved.*apply when ready/i)).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: /saved adjustment low block/i }).length).toBeGreaterThan(0)
    fireEvent.click(screen.getAllByRole('button', { name: /unpaired reference block, frames 0 through 19/i })[0])
    fireEvent.click(screen.getByRole('button', { name: /saved adjustment low block, frames 0 through 39/i }))
    expect(await screen.findByText(/L20.*H41/i)).toBeInTheDocument()
    fireEvent.change(screen.getByRole('slider', { name: /^unified timeline position$/i }), { target: { value: 2 } })
    fireEvent.change(screen.getByRole('slider', { name: /jog reference.*frame/i }), { target: { value: 21 } })
    fireEvent.click(screen.getByRole('button', { name: /mark match in from playheads/i }))
    expect(await screen.findByText(/L0.*H21/i)).toBeInTheDocument()
    await waitFor(() => expect(axios.patch).toHaveBeenCalledWith(
      expect.stringMatching(/match-review$/),
      expect.objectContaining({ revision: 2, span_id: 'shared', operation: 'set_match_draft' }),
    ))
    fireEvent.click(screen.getByRole('button', { name: /apply in \/ out/i }))
    await waitFor(() => expect(axios.patch).toHaveBeenCalledWith(
      expect.stringMatching(/match-review$/),
      expect.objectContaining({
        revision: 3, span_id: 'shared', operation: 'apply_match_draft',
      }),
    ))
    expect(await screen.findByText(/H \+1/i)).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: /adjusted.*review low block/i }).length).toBeGreaterThan(0)
    fireEvent.click(await screen.findByRole('button', { name: /confirm match/i }))
    await waitFor(() => expect(axios.patch).toHaveBeenCalledWith(
      expect.stringMatching(/match-review$/),
      expect.objectContaining({ revision: 4, span_id: 'shared', operation: 'set_status', status: 'confirmed' }),
    ))
    expect(await screen.findByRole('button', { name: /use dense confirmed pairs/i })).toBeEnabled()

    fireEvent.change(screen.getByRole('slider', { name: /^unified timeline position$/i }), { target: { value: 1 } })
    expect(screen.getByText(/supplemental.*has no source video at/i)).toBeInTheDocument()
    expect(screen.getByAltText(/reference.*frame 10/i)).toBeInTheDocument()

    fireEvent.click(screen.getAllByRole('button', { name: /unpaired low block, frames 40 through 49/i })[0])
    expect(screen.getByRole('group', { name: /frame match controls/i })).toBeInTheDocument()
    fireEvent.change(screen.getByRole('slider', { name: /jog reference.*frame/i }), { target: { value: 61 } })
    fireEvent.click(screen.getByRole('button', { name: /mark match in from playheads/i }))
    await waitFor(() => expect(axios.patch).toHaveBeenCalledWith(
      expect.stringMatching(/match-review$/),
      expect.objectContaining({ revision: 5, span_id: 'gap', operation: 'set_match_draft' }),
    ))
    expect(await screen.findByText(/adjustment saved.*apply when ready/i)).toBeInTheDocument()
    fireEvent.change(screen.getByRole('slider', { name: /^unified timeline position$/i }), { target: { value: 6.5 } })
    fireEvent.click(screen.getByRole('button', { name: /mark match out from playheads/i }))
    await waitFor(() => expect(axios.patch).toHaveBeenCalledWith(
      expect.stringMatching(/match-review$/),
      expect.objectContaining({ revision: 6, span_id: 'gap', operation: 'set_match_draft' }),
    ))
    expect(await screen.findByText(/adjustment saved.*apply when ready/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /create proposed match/i }))
    await waitFor(() => expect(axios.patch).toHaveBeenCalledWith(
      expect.stringMatching(/match-review$/),
      expect.objectContaining({
        revision: 7, span_id: 'gap', operation: 'apply_match_draft',
      }),
    ))
    expect(await screen.findByRole('button', { name: /use dense confirmed pairs/i })).toBeDisabled()

    axios.patch.mockRejectedValueOnce({ response: { data: { detail: 'Draft storage unavailable' } } })
    fireEvent.click(screen.getByRole('button', { name: /mark match in from playheads/i }))
    expect(await screen.findByText(/adjustment not saved.*retry or discard/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /apply in \/ out/i })).toBeDisabled()
    fireEvent.click(screen.getByRole('button', { name: /retry saving frame draft/i }))
    expect(await screen.findByText(/adjustment saved.*apply when ready/i)).toBeInTheDocument()
  })
})
