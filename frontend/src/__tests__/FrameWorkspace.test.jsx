import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, expect, it, vi } from 'vitest'
import axios from 'axios'
import FrameWorkspace from '../FrameWorkspace'
vi.mock('axios')
const media = { width: 80, height: 48, frame_count: 10, duration: 1, variable_frame_rate: true, issues: [] }
const initial = { revision: 1, ranges: [], locks: [], history: [], geometry: { framing: 'fit', crops: {} }, media: { low: media, reference: media }, storyboard: [], output_duration_seconds: 2 }
beforeEach(() => {
  vi.resetAllMocks()
  axios.get.mockResolvedValue({ data: { frames: Array.from({ length: 10 }, (_, i) => ({ frame_index: i, time_seconds: i * .1, duration_seconds: .1 })) } })
})
function mount(review = initial) { return render(<FrameWorkspace api="/api/v1" job={{ id: 'job' }} initialReview={review} onQueued={vi.fn()} />) }
it('waits for exact images and locks only the displayed pair', async () => {
  axios.patch.mockResolvedValue({ data: { ...initial, revision: 2, can_undo: true, locks: [{ id: 'pair', low_frame: 0, reference_frame: 0 }] } })
  mount()
  const lock = screen.getByRole('button', { name: 'Lock A 00001 ↔ B 00001' })
  expect(lock).toBeDisabled()
  fireEvent.load(screen.getByAltText('A exact source frame 00001'))
  expect(lock).toBeDisabled()
  fireEvent.load(screen.getByAltText('B exact source frame 00001'))
  fireEvent.click(lock)
  await waitFor(() => expect(axios.patch).toHaveBeenCalledWith('/api/v1/jobs/job/match-review', expect.objectContaining({ operation: 'lock_pair', low_frame: 0, reference_frame: 0, revision: 1 })))
  expect(await screen.findByAltText('Saved A 00001')).toBeInTheDocument()
  expect(screen.getByText('This exact pair is locked')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'Next A frame' }))
  expect(screen.getByRole('button', { name: 'Lock A 00002 ↔ B 00001' })).toBeDisabled()
  expect(axios.patch).toHaveBeenCalledTimes(1)
})
it('does not step frames when arrows are used inside frame entry', () => {
  mount()
  fireEvent.keyDown(screen.getByRole('spinbutton', { name: 'A frame number' }), { key: 'ArrowRight' })
  expect(screen.getByAltText('A exact source frame 00001')).toBeInTheDocument()
  expect(axios.patch).not.toHaveBeenCalled()
})
it('keeps the candidate and saved locks on a revision conflict', async () => {
  axios.patch.mockRejectedValue({ response: { data: { detail: 'Review changed; reload before saving' } } })
  mount()
  fireEvent.load(screen.getByAltText('A exact source frame 00001')); fireEvent.load(screen.getByAltText('B exact source frame 00001'))
  fireEvent.click(screen.getByRole('button', { name: 'Lock A 00001 ↔ B 00001' }))
  expect(await screen.findByRole('alert')).toHaveTextContent('Review changed')
  expect(screen.getByAltText('A exact source frame 00001')).toBeInTheDocument()
})
