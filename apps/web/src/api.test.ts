import { afterEach, describe, expect, it, vi } from 'vitest'

import { api } from './api'

describe('api client', () => {
  afterEach(() => vi.restoreAllMocks())

  it('encodes pair requests as typed JSON', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue(
      new Response(JSON.stringify({ pathlens_score: 51 }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    )
    await api.predict('DB00001', 'P00002')
    expect(fetchMock).toHaveBeenCalledWith('/api/v1/predict', expect.objectContaining({ method: 'POST' }))
    expect(JSON.parse(String(fetchMock.mock.calls[0][1]?.body))).toEqual({
      source_id: 'DB00001',
      target_id: 'P00002',
    })
  })
})
