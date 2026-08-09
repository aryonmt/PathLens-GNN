import type {
  Entity,
  EntitySearch,
  Explanation,
  GraphPayload,
  Health,
  PairScore,
  Recommendations,
} from './types'

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(body.detail ?? `Request failed with status ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const api = {
  health: () => request<Health>('/api/v1/health'),
  search: (query: string, type?: 'drug' | 'protein') => {
    const params = new URLSearchParams({ q: query, limit: '25' })
    if (type) params.set('type', type)
    return request<EntitySearch>(`/api/v1/entities?${params}`)
  },
  entity: (id: string) => request<Entity>(`/api/v1/entities/${encodeURIComponent(id)}`),
  recommendations: (id: string) =>
    request<Recommendations>(`/api/v1/recommendations/${encodeURIComponent(id)}?limit=100`),
  predict: (sourceId: string, targetId: string) =>
    request<PairScore>('/api/v1/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source_id: sourceId, target_id: targetId }),
    }),
  explanation: (sourceId: string, targetId: string) =>
    request<Explanation>(
      `/api/v1/explanations/${encodeURIComponent(sourceId)}/${encodeURIComponent(targetId)}`,
    ),
  pairGraph: (sourceId: string, targetId: string) => {
    const params = new URLSearchParams({ source_id: sourceId, target_id: targetId })
    return request<GraphPayload>(`/api/v1/graph/pair?${params}`)
  },
  egoGraph: (id: string, hops: number) =>
    request<GraphPayload>(`/api/v1/graph/ego/${encodeURIComponent(id)}?hops=${hops}`),
  exportUrl: (id: string) =>
    `${API_BASE}/api/v1/exports/recommendations/${encodeURIComponent(id)}.csv`,
}
