import {
  Activity,
  Beaker,
  Box,
  ChevronRight,
  Download,
  GitBranch,
  Search,
  Sparkles,
} from 'lucide-react'
import { FormEvent, useEffect, useMemo, useState } from 'react'

import { api } from './api'
import { GraphExplorer } from './components/GraphExplorer'
import type {
  Entity,
  Explanation,
  GraphPayload,
  Health,
  PairScore,
  Recommendation,
  Recommendations,
} from './types'

const CHANNELS = ['Direct context', '2-hop projection', '3-hop bridge']

export default function App() {
  const [health, setHealth] = useState<Health | null>(null)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<Entity[]>([])
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(null)
  const [recommendations, setRecommendations] = useState<Recommendations | null>(null)
  const [selectedRecommendation, setSelectedRecommendation] = useState<Recommendation | null>(null)
  const [score, setScore] = useState<PairScore | null>(null)
  const [explanation, setExplanation] = useState<Explanation | null>(null)
  const [graph, setGraph] = useState<GraphPayload | null>(null)
  const [graphMode, setGraphMode] = useState<'focused' | 'ego'>('focused')
  const [hops, setHops] = useState(2)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')

  useEffect(() => {
    let cancelled = false
    const wake = async () => {
      try {
        const state = await api.health()
        if (!cancelled) setHealth(state)
      } catch {
        if (!cancelled) setHealth({ ready: false, error: 'Backend is waking up or unavailable.' })
      }
    }
    void wake()
    const timer = window.setInterval(wake, 10_000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [])

  async function runSearch(event: FormEvent) {
    event.preventDefault()
    if (!query.trim()) return
    setMessage('')
    try {
      const response = await api.search(query.trim())
      setResults(response.items)
      if (!response.items.length) setMessage('No known graph entity matched that search.')
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Search failed')
    }
  }

  async function chooseEntity(entity: Entity) {
    setLoading(true)
    setSelectedEntity(entity)
    setSelectedRecommendation(null)
    setScore(null)
    setExplanation(null)
    setGraph(null)
    setResults([])
    try {
      setRecommendations(await api.recommendations(entity.entity_id))
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not load recommendations')
    } finally {
      setLoading(false)
    }
  }

  async function chooseRecommendation(item: Recommendation) {
    if (!selectedEntity) return
    setLoading(true)
    setMessage('')
    setSelectedRecommendation(item)
    setGraphMode('focused')
    try {
      const [pairScore, evidence, graphPayload] = await Promise.all([
        api.predict(selectedEntity.entity_id, item.target_id),
        api.explanation(selectedEntity.entity_id, item.target_id),
        api.pairGraph(selectedEntity.entity_id, item.target_id),
      ])
      setScore(pairScore)
      setExplanation(evidence)
      setGraph(graphPayload)
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Could not load pair evidence')
    } finally {
      setLoading(false)
    }
  }

  async function switchGraph(nextMode: 'focused' | 'ego', nextHops = hops) {
    if (!selectedEntity || !selectedRecommendation) return
    setGraphMode(nextMode)
    setLoading(true)
    try {
      setGraph(
        nextMode === 'focused'
          ? await api.pairGraph(selectedEntity.entity_id, selectedRecommendation.target_id)
          : await api.egoGraph(selectedEntity.entity_id, nextHops),
      )
    } finally {
      setLoading(false)
    }
  }

  const maxContribution = useMemo(
    () => Math.max(0.001, ...(score?.contributions.map(Math.abs) ?? [1])),
    [score],
  )

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-mark"><GitBranch size={20} /></div>
        <div>
          <div className="brand">PathLens<span>GNN</span></div>
          <div className="subtitle">Molecular interaction intelligence</div>
        </div>
        <div className={`status ${health?.ready ? 'ready' : 'waking'}`}>
          <i />{health?.ready ? `Model ${health.model_version}` : 'Backend waking'}
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="eyebrow"><Sparkles size={14} /> Path-aware research prioritization</div>
          <h1>See the graph evidence<br />behind the next candidate.</h1>
          <p>Rank unknown drug–target pairs already represented in BioSNAP, then inspect the model channels and structural paths that influenced each score.</p>
          <form className="search-box" onSubmit={runSearch}>
            <Search size={20} />
            <input
              aria-label="Search a known drug or protein"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search DrugBank ID, UniProt ID, or bundled name…"
            />
            <button type="submit">Explore</button>
          </form>
          {results.length > 0 && (
            <div className="search-results">
              {results.map((entity) => (
                <button key={entity.entity_id} onClick={() => void chooseEntity(entity)}>
                  <span className={`entity-icon ${entity.entity_type}`}><Box size={16} /></span>
                  <span><strong>{entity.display_name}</strong><small>{entity.entity_id} · {entity.entity_type}</small></span>
                  <ChevronRight size={17} />
                </button>
              ))}
            </div>
          )}
          {message && <div className="message">{message}</div>}
        </section>

        <section className="workspace">
          <aside className="ranking-panel panel">
            <div className="panel-heading">
              <div><span className="kicker">Discovery</span><h2>Ranked candidates</h2></div>
              {selectedEntity && (
                <a className="icon-button" href={api.exportUrl(selectedEntity.entity_id)} title="Export CSV"><Download size={17} /></a>
              )}
            </div>
            {selectedEntity ? (
              <div className="source-card">
                <span className={`entity-icon ${selectedEntity.entity_type}`}><Beaker size={17} /></span>
                <div><strong>{selectedEntity.display_name}</strong><small>{selectedEntity.entity_id} · degree {selectedEntity.degree}</small></div>
              </div>
            ) : <div className="empty-state">Search and select a known graph entity to begin.</div>}
            <div className="rank-list">
              {recommendations?.items.map((item) => (
                <button
                  className={selectedRecommendation?.target_id === item.target_id ? 'active' : ''}
                  key={item.target_id}
                  onClick={() => void chooseRecommendation(item)}
                >
                  <span className="rank">{String(item.rank).padStart(2, '0')}</span>
                  <span className="candidate"><strong>{item.target_name}</strong><small>{item.target_id} · {item.bridge_count ? '3-hop support' : 'model context'}</small></span>
                  <span className="mini-score">{item.pathlens_score.toFixed(1)}</span>
                </button>
              ))}
            </div>
          </aside>

          <section className="graph-panel panel">
            <div className="panel-heading graph-tools">
              <div><span className="kicker">Evidence graph</span><h2>{graphMode === 'focused' ? 'Focused pair explanation' : `${hops}-hop ego network`}</h2></div>
              <div className="segmented">
                <button className={graphMode === 'focused' ? 'active' : ''} onClick={() => void switchGraph('focused')}>Focused</button>
                <button className={graphMode === 'ego' ? 'active' : ''} onClick={() => void switchGraph('ego')}>Ego</button>
              </div>
            </div>
            {graphMode === 'ego' && (
              <div className="hop-control">
                {[1, 2, 3].map((value) => <button key={value} className={hops === value ? 'active' : ''} onClick={() => { setHops(value); void switchGraph('ego', value) }}>{value}-hop</button>)}
              </div>
            )}
            <GraphExplorer data={graph} loading={loading} mode={graphMode} />
          </section>

          <aside className="evidence-panel panel">
            <div className="panel-heading"><div><span className="kicker">Model behavior</span><h2>Evidence profile</h2></div></div>
            {score ? (
              <>
                <div className="score-orbit"><div><strong>{score.pathlens_score.toFixed(1)}</strong><span>PathLens score</span></div></div>
                <p className="score-copy">Relative research-priority signal for this graph candidate—not a biological probability.</p>
                <div className="contributions">
                  {CHANNELS.map((channel, index) => (
                    <div key={channel}>
                      <div className="metric-label"><span>{channel}</span><strong>{score.contributions[index].toFixed(3)}</strong></div>
                      <div className="bar"><i style={{ width: `${Math.abs(score.contributions[index]) / maxContribution * 100}%` }} /></div>
                      <small>Gate {(score.gate_weights[index] * 100).toFixed(1)}%</small>
                    </div>
                  ))}
                </div>
                <div className="path-summary">
                  <div><Activity size={17} /><span><strong>{explanation?.bridge_paths.length ?? 0}</strong> displayed 3-hop bridge path(s)</span></div>
                  <div><GitBranch size={17} /><span><strong>{(explanation?.projection_context.similar_drugs.length ?? 0) + (explanation?.projection_context.similar_proteins.length ?? 0)}</strong> projection-context neighbors</span></div>
                </div>
                {explanation?.bridge_paths[0] && (
                  <div className="top-path"><span>Top structural path</span><code>{explanation.bridge_paths[0].nodes.join(' → ')}</code></div>
                )}
              </>
            ) : <div className="empty-state">Select a candidate to inspect channel contributions and graph evidence.</div>}
          </aside>
        </section>

        <section className="disclaimer">
          <Activity size={18} />
          <p><strong>Research use only.</strong> Unknown non-edges may include undiscovered positives. PathLens output is not experimental validation, causal evidence, diagnosis, or medical advice.</p>
        </section>
      </main>
    </div>
  )
}
