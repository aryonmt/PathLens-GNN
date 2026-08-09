import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ForceGraph3D, { type ForceGraphMethods } from 'react-force-graph-3d'
import * as THREE from 'three'

import type { GraphLink, GraphNode, GraphPayload } from '../types'

interface GraphExplorerProps {
  data: GraphPayload | null
  loading: boolean
  mode: 'focused' | 'ego'
}

function nodeId(value: string | GraphNode): string {
  return typeof value === 'string' ? value : value.id
}

export function GraphExplorer({ data, loading, mode }: GraphExplorerProps) {
  const graphRef = useRef<ForceGraphMethods<GraphNode, GraphLink> | undefined>(undefined)
  const containerRef = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ width: 900, height: 560 })

  useEffect(() => {
    if (!containerRef.current) return
    const observer = new ResizeObserver(([entry]) => {
      setSize({
        width: Math.max(320, Math.floor(entry.contentRect.width)),
        height: Math.max(420, Math.floor(entry.contentRect.height)),
      })
    })
    observer.observe(containerRef.current)
    return () => observer.disconnect()
  }, [])

  const graphData = useMemo(
    () => ({ nodes: data?.nodes ?? [], links: data?.links ?? [] }),
    [data],
  )

  const focusNode = useCallback((node: GraphNode) => {
    const distance = 68
    const x = node.x ?? 1
    const y = node.y ?? 1
    const z = node.z ?? 1
    const magnitude = Math.hypot(x, y, z) || 1
    const ratio = 1 + distance / magnitude
    graphRef.current?.cameraPosition(
      { x: x * ratio, y: y * ratio, z: z * ratio },
      { x, y, z },
      900,
    )
  }, [])

  const makeNode = useCallback((node: GraphNode) => {
    const isDrug = node.entity_type === 'drug'
    const geometry = isDrug
      ? new THREE.IcosahedronGeometry(node.role === 'candidate' ? 6 : 4.2, 1)
      : new THREE.OctahedronGeometry(node.role === 'candidate' ? 6 : 4.4, 0)
    const color = isDrug ? '#27d3c2' : '#f7b955'
    const material = new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: node.role === 'candidate' ? 0.55 : 0.15,
      roughness: 0.36,
      metalness: 0.28,
      transparent: true,
      opacity: node.role === 'context' ? 0.82 : 1,
    })
    return new THREE.Mesh(geometry, material)
  }, [])

  return (
    <div className="graph-shell" ref={containerRef} aria-label={`${mode} 3D graph`}>
      {loading && <div className="graph-overlay"><span className="spinner" />Loading evidence…</div>}
      {!loading && graphData.nodes.length === 0 && (
        <div className="graph-overlay muted">Select a ranked candidate to open the 3D evidence graph.</div>
      )}
      <ForceGraph3D<GraphNode, GraphLink>
        ref={graphRef}
        width={size.width}
        height={size.height}
        graphData={graphData}
        backgroundColor="#071117"
        showNavInfo={false}
        nodeLabel={(node) => `${node.label} · ${node.entity_type} · degree ${node.degree}`}
        nodeThreeObject={makeNode}
        linkColor={(link) =>
          link.kind === 'prediction' ? '#f56f8a' : link.kind === 'support' ? '#f7b955' : '#4f6b72'
        }
        linkWidth={(link) => (link.kind === 'prediction' ? 3.4 : link.kind === 'support' ? 2.2 : 0.8)}
        linkOpacity={0.72}
        linkDirectionalParticles={(link) => (link.kind === 'prediction' ? 3 : link.kind === 'support' ? 1 : 0)}
        linkDirectionalParticleColor={(link) => (link.kind === 'prediction' ? '#ffffff' : '#f7b955')}
        linkDirectionalParticleWidth={2.2}
        linkDirectionalParticleSpeed={0.006}
        cooldownTicks={mode === 'focused' ? 70 : 110}
        d3AlphaDecay={0.035}
        d3VelocityDecay={0.38}
        onNodeClick={focusNode}
        onEngineStop={() => graphRef.current?.zoomToFit(700, 70)}
        linkSource="source"
        linkTarget="target"
        nodeId="id"
        onLinkClick={(link) => {
          const source = graphData.nodes.find((node) => node.id === nodeId(link.source))
          if (source) focusNode(source)
        }}
      />
      <div className="graph-legend">
        <span><i className="dot drug" />Drug</span>
        <span><i className="dot protein" />Protein</span>
        <span><i className="line prediction" />Candidate</span>
        <span><i className="line support" />3-hop support</span>
      </div>
    </div>
  )
}
