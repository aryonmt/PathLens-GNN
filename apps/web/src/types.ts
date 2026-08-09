export type EntityType = 'drug' | 'protein'

export interface Versioned {
  model_version: string
  dataset_version: string
}

export interface Entity extends Versioned {
  entity_id: string
  entity_type: EntityType
  display_name: string
  degree: number
}

export interface EntitySearch extends Versioned {
  items: Entity[]
  next_cursor: number | null
}

export interface Recommendation {
  source_id: string
  target_id: string
  target_type: EntityType
  target_name: string
  rank: number
  pathlens_score: number
  bridge_count: number
}

export interface Recommendations extends Versioned {
  source: Entity
  items: Recommendation[]
}

export interface PairScore extends Versioned {
  source_id: string
  target_id: string
  drug_id: string
  protein_id: string
  known: boolean
  pathlens_score: number
  logit: number
  gate_weights: number[]
  expert_logits: number[]
  contributions: number[]
}

export interface Explanation extends Versioned {
  source_id: string
  target_id: string
  projection_context: {
    similar_drugs: Array<{ entity_id: string; score: number }>
    similar_proteins: Array<{ entity_id: string; score: number }>
  }
  bridge_paths: Array<{ nodes: string[]; weight: number }>
  channel_contributions: number[]
  truncated: boolean
}

export interface GraphNode {
  id: string
  label: string
  entity_type: EntityType
  degree: number
  role: string
  x?: number
  y?: number
  z?: number
}

export interface GraphLink {
  source: string | GraphNode
  target: string | GraphNode
  kind: 'known' | 'prediction' | 'support' | string
  weight: number
  hop: number
}

export interface GraphPayload extends Versioned {
  nodes: GraphNode[]
  links: GraphLink[]
  truncated: boolean
}

export interface Health {
  ready: boolean
  model_version?: string
  dataset_version?: string
  error?: string
}
