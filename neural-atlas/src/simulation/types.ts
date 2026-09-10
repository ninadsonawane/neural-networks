export type Architecture = 'mlp' | 'rnn' | 'lstm' | 'attention' | 'transformer';
export interface SimulationConfig {
  contextLength: number;
  temperature: number;
  recurrentMultiplier: number;
  pinClue: boolean;
  gatedVariant: 'lstm' | 'gru';
  attentionEnabled: boolean;
  sourceLength: number;
  head: number;
  layer: number;
  aggregateHeads: boolean;
  positionalEncoding: boolean;
  causalMask: boolean;
  gateOverrides?: Partial<Gates>;
}
export interface Token { id: number; text: string; position: number; isClue: boolean }
export interface Probability { token: string; probability: number }
export interface Gates { forget: number; input: number; candidate: number; output: number; update: number; reset: number }
export interface StepSummary {
  index: number;
  hidden: number[];
  cell: number[];
  hiddenNorm: number;
  cellNorm: number;
  gates: Gates;
  retainedIndices: number[];
  logits: number[];
  probabilities: Probability[];
  gradients: number[];
  clueStrength: number;
}
export interface Trace {
  schemaVersion: 1;
  provenance: 'educational-simulation' | 'imported-trace';
  architecture: Architecture;
  input: string;
  tokens: Token[];
  vocabulary: string[];
  steps: StepSummary[];
  attention: number[][];
  encoderTokens: string[];
  decoderTokens: string[];
  encoderStates: number[][];
  alignment: number[][];
  metadata: { label: string; description: string; source?: string };
}
export interface TraceAdapter { load(input: string, architecture: Architecture, config: SimulationConfig): Trace | Promise<Trace> }
