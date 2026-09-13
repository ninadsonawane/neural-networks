import type { Architecture, Gates, SimulationConfig, StepSummary, Token, Trace } from './types';

export const DEFAULT_CONFIG: SimulationConfig = {
  contextLength: 3, temperature: 0.8, recurrentMultiplier: 0.72,
  pinClue: false, gatedVariant: 'lstm', attentionEnabled: true,
  sourceLength: 8, head: 2, layer: 1, aggregateHeads: false,
  positionalEncoding: true, causalMask: true,
};

const SIZE = 8;
const VOCABULARY = ['.', ' ', ...'abcdefghijklmnopqrstuvwxyz'];
const zero = () => Array<number>(SIZE).fill(0);
const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;
const norm = (values: number[]) => Math.hypot(...values);
const sigmoid = (value: number) => 1 / (1 + Math.exp(-value));
const clamp = (value: number, low: number, high: number, fallback: number) =>
  Number.isFinite(value) ? Math.max(low, Math.min(high, value)) : fallback;
const blankGates = (): Gates => ({ forget: 0, input: 0, candidate: 0, output: 0, update: 0, reset: 0 });

export function displayToken(text: string): string {
  return text === ' ' ? '␣' : text === '\n' ? '↵' : text === '\t' ? '⇥' : text;
}

/** A fixed character embedding, deliberately independent of the complete prompt. */
function embedding(text: string): number[] {
  const points = Array.from(text).map((character) => character.toLowerCase().codePointAt(0) ?? 0);
  return Array.from({ length: SIZE }, (_, d) => mean(points.map((point) =>
    0.65 * Math.sin(point * (d + 1) * 0.173) + 0.35 * Math.cos(point * 0.071 + d * 1.7))));
}

function softmax(scores: number[]): number[] {
  const maximum = Math.max(...scores);
  const values = scores.map((score) => score === -Infinity ? 0 : Math.exp(score - maximum));
  const total = values.reduce((a, b) => a + b, 0);
  return values.map((value) => value / total);
}

function readout(hidden: number[], temperature: number) {
  const logits = VOCABULARY.map((character, v) => {
    const projection = hidden.reduce((sum, value, d) => sum + value * Math.sin((v + 1) * (d + 2) * 0.39), 0);
    return projection * 0.65 + (' etaoin'.includes(character) ? 0.5 : 0);
  });
  const weights = softmax(logits.map((logit) => logit / temperature));
  return {
    logits,
    probabilities: VOCABULARY.map((token, index) => ({ token, probability: weights[index] }))
      .sort((a, b) => b.probability - a.probability),
  };
}

function summary(index: number, hidden: number[], cell: number[], gates: Gates,
  retainedIndices: number[], gradients: number[], clueStrength: number, temperature: number): StepSummary {
  return { index, hidden: [...hidden], cell: [...cell], hiddenNorm: norm(hidden), cellNorm: norm(cell),
    gates, retainedIndices, gradients, clueStrength, ...readout(hidden, temperature) };
}

function normalizeConfig(config: SimulationConfig): SimulationConfig {
  return { ...config,
    contextLength: Math.round(clamp(config.contextLength, 1, 32, 3)),
    temperature: clamp(config.temperature, 0.1, 3, 0.8),
    recurrentMultiplier: clamp(config.recurrentMultiplier, 0.05, 1.5, 0.72),
    sourceLength: Math.round(clamp(config.sourceLength, 4, config.customAttention ? 80 : 16, 8)),
    head: Math.round(clamp(config.head, 0, 3, 2)),
    layer: Math.round(clamp(config.layer, 0, 3, 1)),
    headCount: Math.round(clamp(config.headCount ?? 4, 1, 4, 4)),
    attentionSharpness: clamp(config.attentionSharpness ?? 1, .2, 4, 1),
  };
}

function fixedWindow(tokens: Token[], config: SimulationConfig): StepSummary[] {
  return tokens.map((_, index) => {
    const start = Math.max(0, index - config.contextLength + 1);
    const window = tokens.slice(start, index + 1);
    const vectors = window.map((token) => embedding(token.text));
    // Window-relative positions are the only positional input. Changing any prefix
    // outside this window cannot change hidden state, logits, or probabilities.
    const hidden = Array.from({ length: SIZE }, (_, d) => Math.tanh(vectors.reduce((sum, vector, position) =>
      sum + vector[d] * (0.35 + (position + 1) / config.contextLength) * Math.cos((position + 1) * (d + 1) * 0.17), 0)));
    const retained = window.map((token) => token.position);
    return summary(index, hidden, zero(), blankGates(), retained,
      Array.from({ length: index + 1 }, (_, j) => j >= start ? 1 : 0),
      window.some((token) => token.isClue) ? 1 : 0, config.temperature);
  });
}

function recurrent(tokens: Token[], config: SimulationConfig): StepSummary[] {
  let hidden = zero();
  const clueIndex = tokens.findIndex((token) => token.isClue);
  return tokens.map((token, index) => {
    const input = embedding(token.text);
    const previous = hidden;
    hidden = input.map((value, d) => Math.tanh(0.64 * value + config.recurrentMultiplier *
      (0.84 * previous[d] + 0.08 * previous[(d + 1) % SIZE])));
    // This is r^distance, an explicit teaching proxy. It is not a derivative
    // through tanh or the full recurrent Jacobian.
    const gradients = Array.from({ length: index + 1 }, (_, j) => config.recurrentMultiplier ** (index - j));
    const retained = gradients.flatMap((strength, j) => strength >= 0.05 ? [j] : []);
    const clueStrength = clueIndex >= 0 && index >= clueIndex ? Math.min(1, gradients[clueIndex]) : 0;
    return summary(index, hidden, zero(), blankGates(), retained, gradients, clueStrength, config.temperature);
  });
}

function gated(tokens: Token[], config: SimulationConfig): StepSummary[] {
  let hidden = zero();
  let cell = zero();
  let clueStrength = 0;
  const retention: number[] = [];
  const clueIndex = tokens.findIndex((token) => token.isClue);
  const gateValue = (gate: keyof Gates, automatic: number) => config.gateOverrides?.[gate] === undefined
    ? automatic : clamp(config.gateOverrides[gate]!, gate === 'candidate' ? -1 : 0, 1, automatic);
  return tokens.map((token, index) => {
    const input = embedding(token.text);
    const previous = hidden;
    const preserving = config.pinClue && index > clueIndex && clueIndex >= 0;
    const acquiring = config.pinClue && token.isClue;
    let gates: Gates;
    let retainedFactor: number;
    if (config.gatedVariant === 'gru') {
      // Convention: z controls the NEW candidate; (1-z) preserves the old state.
      const update = input.map((value, d) => gateValue('update', preserving ? 0.015 : acquiring ? 0.96 : sigmoid(-0.4 + value * 0.7 + previous[d] * 0.25)));
      const reset = input.map((value, d) => gateValue('reset', sigmoid(0.4 + value * 0.8 - previous[d] * 0.3)));
      const candidate = input.map((value, d) => gateValue('candidate', Math.tanh(value * 0.9 + reset[d] * previous[d] * 0.45)));
      hidden = candidate.map((value, d) => (1 - update[d]) * previous[d] + update[d] * value);
      cell = zero();
      gates = { forget: 1 - mean(update), input: mean(update), candidate: mean(candidate), output: 1, update: mean(update), reset: mean(reset) };
      retainedFactor = 1 - gates.update;
    } else {
      const forget = input.map((value, d) => gateValue('forget', preserving ? 0.995 : sigmoid(0.85 + value * 0.5 + previous[d] * 0.3)));
      const write = input.map((value, d) => gateValue('input', preserving ? 0.006 : acquiring ? 0.96 : sigmoid(-0.3 + value * 0.7 - previous[d] * 0.2)));
      const candidate = input.map((value, d) => gateValue('candidate', Math.tanh(value * 0.9 + previous[d] * 0.25)));
      const output = input.map((value, d) => gateValue('output', sigmoid(0.6 + value * 0.4 + previous[d] * 0.2)));
      cell = cell.map((value, d) => forget[d] * value + write[d] * candidate[d]);
      hidden = cell.map((value, d) => output[d] * Math.tanh(value));
      gates = { forget: mean(forget), input: mean(write), candidate: mean(candidate), output: mean(output), update: 0, reset: 0 };
      retainedFactor = gates.forget;
    }
    retention.push(retainedFactor);
    clueStrength = token.isClue ? 1 : clueStrength * retainedFactor;
    const gradients = Array.from({ length: index + 1 }, (_, j) => retention.slice(j + 1).reduce((a, b) => a * b, 1));
    const retained = gradients.flatMap((strength, j) => strength >= 0.05 ? [j] : []);
    return summary(index, hidden, cell, gates, retained, gradients, clueStrength, config.temperature);
  });
}

function dot(a: number[], b: number[]): number {
  return a.reduce((sum, value, index) => sum + value * b[index], 0) / Math.sqrt(SIZE);
}

function transformer(tokens: Token[], config: SimulationConfig): { steps: StepSummary[]; attention: number[][] } {
  let states = tokens.map((token) => embedding(token.text).map((value, dimension) =>
    value + (config.positionalEncoding ? 0.24 * Math.sin(token.position / 10 ** (dimension / SIZE)) : 0)));
  let attention: number[][] = [];
  const clue = tokens.find((token) => token.isClue);
  const headWeights = (query: number, head: number, layer: number): number[] => softmax(tokens.map((source, key) => {
    if (config.causalMask && key > query) return -Infinity;
    if (config.attentionContext && key < query - config.attentionContext + 1) return -Infinity;
    const distance = Math.abs(query - key);
    let score = dot(states[query], states[key]) * (0.4 + layer * 0.15);
    if (head === 0) score += config.positionalEncoding ? -distance * (0.7 - layer * 0.1) : 0;
    if (head === 1) score += source.text.toLowerCase() === tokens[query].text.toLowerCase() ? 3.5 : 0;
    if (head === 2) score += source.text.toLowerCase() === clue?.text.toLowerCase() ? 4.6 : 0;
    if (head === 3 && config.positionalEncoding) score += 1.6 * Math.cos((query - key) * (0.7 + layer * 0.2));
    return score * (config.attentionSharpness ?? 1);
  }));
  for (let layer = 0; layer <= config.layer; layer++) {
    attention = tokens.map((_, query) => {
      if (!config.aggregateHeads) return headWeights(query, config.head, layer);
      const heads = Array.from({ length: config.headCount ?? 4 }, (_, head) => headWeights(query, head, layer));
      return tokens.map((_, key) => mean(heads.map((weights) => weights[key])));
    });
    const previous = states;
    states = attention.map((weights, query) => Array.from({ length: SIZE }, (_, d) =>
      Math.tanh(previous[query][d] * 0.3 + weights.reduce((sum, weight, key) => sum + weight * previous[key][d], 0))));
  }
  const steps = states.map((hidden, index) => {
    const weights = attention[index];
    const retained = weights.flatMap((weight, key) => weight >= 0.025 ? [key] : []);
    return summary(index, hidden, zero(), blankGates(), retained, weights.slice(0, index + 1),
      clue ? weights[clue.position] : 0, config.temperature);
  });
  return { steps, attention };
}

function translation(tokens: Token[], config: SimulationConfig) {
  if (config.customAttention) {
    const encoderTokens=tokens.map(token=>token.text), decoderTokens=[...encoderTokens], encoderStates=encoderTokens.map(embedding);
    const alignment=tokens.map((query,i)=>softmax(tokens.map((source,j)=>j>i||j<i-config.sourceLength+1?-Infinity:((source.isClue?3.5:source.text===query.text?2:0)-Math.abs(i-j)*.04+(config.head===1&&j===i?2:config.head===2&&source.isClue?2:config.head===3?Math.cos(i-j):0))*(config.attentionSharpness??1))));
    const clue=tokens.findIndex(t=>t.isClue);
    const steps=tokens.map((_,i)=>{const weights=alignment[i];const hidden=Array.from({length:SIZE},(_,d)=>Math.tanh(encoderStates.reduce((sum,state,j)=>sum+state[d]*weights[j],0)));return summary(i,hidden,zero(),blankGates(),weights.flatMap((w,j)=>w>.025?[j]:[]),weights.slice(0,i+1),clue>=0?weights[clue]:0,config.temperature);});
    return {encoderTokens,decoderTokens,encoderStates,alignment,steps};
  }
  const source = ['meri', 'billi', 'ghar', 'par', 'hai', 'aaj', 'bahut', 'shaant', 'aur', 'khush', 'kyunki', 'baarish', 'ab', 'ruk', 'gayi', 'hai'];
  const target = ['my', 'cat', 'home', 'at', 'is', 'today', 'very', 'calm', 'and', 'happy', 'because', 'rain', 'now', 'stopped', 'has', 'it'];
  const encoderTokens = source.slice(0, config.sourceLength);
  const order = [0, 1, 4, 3, 2, ...Array.from({ length: 11 }, (_, i) => i + 5)].filter((i) => i < config.sourceLength);
  const decoderTokens = order.map((i) => target[i]);
  const encoderStates = encoderTokens.map(embedding);
  const alignment = order.map((sourceIndex) => softmax(encoderTokens.map((_, index) =>
    config.attentionEnabled ? (index === sourceIndex ? 3.5 : 0) - Math.abs(index - sourceIndex) * 0.3 : 0)));
  const compressed = Array.from({ length: SIZE }, (_, d) => Math.tanh(mean(encoderStates.map((state) => state[d])) *
    Math.min(1, 5 / config.sourceLength)));
  const steps = tokens.map((_, index) => {
    const weights = alignment[index % alignment.length];
    const hidden = config.attentionEnabled ? Array.from({ length: SIZE }, (_, d) =>
      Math.tanh(encoderStates.reduce((sum, state, j) => sum + state[d] * weights[j], 0))) : compressed;
    return summary(index, hidden, zero(), blankGates(), [], Array<number>(index + 1).fill(0),
      config.attentionEnabled ? 1 : Math.min(1, 5 / config.sourceLength), config.temperature);
  });
  return { encoderTokens, decoderTokens, encoderStates, alignment, steps };
}

export function buildTrace(input: string, architecture: Architecture, rawConfig: SimulationConfig): Trace {
  const config = normalizeConfig(rawConfig);
  const characters = Array.from(input).slice(0, 80);
  if (!characters.length) characters.push(' ');
  const clueIndex = config.clueIndex === undefined ? characters.findIndex((character) => /\p{L}/u.test(character)) : Math.max(0,Math.min(characters.length-1,Math.floor(config.clueIndex)));
  const tokens = characters.map((text, position) => ({ id: position, text, position, isClue: position === clueIndex }));
  const trace: Trace = {
    schemaVersion: 1, provenance: 'educational-simulation', architecture, input: characters.join(''),
    tokens, vocabulary: [...VOCABULARY], steps: [], attention: [],
    encoderTokens: [], decoderTokens: [], encoderStates: [], alignment: [],
    metadata: { label: 'Educational simulation', description: 'Deterministic eight-dimensional teaching states and illustrative character probabilities; no trained model or checkpoint is running.' },
  };
  if (architecture === 'mlp') trace.steps = fixedWindow(tokens, config);
  if (architecture === 'rnn') {
    trace.steps = recurrent(tokens, config);
    trace.metadata.description += ' Backward signal uses r^distance, a sensitivity proxy, not the exact derivative through tanh.';
  }
  if (architecture === 'lstm') {
    trace.steps = gated(tokens, config);
    trace.metadata.description += ` Actual small ${config.gatedVariant.toUpperCase()} gate equations with fixed illustrative weights. Retention is a gate-product proxy. GRU is planned repository work.`;
  }
  if (architecture === 'transformer') {
    Object.assign(trace, transformer(tokens, config));
    trace.metadata.description += ' Four hand-designed attention heads illustrate locality, repetition, distant clue retrieval, and position. These are not learned heads.';
  }
  if (architecture === 'attention') {
    Object.assign(trace, translation(tokens, config));
    trace.metadata.description += config.customAttention ? ' User-character source retrieval with deterministic alignment scores; this is not translation or trained inference.' : ' Separate conceptual word-level Hinglish → English alignment, with illustrative token correspondences; this is not translation inference. Disabled attention shows uniform readout from a compressed state.';
  }
  return trace;
}
