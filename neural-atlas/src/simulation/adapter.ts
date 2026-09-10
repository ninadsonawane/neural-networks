import { buildTrace } from './engine';
import type { Architecture, Gates, SimulationConfig, StepSummary, Trace, TraceAdapter } from './types';

const ARCHITECTURES: Architecture[] = ['mlp', 'rnn', 'lstm', 'attention', 'transformer'];
const MAX_JSON_BYTES = 2 * 1024 * 1024;

function fail(path: string, message: string): never {
  throw new Error(`Invalid trace at ${path}: ${message}`);
}
function object(value: unknown, path: string): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) fail(path, 'expected an object');
  return value as Record<string, unknown>;
}
function list(value: unknown, path: string, maximum = 4096): unknown[] {
  if (!Array.isArray(value) || value.length > maximum) fail(path, `expected an array with at most ${maximum} entries`);
  return value;
}
function string(value: unknown, path: string, maximum = 4096, allowEmpty = false): string {
  if (typeof value !== 'string' || value.length > maximum || (!allowEmpty && !value.length)) fail(path, 'expected a nonempty string within the size limit');
  return value;
}
function number(value: unknown, path: string, low = -Number.MAX_VALUE, high = Number.MAX_VALUE): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < low || value > high) fail(path, `expected a finite number between ${low} and ${high}`);
  return value;
}
function integer(value: unknown, path: string, low: number, high: number): number {
  const result = number(value, path, low, high);
  if (!Number.isSafeInteger(result)) fail(path, 'expected an integer');
  return result;
}
function vector(value: unknown, path: string, dimension?: number, low?: number, high?: number): number[] {
  const values = list(value, path);
  if (dimension !== undefined && values.length !== dimension) fail(path, `expected ${dimension} entries, received ${values.length}`);
  return values.map((entry, i) => number(entry, `${path}[${i}]`, low, high));
}
function distribution(value: unknown, path: string, dimension: number): number[] {
  const values = vector(value, path, dimension, 0, 1);
  if (Math.abs(values.reduce((sum, entry) => sum + entry, 0) - 1) > 1e-5) fail(path, 'probabilities must sum to 1');
  return values;
}
function matrix(value: unknown, path: string, rows: number, columns: number, normalized: boolean): number[][] {
  const values = list(value, path, 256);
  if (values.length !== rows) fail(path, `expected ${rows} rows, received ${values.length}`);
  return values.map((entry, i) => normalized ? distribution(entry, `${path}[${i}]`, columns) : vector(entry, `${path}[${i}]`, columns));
}
function stateNorm(value: unknown, state: number[], path: string): number {
  const actual = number(value, path, 0);
  const expected = Math.hypot(...state);
  if (!Number.isFinite(expected) || Math.abs(actual - expected) > Math.max(1e-5, expected * 1e-4)) fail(path, 'does not match the vector Euclidean norm');
  return actual;
}

/** Validate and copy data across the trust boundary. No imported code is executed. */
export function validateTrace(value: unknown): Trace {
  const data = object(value, 'trace');
  if (data.schemaVersion !== 1) fail('schemaVersion', 'only version 1 is supported');
  if (data.provenance !== 'educational-simulation' && data.provenance !== 'imported-trace') fail('provenance', 'expected educational-simulation or imported-trace');
  if (!ARCHITECTURES.includes(data.architecture as Architecture)) fail('architecture', 'unknown architecture');
  const architecture = data.architecture as Architecture;
  const input = string(data.input, 'input', 32768);
  const rawTokens = list(data.tokens, 'tokens', 256);
  if (!rawTokens.length) fail('tokens', 'at least one token is required');
  const ids = new Set<number>();
  const tokens = rawTokens.map((entry, index) => {
    const token = object(entry, `tokens[${index}]`);
    const id = integer(token.id, `tokens[${index}].id`, 0, Number.MAX_SAFE_INTEGER);
    if (ids.has(id)) fail(`tokens[${index}].id`, 'token IDs must be unique');
    ids.add(id);
    const position = integer(token.position, `tokens[${index}].position`, index, index);
    const text = string(token.text, `tokens[${index}].text`, 128);
    if (typeof token.isClue !== 'boolean') fail(`tokens[${index}].isClue`, 'expected a boolean');
    return { id, text, position, isClue: token.isClue };
  });
  if (tokens.map((token) => token.text).join('') !== input) fail('input', 'must equal the concatenated token text');
  const vocabulary = list(data.vocabulary, 'vocabulary').map((entry, i) => string(entry, `vocabulary[${i}]`, 128));
  if (!vocabulary.length || new Set(vocabulary).size !== vocabulary.length) fail('vocabulary', 'requires distinct, nonempty token strings');
  const rawSteps = list(data.steps, 'steps', 256);
  if (rawSteps.length !== tokens.length) fail('steps', 'requires exactly one step per token');
  const firstStep = object(rawSteps[0], 'steps[0]');
  const hiddenDimension = list(firstStep.hidden, 'steps[0].hidden').length;
  const cellDimension = list(firstStep.cell, 'steps[0].cell').length;
  if (!hiddenDimension) fail('steps[0].hidden', 'hidden state cannot be empty');
  if (cellDimension !== 0 && cellDimension !== hiddenDimension) fail('steps[0].cell', 'cell dimension must be zero or equal hidden dimension');
  const steps: StepSummary[] = rawSteps.map((entry, index) => {
    const path = `steps[${index}]`;
    const step = object(entry, path);
    integer(step.index, `${path}.index`, index, index);
    const hidden = vector(step.hidden, `${path}.hidden`, hiddenDimension);
    const cell = vector(step.cell, `${path}.cell`, cellDimension);
    const rawGates = object(step.gates, `${path}.gates`);
    const gates = Object.fromEntries(['forget', 'input', 'candidate', 'output', 'update', 'reset'].map((key) =>
      [key, number(rawGates[key], `${path}.gates.${key}`, key === 'candidate' ? -1 : 0, 1)])) as unknown as Gates;
    const retainedIndices = list(step.retainedIndices, `${path}.retainedIndices`, tokens.length)
      .map((value, i) => integer(value, `${path}.retainedIndices[${i}]`, 0, architecture === 'transformer' ? tokens.length - 1 : index));
    if (new Set(retainedIndices).size !== retainedIndices.length) fail(`${path}.retainedIndices`, 'indices must be unique');
    const probabilities = list(step.probabilities, `${path}.probabilities`).map((entry, i) => {
      const probability = object(entry, `${path}.probabilities[${i}]`);
      return { token: string(probability.token, `${path}.probabilities[${i}].token`, 128),
        probability: number(probability.probability, `${path}.probabilities[${i}].probability`, 0, 1) };
    });
    if (probabilities.length !== vocabulary.length || new Set(probabilities.map((p) => p.token)).size !== vocabulary.length ||
      probabilities.some((p) => !vocabulary.includes(p.token))) fail(`${path}.probabilities`, 'must contain every vocabulary token exactly once');
    distribution(probabilities.map((p) => p.probability), `${path}.probabilities`, vocabulary.length);
    if (probabilities.some((p, i) => i > 0 && p.probability > probabilities[i - 1].probability + 1e-12)) fail(`${path}.probabilities`, 'must be sorted from highest to lowest probability');
    return { index, hidden, cell, gates, retainedIndices, probabilities,
      hiddenNorm: stateNorm(step.hiddenNorm, hidden, `${path}.hiddenNorm`),
      cellNorm: stateNorm(step.cellNorm, cell, `${path}.cellNorm`),
      logits: vector(step.logits, `${path}.logits`, vocabulary.length),
      gradients: vector(step.gradients, `${path}.gradients`, index + 1, 0),
      clueStrength: number(step.clueStrength, `${path}.clueStrength`, 0, 1) };
  });
  const rawAttention = list(data.attention, 'attention', 256);
  if (architecture === 'transformer' && !rawAttention.length) fail('attention', 'transformer traces require an attention matrix');
  const attention = matrix(rawAttention, 'attention', rawAttention.length ? tokens.length : 0, tokens.length, true);
  const encoderTokens = list(data.encoderTokens, 'encoderTokens', 256).map((token, i) => string(token, `encoderTokens[${i}]`, 128));
  const decoderTokens = list(data.decoderTokens, 'decoderTokens', 256).map((token, i) => string(token, `decoderTokens[${i}]`, 128));
  if (Boolean(encoderTokens.length) !== Boolean(decoderTokens.length)) fail('decoderTokens', 'encoder and decoder tokens must both be present or both empty');
  if (architecture === 'attention' && !encoderTokens.length) fail('encoderTokens', 'attention-bridge traces require encoder and decoder tokens');
  const rawEncoderStates = list(data.encoderStates, 'encoderStates', 256);
  const encoderDimension = rawEncoderStates.length ? list(rawEncoderStates[0], 'encoderStates[0]').length : 0;
  if (encoderTokens.length && !encoderDimension) fail('encoderStates', 'requires a nonempty state for each source token');
  const encoderStates = matrix(rawEncoderStates, 'encoderStates', encoderTokens.length, encoderDimension, false);
  const alignment = matrix(data.alignment, 'alignment', decoderTokens.length, encoderTokens.length, true);
  const metadata = object(data.metadata, 'metadata');
  return { schemaVersion: 1, provenance: data.provenance, architecture, input, tokens, vocabulary, steps, attention,
    encoderTokens, decoderTokens, encoderStates, alignment,
    metadata: { label: string(metadata.label, 'metadata.label', 256),
      description: string(metadata.description, 'metadata.description', 5000),
      ...(metadata.source === undefined ? {} : { source: string(metadata.source, 'metadata.source', 512) }) } };
}

export function loadTraceJson(json: string, sourceLabel = 'Imported JSON trace'): Trace {
  if (new TextEncoder().encode(json).byteLength > MAX_JSON_BYTES) throw new Error('Trace JSON exceeds the 2 MB import limit.');
  let value: unknown;
  try { value = JSON.parse(json); }
  catch { throw new Error('Could not parse trace JSON. Check that the file contains valid JSON.'); }
  const trace = validateTrace(value);
  const source = string(sourceLabel, 'sourceLabel', 512);
  trace.provenance = 'imported-trace';
  trace.metadata = { label: 'Imported trace', source,
    description: `Imported data from ${source}. Importing does not verify that values came from a trained checkpoint. ${trace.metadata.description}`.slice(0, 5000) };
  return trace;
}

export class EducationalTraceAdapter implements TraceAdapter {
  async load(input: string, architecture: Architecture, config: SimulationConfig): Promise<Trace> {
    return buildTrace(input, architecture, config);
  }
}

/** The same asynchronous interface can later fetch exported PyTorch traces. */
export class JsonTraceAdapter implements TraceAdapter {
  private readonly trace: Trace;
  constructor(json: string, sourceLabel = 'Imported JSON trace') {
    this.trace = loadTraceJson(json, sourceLabel);
  }
  readonly load: (input: string, architecture: Architecture, config: SimulationConfig) => Promise<Trace> = async () =>
    structuredClone(this.trace);
}
