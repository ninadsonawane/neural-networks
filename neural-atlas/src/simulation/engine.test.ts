import assert from 'node:assert/strict';
import test from 'node:test';
import { EducationalTraceAdapter, JsonTraceAdapter, loadTraceJson, validateTrace } from './adapter';
import { buildTrace, DEFAULT_CONFIG, displayToken } from './engine';
import type { Architecture, SimulationConfig, Trace } from './types';

const architectures: Architecture[] = ['mlp', 'rnn', 'lstm', 'attention', 'transformer'];
const sentence = 'the cat that chased the mouse was ';
const trace = (architecture: Architecture, config: Partial<SimulationConfig> = {}, input = sentence) =>
  buildTrace(input, architecture, { ...DEFAULT_CONFIG, ...config });
const last = (value: Trace) => value.steps[value.steps.length - 1];
const near = (actual: number, expected: number, tolerance = 1e-10) =>
  assert.ok(Math.abs(actual - expected) < tolerance, `${actual} is not approximately ${expected}`);

test('every architecture produces deterministic, finite traces accepted by the import schema', () => {
  for (const architecture of architectures) {
    const first = trace(architecture);
    assert.deepEqual(first, trace(architecture));
    assert.deepEqual(validateTrace(first), first);
    assert.equal(first.tokens.length, Array.from(sentence).length);
    assert.equal(first.vocabulary.length, 28);
    for (const step of first.steps) {
      assert.equal(step.hidden.length, 8);
      assert.ok(step.hidden.every(Number.isFinite));
      assert.equal(step.probabilities.length, first.vocabulary.length);
      near(step.probabilities.reduce((total, entry) => total + entry.probability, 0), 1);
      assert.ok(step.probabilities.every((entry, index, entries) => index === 0 || entries[index - 1].probability >= entry.probability));
    }
  }
});

test('Unicode tokenization preserves code points, caps input, and locates the first meaningful clue', () => {
  const result = trace('mlp', {}, ' 🙂ह' + '🌒'.repeat(90));
  assert.equal(result.tokens.length, 80);
  assert.deepEqual(result.tokens.slice(0, 3).map((token) => token.text), [' ', '🙂', 'ह']);
  assert.equal(result.tokens.find((token) => token.isClue)?.position, 2);
  assert.equal(trace('mlp', {}, '').tokens.length, 1);
  assert.equal(displayToken(' '), '␣');
  assert.equal(displayToken('\n'), '↵');
});

test('MLP output has no information pathway from characters outside the fixed window', () => {
  for (const contextLength of [3, 5, 8]) {
    const suffix = 'same end'.slice(-contextLength);
    const a = last(trace('mlp', { contextLength }, `cat ${suffix}`));
    const b = last(trace('mlp', { contextLength }, `a much longer dog prefix ${suffix}`));
    assert.deepEqual(a.hidden, b.hidden);
    assert.deepEqual(a.logits, b.logits);
    assert.deepEqual(a.probabilities, b.probabilities);
    assert.equal(a.clueStrength, 0);
    assert.notDeepEqual(a.probabilities, last(trace('mlp', { contextLength }, `cat ${suffix.slice(0, -1)}z`)).probabilities);
  }
});

test('temperature changes distribution sharpness without inventing new logits', () => {
  const cold = last(trace('mlp', { temperature: 0.2 }));
  const hot = last(trace('mlp', { temperature: 2 }));
  assert.deepEqual(cold.logits, hot.logits);
  assert.ok(cold.probabilities[0].probability > hot.probabilities[0].probability);
});

test('RNN carries a prefix in state and exposes the stated r^distance sensitivity proxy', () => {
  const cat = trace('rnn', { recurrentMultiplier: 0.8 }, 'cat xyz');
  const dog = trace('rnn', { recurrentMultiplier: 0.8 }, 'dog xyz');
  assert.notDeepEqual(last(cat).hidden, last(dog).hidden);
  near(last(cat).gradients[0], 0.8 ** 6);
  near(last(cat).gradients[6], 1);
  assert.ok(last(trace('rnn', { recurrentMultiplier: 1.15 })).gradients[0] > 1);
  assert.ok(last(trace('rnn', { recurrentMultiplier: 0.5 })).gradients[0] < 1e-7);
  assert.match(cat.metadata.description, /proxy, not the exact derivative/);
});

test('LSTM scalar gate overrides obey the cell and hidden equations', () => {
  const result = trace('lstm', { gateOverrides: { forget: 0.5, input: 0.25, candidate: 0.8, output: 0.75 } }, 'ab');
  for (const value of result.steps[0].cell) near(value, 0.2);
  for (const value of result.steps[1].cell) near(value, 0.3);
  for (const value of result.steps[1].hidden) near(value, 0.75 * Math.tanh(0.3));
});

test('GRU update gate controls new-state write and zero update preserves the state', () => {
  const closed = trace('lstm', { gatedVariant: 'gru', gateOverrides: { update: 0 } });
  assert.ok(closed.steps.every((step) => step.hidden.every((value) => value === 0)));
  const open = trace('lstm', { gatedVariant: 'gru', gateOverrides: { update: 1, candidate: 0.6 } });
  assert.ok(open.steps.every((step) => step.hidden.every((value) => Math.abs(value - 0.6) < 1e-10)));
  assert.ok(open.steps.every((step) => step.cellNorm === 0));
});

test('pinning a clue meaningfully increases gate-path retention in LSTM and GRU', () => {
  for (const gatedVariant of ['lstm', 'gru'] as const) {
    const automatic = trace('lstm', { gatedVariant, pinClue: false });
    const pinned = trace('lstm', { gatedVariant, pinClue: true });
    assert.ok(last(pinned).clueStrength > 0.5);
    assert.ok(last(pinned).clueStrength > last(automatic).clueStrength * 100);
    for (const step of pinned.steps) {
      for (const gate of ['forget', 'input', 'output', 'update', 'reset'] as const) {
        assert.ok(step.gates[gate] >= 0 && step.gates[gate] <= 1);
      }
      assert.ok(step.gates.candidate >= -1 && step.gates.candidate <= 1);
    }
  }
});

test('causal masking has exactly zero probability for future keys at every head and layer', () => {
  for (const head of [0, 1, 2, 3]) {
    for (const aggregateHeads of [false, true]) {
      const result = trace('transformer', { head, aggregateHeads, layer: 3 });
      result.attention.forEach((row, query) => {
        near(row.reduce((sum, value) => sum + value, 0), 1);
        row.forEach((value, key) => { if (key > query) assert.equal(value, 0); });
      });
    }
  }
});

test('a causal Transformer cannot change a prefix state when a suffix is appended', () => {
  const prefix = trace('transformer', {}, 'the cat');
  const extended = trace('transformer', {}, 'the cat ran away');
  prefix.steps.forEach((step, index) => {
    assert.deepEqual(step.hidden, extended.steps[index].hidden);
    assert.deepEqual(step.probabilities, extended.steps[index].probabilities);
  });
  const unmasked = trace('transformer', { causalMask: false }, 'the cat');
  assert.ok(unmasked.attention[0].slice(1).every((value) => value > 0));
});

test('head, position, layer, and aggregation controls materially change attention or states', () => {
  const base = trace('transformer', { head: 0 });
  assert.notDeepEqual(base.attention, trace('transformer', { head: 1 }).attention);
  assert.notDeepEqual(base.attention, trace('transformer', { head: 0, positionalEncoding: false }).attention);
  assert.notDeepEqual(base.attention, trace('transformer', { head: 0, aggregateHeads: true }).attention);
  assert.notDeepEqual(last(base).hidden, last(trace('transformer', { head: 0, layer: 3 })).hidden);
  const distant = trace('transformer', { head: 2 }, 'x the river was calm');
  const weights = distant.attention[distant.attention.length - 1];
  assert.equal(weights.indexOf(Math.max(...weights)), 0);
});

test('attention bridge uses separate conceptual word tokens and normalized source alignment', () => {
  for (const sourceLength of [4, 8, 16]) {
    const enabled = trace('attention', { sourceLength });
    const disabled = trace('attention', { sourceLength, attentionEnabled: false });
    assert.equal(enabled.encoderTokens.length, sourceLength);
    assert.equal(enabled.decoderTokens.length, sourceLength);
    assert.equal(enabled.encoderStates.length, sourceLength);
    enabled.alignment.forEach((row) => near(row.reduce((a, b) => a + b, 0), 1));
    assert.ok(disabled.alignment.every((row) => row.every((weight) => Math.abs(weight - 1 / sourceLength) < 1e-10)));
    assert.notDeepEqual(enabled.alignment, disabled.alignment);
    assert.equal(enabled.tokens[0].text.length, 1);
    assert.match(enabled.metadata.description, /word-level/);
  }
});

test('exported simulations roundtrip with imported provenance and explicit source labels', async () => {
  for (const architecture of architectures) {
    const original = trace(architecture);
    const json = JSON.stringify(original);
    const imported = loadTraceJson(json, 'notebook-export.json');
    assert.equal(imported.provenance, 'imported-trace');
    assert.equal(imported.metadata.source, 'notebook-export.json');
    assert.match(imported.metadata.description, /does not verify/);
    assert.deepEqual(imported.steps, original.steps);
    assert.deepEqual(validateTrace(imported), imported);
    const adapter = new JsonTraceAdapter(json, 'notebook-export.json');
    const first = await adapter.load(sentence, architecture, DEFAULT_CONFIG);
    first.steps[0].hidden[0] = 42;
    assert.notEqual((await adapter.load(sentence, architecture, DEFAULT_CONFIG)).steps[0].hidden[0], 42);
  }
  const educational = new EducationalTraceAdapter();
  assert.deepEqual(await educational.load(sentence, 'mlp', DEFAULT_CONFIG), trace('mlp'));
});

test('JSON boundary rejects malformed JSON, unsupported versions, unsafe numbers, and incoherent shapes', () => {
  assert.throws(() => loadTraceJson('{'), /valid JSON/);
  assert.throws(() => loadTraceJson(' '.repeat(2 * 1024 * 1024 + 1)), /2 MB/);
  const mutate = (fn: (result: Trace) => void) => { const result = trace('transformer'); fn(result); return result; };
  assert.throws(() => validateTrace({ ...trace('mlp'), schemaVersion: 2 }), /version 1/);
  assert.throws(() => validateTrace(mutate((t) => { t.steps[0].hidden[0] = Infinity; })), /finite number/);
  assert.throws(() => validateTrace(mutate((t) => { t.steps[1].hidden.pop(); })), /expected 8 entries/);
  assert.throws(() => validateTrace(mutate((t) => { t.steps[1].hiddenNorm = 900; })), /Euclidean norm/);
  assert.throws(() => validateTrace(mutate((t) => { t.steps[0].probabilities[0].probability = 0.9; })), /sum to 1/);
  assert.throws(() => validateTrace(mutate((t) => { t.steps[0].probabilities[0].token = 'unknown'; })), /every vocabulary token/);
  assert.throws(() => validateTrace(mutate((t) => { t.attention[0][0] = 0.5; })), /sum to 1/);
  assert.throws(() => validateTrace(mutate((t) => { t.tokens[1].position = 7; })), /finite number between 1 and 1/);
  assert.throws(() => validateTrace(mutate((t) => { t.tokens[1].id = 0; })), /IDs must be unique/);
  assert.throws(() => validateTrace(mutate((t) => { t.steps[0].gates.forget = 2; })), /between 0 and 1/);
  assert.throws(() => validateTrace(mutate((t) => { t.steps[0].gradients = []; })), /expected 1 entries/);
  assert.throws(() => validateTrace({ ...trace('attention'), alignment: [[1]] }), /expected 8 rows/);
  assert.throws(() => validateTrace({ ...trace('mlp'), input: 'mismatched' }), /concatenated token text/);
});
