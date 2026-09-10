import assert from 'node:assert/strict';
import test from 'node:test';
import { buildTrace, DEFAULT_CONFIG } from '../simulation/engine';
import type { Architecture, SimulationConfig, Trace } from '../simulation/types';
import { buildExplorerModel } from './models';
import type { ExplorerModel, ExplorerPart } from './types';

const architectures: Architecture[] = ['mlp', 'rnn', 'lstm', 'attention', 'transformer'];
const sentence = 'the cat that chased the mouse was ';
function model(architecture: Architecture, changes: Partial<SimulationConfig> = {}, step?: number) {
  const config = { ...DEFAULT_CONFIG, ...changes };
  const trace = buildTrace(sentence, architecture, config);
  return { trace, config, model: buildExplorerModel({ trace, config, step: step ?? trace.steps.length - 1, decoderStep: 4 }) };
}
const part = (model: ExplorerModel, id: string) => {
  const result = model.parts.find((item) => item.id === id);
  assert.ok(result, `Missing part ${id}`);
  return result;
};
const overlaps = (a: ExplorerPart, b: ExplorerPart) => [0, 1, 2].every((axis) =>
  Math.abs(a.explodedPosition[axis] - b.explodedPosition[axis]) < (a.scale[axis] + b.scale[axis]) / 2 - 0.005);

test('all five museum models have stable semantic IDs, valid paths and bounded geometry', () => {
  for (const architecture of architectures) {
    for (const step of [0, 4, sentence.length - 1]) {
      const value = model(architecture, {}, step).model;
      assert.deepEqual(value, model(architecture, {}, step).model, `${architecture} must be deterministic`);
      const ids = new Set(value.parts.map((item) => item.id));
      assert.equal(ids.size, value.parts.length, 'IDs must be unique');
      assert.ok(value.parts.length > 3 && value.parts.length <= 50);
      assert.ok(value.flow.length > 2);
      for (const id of value.flow) assert.ok(ids.has(id), `Flow endpoint ${id} exists`);
      assert.equal(new Set(value.links.map((link) => link.id)).size, value.links.length);
      for (const link of value.links) {
        assert.ok(ids.has(link.from) && ids.has(link.to), `Valid ${link.id}`);
        if (link.weight !== undefined) assert.ok(Number.isFinite(link.weight) && link.weight >= 0);
      }
      for (const item of value.parts) {
        assert.ok(item.purpose.length > 20 && item.why.length > 20);
        if (item.parentId) assert.ok(ids.has(item.parentId));
        if (item.value !== undefined) assert.ok(Number.isFinite(item.value));
        for (let axis = 0; axis < 3; axis++) {
          assert.ok(Number.isFinite(item.position[axis]) && Number.isFinite(item.explodedPosition[axis]));
          assert.ok(item.scale[axis] > 0);
          assert.ok(Math.abs(item.position[axis]) + item.scale[axis] / 2 <= [3.1, 2.8, 1.4][axis], `${architecture} ${item.id} assembled axis ${axis}`);
          assert.ok(Math.abs(item.explodedPosition[axis]) + item.scale[axis] / 2 <= [5, 3, 1.4][axis], `${architecture} ${item.id} exploded axis ${axis}`);
        }
      }
    }
  }
});

test('exploded primary pieces occupy separate volumes in every architecture and GRU variant', () => {
  for (const architecture of architectures) {
    for (const variant of ['lstm', 'gru'] as const) {
      const value = model(architecture, { gatedVariant: variant }).model;
      const primary = value.parts.filter((item) => !item.detailOnly);
      for (let left = 0; left < primary.length; left++) {
        for (let right = left + 1; right < primary.length; right++) {
          assert.equal(overlaps(primary[left], primary[right]), false, `${architecture}/${variant}: ${primary[left].id} overlaps ${primary[right].id}`);
        }
      }
    }
  }
});

test('MLP retains the actual trace window and ghosts cannot feed the dense path', () => {
  for (const contextLength of [3, 8]) {
    const { trace, model: value } = model('mlp', { contextLength });
    const retained = trace.steps.at(-1)!.retainedIndices;
    const tokens = value.parts.filter((item) => item.id.startsWith('token-') && !item.ghost);
    assert.deepEqual(tokens.map((item) => item.tokenIndex), retained);
    for (const ghost of value.parts.filter((item) => item.ghost)) {
      assert.ok(!retained.includes(ghost.tokenIndex!));
      assert.equal(value.links.some((link) => link.from === ghost.id), false);
    }
    assert.ok(tokens.every((item) => value.links.some((link) => link.from === item.id && link.to === `embedding-${item.tokenIndex}`)));
  }
});

test('RNN unfolded states preserve the real sequence indices and exact recorded norms', () => {
  const { trace, model: value } = model('rnn');
  const states = value.parts.filter((item) => item.id.startsWith('state-'));
  assert.equal(states.length, 7);
  for (const [slot, state] of states.entries()) {
    assert.equal(state.value, trace.steps[state.tokenIndex!].hiddenNorm);
    if (slot) assert.ok(value.links.some((link) => link.from === states[slot - 1].id && link.to === state.id && link.kind === 'memory'));
  }
  assert.equal(states.at(-1)?.tokenIndex, trace.steps.length - 1);
});

test('LSTM and GRU expose only their own gates, with signed candidate and exact override values', () => {
  const lstm = model('lstm', { gateOverrides: { forget: 0.97, input: 0.1, output: 0.3, candidate: -0.6 } });
  assert.deepEqual(lstm.model.parts.filter((item) => item.category === 'gates').map((item) => item.id), ['gate-forget', 'gate-input', 'gate-candidate', 'gate-output']);
  for (const gate of ['forget', 'input', 'candidate', 'output'] as const) assert.equal(part(lstm.model, `gate-${gate}`).value, lstm.trace.steps.at(-1)!.gates[gate]);
  assert.ok(part(lstm.model, 'gate-candidate').value! < 0);
  assert.ok(part(lstm.model, 'cell-rail'));
  const gru = model('lstm', { gatedVariant: 'gru', gateOverrides: { reset: 0.4, update: 0.85 } });
  assert.deepEqual(gru.model.parts.filter((item) => item.category === 'gates').map((item) => item.id), ['gate-reset', 'gate-update']);
  assert.ok(!gru.model.parts.some((item) => item.id === 'cell-rail'));
  assert.equal(part(gru.model, 'gate-update').value, gru.trace.steps.at(-1)!.gates.update);
  assert.match(part(gru.model, 'gate-update').math!, /z controls NEW content/);
});

test('attention strands use the selected decoder row without renormalizing the visible source window', () => {
  const { trace, config } = model('attention', { sourceLength: 16 });
  for (const decoderStep of [0, 7, 15]) {
    const value = buildExplorerModel({ trace, config, step: 1, decoderStep });
    const links = value.links.filter((link) => link.kind === 'attention');
    assert.equal(links.length, 8);
    for (const link of links) {
      const source = part(value, link.from).tokenIndex!;
      assert.equal(link.weight, trace.alignment[decoderStep][source]);
    }
    assert.equal(part(value, 'decoder-state').decoderIndex, decoderStep);
    assert.ok(part(value, `decoder-${decoderStep}`));
    assert.match(value.description, /not|outside/);
  }
  const disabled = model('attention', { attentionEnabled: false }).model;
  assert.ok(part(disabled, 'compression-bottleneck'));
  assert.ok(!disabled.parts.some((item) => item.id === 'attention-context'));
  assert.equal(disabled.links.filter((link) => link.kind === 'attention').length, 0);
});

function longTrace(architecture: 'attention' | 'transformer'): Trace {
  const trace = buildTrace('abc', architecture, DEFAULT_CONFIG);
  trace.provenance = 'imported-trace';
  trace.tokens = Array.from({ length: 256 }, (_, index) => ({ id: index, text: `t${index}`, position: index, isClue: index === 0 }));
  trace.steps = trace.tokens.map((_, index) => ({ ...trace.steps[0], index }));
  const matrix = Array.from({ length: 256 }, (_, query) => Array.from({ length: 256 }, (_, key) => key === (query + 55) % 256 ? 1 : 0));
  if (architecture === 'transformer') trace.attention = matrix;
  else {
    trace.encoderTokens = trace.tokens.map((token) => `source${token.position}`);
    trace.decoderTokens = trace.tokens.map((token) => `target${token.position}`);
    trace.encoderStates = trace.tokens.map(() => [0.2, 0.4]);
    trace.alignment = matrix;
  }
  return trace;
}

test('long imported traces keep global source/query indices beyond the first 16 positions', () => {
  const trace = longTrace('attention');
  const value = buildExplorerModel({ trace, config: DEFAULT_CONFIG, step: 255, decoderStep: 200 });
  assert.equal(part(value, 'decoder-200').label, 'target200');
  assert.equal(part(value, 'source-255').label, 'source255');
  assert.equal(value.links.find((link) => link.from === 'source-state-255' && link.kind === 'attention')?.weight, 1);
  const transformerTrace = longTrace('transformer');
  const tf = buildExplorerModel({ trace: transformerTrace, config: DEFAULT_CONFIG, step: 200, decoderStep: 0 });
  assert.equal(part(tf, 'token-200').tokenIndex, 200);
  assert.equal(part(tf, 'token-255').tokenIndex, 255);
  assert.equal(tf.links.find((link) => link.from === 'token-255' && link.kind === 'attention')?.weight, 1);
  assert.match(part(tf, 'attention-mix').purpose, /does not record separate per-head/);
});

test('Transformer head internals are hierarchical explanations, with exact weights only from the current trace', () => {
  const { trace, config, model: value } = model('transformer');
  for (let head = 0; head < 4; head++) {
    assert.equal(value.parts.filter((item) => item.parentId === `head-${head}`).length, 4);
    for (const suffix of ['q', 'k', 'v', 'weights']) {
      const child = part(value, `head-${head}-${suffix}`);
      assert.equal(child.detailOnly, true);
      assert.equal(child.value, undefined);
      assert.match(child.purpose, /no Q\/K\/V tensor values are inferred/);
    }
  }
  for (const link of value.links.filter((link) => link.weight !== undefined)) {
    assert.equal(link.to, `head-${config.head}`);
    assert.equal(link.weight, trace.attention.at(-1)![part(value, link.from).tokenIndex!]);
  }
  assert.match(part(value, 'feed-forward').purpose, /does not execute or export a separate FFN/);
  assert.match(part(value, 'residual-stream').math!, /simplified/);
});

test('selection bounds and an unsorted imported probability list do not invent data', () => {
  const { trace, config } = model('mlp');
  trace.provenance = 'imported-trace';
  trace.steps[0].probabilities = [{ token: 'a', probability: 0.2 }, { token: 'b', probability: 0.8 }];
  const value = buildExplorerModel({ trace, config, step: -50, decoderStep: 900 });
  assert.equal(part(value, 'output').value, 0.8);
  assert.match(part(value, 'output').purpose, /“b”/);
  const tf = model('transformer');
  const bounded = buildExplorerModel({ trace: tf.trace, config: { ...tf.config, head: 99 }, step: 9999, decoderStep: 0 });
  assert.ok(part(bounded, `token-${tf.trace.tokens.length - 1}`));
  assert.ok(bounded.links.every((link) => bounded.parts.some((item) => item.id === link.to)));
});

test('imported models are independent of stale simulation settings for all five architectures', () => {
  const before: SimulationConfig = { ...DEFAULT_CONFIG, head: 0, layer: 0, attentionEnabled: false, aggregateHeads: false, positionalEncoding: false, causalMask: false, gatedVariant: 'lstm' };
  const after: SimulationConfig = { ...DEFAULT_CONFIG, head: 3, layer: 3, attentionEnabled: true, aggregateHeads: true, positionalEncoding: true, causalMask: true, gatedVariant: 'gru', gateOverrides: { update: 0.7 } };
  for (const architecture of architectures) {
    const trace = buildTrace(sentence, architecture, after);
    trace.provenance = 'imported-trace';
    const a = buildExplorerModel({ trace, config: before, step: 12, decoderStep: 4 });
    const b = buildExplorerModel({ trace, config: after, step: 12, decoderStep: 4 });
    assert.deepEqual(a, b, `${architecture}: unrecorded settings cannot change the imported visualization`);
  }
});

test('an imported attention recording displays supplied alignment even when the simulation bridge was closed', () => {
  const trace = buildTrace(sentence, 'attention', { ...DEFAULT_CONFIG, attentionEnabled: true });
  trace.provenance = 'imported-trace';
  const value = buildExplorerModel({ trace, config: { ...DEFAULT_CONFIG, attentionEnabled: false }, step: 3, decoderStep: 4 });
  assert.equal(value.parts.some((item) => item.id === 'compression-bottleneck'), false);
  assert.equal(part(value, 'attention-context').label, 'Recorded alignment');
  assert.match(part(value, 'attention-context').purpose, /does not say whether the original model used/);
  const weighted = value.links.filter((link) => link.weight !== undefined);
  assert.equal(weighted.length, trace.encoderTokens.length);
  for (const link of weighted) assert.equal(link.weight, trace.alignment[4][part(value, link.from).tokenIndex!]);
});

test('imported gated memory shows supplied arrays and scalars without guessing GRU or LSTM from placeholder zeros', () => {
  const trace = buildTrace(sentence, 'lstm', { ...DEFAULT_CONFIG, gatedVariant: 'gru' });
  trace.provenance = 'imported-trace';
  const value = buildExplorerModel({ trace, config: { ...DEFAULT_CONFIG, gatedVariant: 'lstm' }, step: 6, decoderStep: 0 });
  assert.match(value.description, /does not identify the LSTM\/GRU variant/);
  const gates = value.parts.filter((item) => item.category === 'gates');
  assert.equal(gates.length, 6, 'All schema fields remain inspectable without guessing which ones were used');
  for (const gate of gates) {
    const name = gate.id.replace('gate-', '') as keyof typeof trace.steps[6]['gates'];
    assert.equal(gate.value, trace.steps[6].gates[name]);
    assert.match(gate.purpose, /does not establish which gates were used/);
  }
  assert.equal(part(value, 'cell-rail').value, 0, 'Placeholder zeros remain exactly recorded');
  assert.match(part(value, 'cell-rail').purpose, /placeholder zeros/);
  for (const step of trace.steps) step.cell = [];
  const withoutCell = buildExplorerModel({ trace, config: DEFAULT_CONFIG, step: 6, decoderStep: 0 });
  assert.equal(withoutCell.parts.some((item) => item.id.startsWith('cell-')), false, 'Absent cell arrays must not be invented');
  const primary = value.parts.filter((item) => !item.detailOnly);
  for (let left = 0; left < primary.length; left++) for (let right = left + 1; right < primary.length; right++) {
    assert.equal(overlaps(primary[left], primary[right]), false, `${primary[left].id} overlaps ${primary[right].id}`);
  }
});

test('imported Transformer matrices are not attributed to a stale head, layer, mask or position setting', () => {
  const trace = longTrace('transformer');
  const value = buildExplorerModel({ trace, config: { ...DEFAULT_CONFIG, head: 3, positionalEncoding: false, causalMask: false }, step: 200, decoderStep: 0 });
  const position = part(value, 'position-encoding');
  assert.equal(position.ghost, false);
  assert.match(position.label, /unrecorded/);
  assert.match(position.purpose, /enabled state are unknown/);
  assert.equal(part(value, 'attention-mix').label, 'Recorded attention matrix');
  for (const link of value.links.filter((item) => item.weight !== undefined)) assert.equal(link.to, 'attention-mix');
  for (let head = 0; head < 4; head++) {
    assert.equal(part(value, `head-${head}`).active, false);
    assert.match(part(value, `head-${head}-weights`).purpose, /not attributed to this schematic head/);
  }
});
