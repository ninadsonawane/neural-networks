import { displayToken } from '../simulation/engine';
import type { Gates, Trace } from '../simulation/types';
import type { ExplorerLink, ExplorerModel, ExplorerPart, ModelBuildOptions, Vec3 } from './types';

/** Geometry is an original explanatory schematic. Numbers always come from the trace. */
const positions = (count: number, width: number) => Array.from({ length: count }, (_, i) =>
  count === 1 ? 0 : (i / (count - 1) - 0.5) * width);
const clampIndex = (index: number, count: number) => Math.max(0, Math.min(count - 1, Math.floor(Number.isFinite(index) ? index : 0)));
const sequence = (start: number, end: number) => Array.from({ length: Math.max(0, end - start) }, (_, i) => start + i);
const magnitude = (values: number[] | undefined) => values?.length ? Math.hypot(...values) : undefined;
const recorded = (trace: Trace) => trace.provenance === 'imported-trace' ? 'Imported trace' : 'Educational simulation';

type PartSpec = Omit<ExplorerPart, 'position' | 'explodedPosition' | 'scale'>;

class Assembly {
  parts: ExplorerPart[] = [];
  links: ExplorerLink[] = [];
  flow: string[] = [];
  constructor(readonly options: ModelBuildOptions) {}
  part(spec: PartSpec, position: Vec3, explodedPosition: Vec3, scale: Vec3) {
    this.parts.push({ ...spec, position, explodedPosition, scale });
    return spec.id;
  }
  link(from: string, to: string, kind: ExplorerLink['kind'] = 'flow', weight?: number) {
    this.links.push({ id: `${kind}:${from}:${to}`, from, to, kind, ...(weight === undefined ? {} : { weight }) });
  }
  token(index: number, position: Vec3, explodedPosition: Vec3, ghost = false, scale: Vec3 = [0.47, 0.42, 0.16]) {
    const { trace, step } = this.options;
    const token = trace.tokens[index];
    return this.part({ id: `token-${index}`, label: displayToken(token.text), category: 'tokens', kind: 'card', tokenIndex: index,
      active: index === step, ghost,
      purpose: `Input token ${index + 1}: “${displayToken(token.text)}”. ${trace.provenance === 'educational-simulation' ? 'This teaching model tokenizes Unicode characters.' : 'Its text and position are recorded by the imported trace.'}`,
      why: ghost ? 'This earlier token lies outside the visible fixed window. It has no direct input path into this prediction.' : 'A model needs discrete input symbols before it can work with numerical representations.',
    }, position, explodedPosition, scale);
  }
  output(position: Vec3, explodedPosition: Vec3) {
    const { trace, step } = this.options;
    const best = trace.steps[step]?.probabilities.reduce<(typeof trace.steps)[number]['probabilities'][number] | undefined>((leading, item) => !leading || item.probability > leading.probability ? item : leading, undefined);
    return this.part({ id: 'output', label: 'Next-token output', category: 'output', kind: 'slab',
      value: best?.probability, active: true,
      purpose: best ? `${recorded(trace)}: the leading recorded candidate is “${displayToken(best.token)}” at ${(100 * best.probability).toFixed(1)}%. Open probabilities to inspect the full distribution.` : 'The readout maps the current representation to possible next tokens.',
      why: 'A representation must become a distribution over possible continuations before a token can be chosen.',
      math: 'p(token | context) = softmax(logits / temperature). The shape is a schematic readout, not a literal layer tensor.',
    }, position, explodedPosition, [1.7, 0.35, 0.92]);
  }
  model(description: string): ExplorerModel {
    return { architecture: this.options.trace.architecture, parts: this.parts, links: this.links, flow: this.flow, description };
  }
}

function fixedWindow(options: ModelBuildOptions): ExplorerModel {
  const a = new Assembly(options);
  const { trace, step } = options;
  const recordedWindow = trace.steps[step].retainedIndices.filter((index) => index >= 0 && index <= step);
  const retained = recordedWindow.slice(-8);
  const start = retained[0] ?? step;
  const ghosts = sequence(0, start).filter((index) => !recordedWindow.includes(index)).slice(-2);
  const xs = positions(retained.length, Math.min(3.6, Math.max(0, retained.length - 1) * 0.53));
  const ex = positions(retained.length, Math.min(7.6, Math.max(0, retained.length - 1) * 0.9));
  retained.forEach((index, slot) => {
    const tokenId = a.token(index, [xs[slot], -1.7, 0.3], [ex[slot], -2.45, 0.35]);
    const embeddingId = a.part({ id: `embedding-${index}`, label: `Embedding · ${index + 1}`, category: 'embeddings', kind: 'slab', tokenIndex: index,
      purpose: `An explanatory wafer for token ${index + 1}'s numerical representation. Embedding coordinates are not exported by this trace.`,
      why: 'Numerical vectors let a shared network process token identity. The window preserves their relative order.',
      math: 'eᵢ = E[tokenᵢ]. Window-relative position affects the teaching model’s fixed projection.',
    }, [xs[slot], -1.11, 0.18], [ex[slot], -1.63, 0.25], [0.43, 0.2, 0.62]);
    a.link(tokenId, embeddingId);
    a.link(embeddingId, 'window-join');
  });
  ghosts.forEach((index, slot) => a.token(index, [-0.3 + slot * 0.6, -1.7, -0.72], [-4.45, -1.35 + slot * 0.7, -0.25], true));
  a.part({ id: 'fixed-window', label: `Fixed window · ${recordedWindow.length} tokens`, category: 'memory', kind: 'slab',
    purpose: `This prediction has ${recordedWindow.length} recorded retained input positions. ${recordedWindow.length > retained.length ? `Only the last ${retained.length} are drawn.` : 'Cards behind the window are no longer inputs.'}`,
    why: 'A fixed-width input is easy to compute, but cannot directly use anything outside its window. Recurrent state is the next idea.',
  }, [0, -1.44, -0.16], [0, -1.13, -0.25], [Math.max(1.6, xs.length > 1 ? xs[xs.length - 1] - xs[0] + 0.6 : 1.6), 0.1, 0.2]);
  a.part({ id: 'window-join', label: 'Ordered context', category: 'embeddings', kind: 'bundle',
    purpose: 'The visible window feeds a single ordered representation into the dense computation.',
    why: 'Combining the current window gives the network a fixed-size input. It does not create persistent memory.',
    math: 'x = [eₜ₋ₙ₊₁; …; eₜ]. This is the standard fixed-window idea; the teaching engine uses a fixed position-weighted projection.',
  }, [0, -0.62, 0.12], [0, -0.76, 0], [2.1, 0.32, 0.88]);
  a.part({ id: 'dense-projection', label: 'Dense projection', category: 'memory', kind: 'slab',
    purpose: 'Mix information from the retained positions using fixed illustrative weights.',
    why: 'Each output feature can depend on every input in the window. There is still no path from an earlier, excluded token.',
    math: 'z = Wx + b (schematic). The trace does not contain a trained dense-layer weight matrix.',
  }, [0, -0.12, 0.08], [0, 0.08, 0.04], [2.3, 0.37, 1.04]);
  a.part({ id: 'dense-state', label: 'Nonlinear state', category: 'memory', kind: 'slab', value: trace.steps[step].hiddenNorm,
    purpose: `${recorded(trace)} hidden-state norm: ${trace.steps[step].hiddenNorm.toFixed(3)}. This is the current window’s representation.`,
    why: 'A nonlinear transformation can combine features more expressively than a sequence of linear maps.',
    math: 'h = tanh(z). This number is ‖h‖₂, not confidence.',
  }, [0, 0.39, 0.08], [0, 1.02, 0.06], [2.04, 0.36, 0.92]);
  a.output([0, 0.94, 0.08], [0, 2.13, 0.08]);
  a.link('fixed-window', 'window-join');
  a.link('window-join', 'dense-projection');
  a.link('dense-projection', 'dense-state');
  a.link('dense-state', 'output');
  a.flow = [...retained.map((index) => `token-${index}`), 'window-join', 'dense-projection', 'dense-state', 'output'];
  return a.model('A fixed window feeds one compact stack. Earlier cards fall behind its memory horizon; recurrent state was invented to carry information onward.');
}

function recurrent(options: ModelBuildOptions): ExplorerModel {
  const a = new Assembly(options);
  const { trace, step } = options;
  const indices = sequence(Math.max(0, step - 6), step + 1);
  const xs = positions(indices.length, Math.min(4, Math.max(0, indices.length - 1) * 0.66));
  const ex = positions(indices.length, Math.min(7.8, Math.max(0, indices.length - 1) * 1.2));
  indices.forEach((index, slot) => {
    const tokenId = a.token(index, [xs[slot], -1.08, 0.2], [ex[slot], -2, 0.1]);
    const embeddingId = a.part({ id: `embedding-${index}`, label: `Input vector · ${index + 1}`, category: 'embeddings', kind: 'slab', tokenIndex: index,
      purpose: `The numeric input for token ${index + 1}. This wafer explains the input path; the trace does not export embedding coordinates.`,
      why: 'The recurrent update combines this step’s input with the previous state.',
    }, [xs[slot], -0.56, 0.16], [ex[slot], -0.84, 0.12], [0.43, 0.21, 0.59]);
    const stateId = a.part({ id: `state-${index}`, label: `State h${index + 1}`, category: 'memory', kind: 'sphere', tokenIndex: index, active: index === step, value: trace.steps[index]?.hiddenNorm,
      purpose: `${recorded(trace)} state after token ${index + 1}. Its norm is ${trace.steps[index]?.hiddenNorm.toFixed(3)}; it summarizes earlier input in a fixed-size vector.`,
      why: 'Sharing one update through time gives the model a memory path. Repeated transformations can still weaken an old signal; gates address that limitation.',
      math: 'hₜ = tanh(Wₓxₜ + Wₕhₜ₋₁ + b). Any backward glow in the teaching trace is a retention proxy, not the full derivative.',
    }, [xs[slot], 0.02, 0.15], [ex[slot], 0.67, 0.16], [0.51, 0.51, 0.51]);
    a.link(tokenId, embeddingId);
    a.link(embeddingId, stateId);
    if (slot) a.link(`state-${indices[slot - 1]}`, stateId, 'memory');
  });
  a.part({ id: 'recurrent-update', label: 'Shared recurrent update', category: 'memory', kind: 'slab',
    purpose: 'One update rule is reused at every time step. The row shows that single mechanism unfolded through time.',
    why: 'Weight sharing makes the same model work across different sequence lengths. The states are sequential, not parallel layers.',
  }, [0, 0.01, -0.32], [0, 1.56, -0.05], [4.6, 0.19, 0.22]);
  a.output([0, 0.83, 0.12], [0, 2.4, 0.12]);
  a.link('recurrent-update', `state-${step}`, 'memory');
  a.link(`state-${step}`, 'output');
  a.flow = indices.flatMap((index) => [`token-${index}`, `embedding-${index}`, `state-${index}`]).concat('output');
  return a.model(`One state travels through time, squeezing the past into a fixed-size vector. ${indices[0] > 0 ? `This view shows steps ${indices[0] + 1}–${step + 1}. ` : ''}Gates are the next idea: decide what to preserve.`);
}

const gateCopy: Record<keyof Gates, { label: string; purpose: string; why: string; math: string }> = {
  forget: { label: 'Forget valve', purpose: 'Scales the old cell state before it continues along the memory rail.', why: 'A dedicated preservation route lets memory survive many updates when the gate stays open.', math: 'fₜ ⊙ cₜ₋₁. A displayed scalar is the trace’s mean forget-gate value.' },
  input: { label: 'Write valve', purpose: 'Controls how much of the proposed candidate is written to the cell state.', why: 'Separating writing from preservation helps keep new inputs from overwriting useful memory.', math: 'iₜ ⊙ gₜ. A displayed scalar is the trace’s mean input-gate value.' },
  candidate: { label: 'Candidate', purpose: 'Proposes signed new content for memory. This is a tanh candidate, not a sigmoid gate.', why: 'The model needs possible new content as well as controls that choose how much to keep.', math: 'gₜ = tanh(W[xₜ; hₜ₋₁] + b). Candidate values can lie between −1 and 1.' },
  output: { label: 'Read valve', purpose: 'Controls how much of the cell state becomes the visible hidden state.', why: 'The information stored for later can differ from the information exposed now.', math: 'hₜ = oₜ ⊙ tanh(cₜ). A displayed scalar is the trace’s mean output-gate value.' },
  update: { label: 'Update valve', purpose: 'Mixes the old hidden state with newly proposed content.', why: 'A GRU combines keeping and writing in one gate, using a single memory state.', math: 'hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ gₜ. Here z controls NEW content.' },
  reset: { label: 'Reset valve', purpose: 'Controls how much of the previous hidden state is used to form the candidate.', why: 'The candidate can ignore old context when a new pattern begins.', math: 'gₜ = tanh(Wₓxₜ + Wₕ(rₜ ⊙ hₜ₋₁)). A displayed scalar is the mean reset gate.' },
};

function recordedGated(options: ModelBuildOptions): ExplorerModel {
  const a = new Assembly(options);
  const { trace, step } = options;
  const summary = trace.steps[step];
  const history = sequence(Math.max(0, step - 4), step + 1);
  const xs = positions(history.length, Math.min(3, (history.length - 1) * 0.65));
  const ex = positions(history.length, Math.min(5.6, (history.length - 1) * 1.1));
  history.forEach((index, slot) => a.token(index, [xs[slot], -1.79, 0.26], [ex[slot], -2.66, 0.25]));
  a.part({ id: 'current-embedding', label: 'Current input · schematic', category: 'embeddings', kind: 'bundle', tokenIndex: step,
    purpose: `Token ${step + 1} accompanies this recorded state. The export does not include its embedding or the exact recurrent computation.`,
    why: 'An input representation is a useful landmark for reading a recurrent trace. Its numerical values cannot be reconstructed from this recording.',
  }, [0, -1.16, 0.22], [0, -1.84, 0.15], [1.1, 0.3, 0.72]);
  a.link(`token-${step}`, 'current-embedding');
  a.part({ id: 'hidden-rail', label: 'Recorded hidden-state path', category: 'memory', kind: 'slab',
    purpose: 'The trace supplies hidden states at consecutive token positions. This path orders those observations; it does not reconstruct an unrecorded cell implementation.',
    why: 'Recurrent models carry a hidden representation through time. This schema labels the architecture as gated memory but does not identify the LSTM/GRU variant.',
  }, [0, -0.65, -0.13], [0, -0.74, -0.08], [3.3, 0.12, 0.18]);
  const arrays = summary.cell.length ? ['hidden', 'cell'] as const : ['hidden'] as const;
  for (const array of arrays) {
    const y = array === 'hidden' ? -0.65 : 0.66;
    const ey = array === 'hidden' ? -0.74 : 1.47;
    if (array === 'cell') {
      a.part({ id: 'cell-rail', label: 'Recorded cell array', category: 'memory', kind: 'slab', value: summary.cellNorm,
        purpose: `The export includes a cell array with ${summary.cell.length} entries and norm ${summary.cellNorm.toFixed(3)}. Presence of this field alone does not establish an LSTM: an exporter may include placeholder zeros.`,
        why: 'Showing a supplied cell array separately preserves the distinction from hidden state without inferring how it was computed.',
      }, [0, y, -0.06], [0, ey, 0.02], [3.3, 0.12, 0.2]);
    }
    for (const side of ['previous', 'current'] as const) {
      const x = side === 'previous' ? -2 : 2;
      const source = side === 'previous' ? trace.steps[step - 1] : summary;
      const norm = source?.[array === 'hidden' ? 'hiddenNorm' : 'cellNorm'];
      a.part({ id: `${array}-${side}`, label: `${side === 'previous' ? 'Previous' : 'Current'} ${array} array`, category: 'memory', kind: 'sphere', value: norm, active: side === 'current',
        purpose: norm === undefined ? `The initial ${array} array is not recorded. Its values are unknown.` : `Imported trace ${array}-array norm: ${norm.toFixed(3)}. The array is shown exactly as supplied.`,
        why: 'Adjacent recorded states let you inspect what changed between observations without claiming access to the unrecorded internal computation.',
      }, [x, y, 0.07], [x * 2, ey, 0.07], [0.5, 0.5, 0.5]);
    }
    a.link(`${array}-previous`, `${array}-rail`, 'memory');
    a.link(`${array}-rail`, `${array}-current`, 'memory');
  }
  const gates: (keyof Gates)[] = ['forget', 'input', 'candidate', 'output', 'update', 'reset'];
  const gx = positions(gates.length, 3.2), ge = positions(gates.length, 6.2);
  gates.forEach((gate, slot) => {
    a.part({ id: `gate-${gate}`, label: `Recorded ${gate}`, category: 'gates', kind: 'valve', value: summary.gates[gate],
      purpose: `Recorded ${gate} scalar: ${summary.gates[gate].toFixed(3)}. The schema requires all six summary fields; it does not establish which gates were used, their reduction convention, or whether unused fields are placeholders.`,
      why: 'This value is available for inspection. The LSTM/GRU variant and exact gate wiring are unknown, so the layout does not infer them from zero or nonzero fields.',
      math: `Canonical meaning when used: ${gateCopy[gate].math} The imported reduction convention is unrecorded.`,
    }, [gx[slot], 0.03, 0.38], [ge[slot], 0.35, 0.38], [0.5, 0.5, 0.4]);
  });
  a.output([0, 1.29, 0.12], [0, 2.54, 0.12]);
  a.link('current-embedding', 'hidden-rail');
  a.link('hidden-current', 'output');
  a.flow = [`token-${step}`, 'current-embedding', 'hidden-rail', 'hidden-current', 'output'];
  return a.model('Recorded gated-memory arrays and scalar summaries. The export does not identify the LSTM/GRU variant, gate wiring, or unused placeholder fields; those settings remain unknown.');
}

function gated(options: ModelBuildOptions): ExplorerModel {
  if (options.trace.provenance === 'imported-trace') return recordedGated(options);
  const a = new Assembly(options);
  const { trace, step, config } = options;
  const summary = trace.steps[step];
  const gru = config.gatedVariant === 'gru';
  const history = sequence(Math.max(0, step - 4), step + 1);
  const xs = positions(history.length, Math.min(3, (history.length - 1) * 0.65));
  const ex = positions(history.length, Math.min(5.6, (history.length - 1) * 1.1));
  history.forEach((index, slot) => a.token(index, [xs[slot], -1.79, 0.26], [ex[slot], -2.66, 0.25]));
  a.part({ id: 'current-embedding', label: 'Current input vector', category: 'embeddings', kind: 'bundle', tokenIndex: step,
    purpose: `Only the current token, ${step + 1}, enters this update. The other cards are its recent history. Embedding values are not exported.`,
    why: 'A gated recurrent cell uses the new input and the previous hidden state to decide its update.',
  }, [0, -1.16, 0.22], [0, -1.84, 0.15], [1.1, 0.3, 0.72]);
  a.link(`token-${step}`, 'current-embedding');
  a.part({ id: 'hidden-rail', label: gru ? 'Single memory rail' : 'Hidden-state rail', category: 'memory', kind: 'slab',
    purpose: gru ? 'A GRU carries one hidden state. It has no separate LSTM cell-state rail.' : 'The hidden state is the exposed signal passed to the next update and the readout.',
    why: gru ? 'A single state with two gates offers a simpler way to control recurrent memory.' : 'Separating stored memory from exposed output allows the cell to retain information without exposing all of it.',
  }, [0, -0.65, -0.13], [0, -0.74, -0.08], [3.3, 0.12, 0.18]);
  for (const side of ['previous', 'current'] as const) {
    const x = side === 'previous' ? -2 : 2;
    const source = side === 'previous' ? trace.steps[step - 1] : summary;
    a.part({ id: `hidden-${side}`, label: side === 'previous' ? 'Previous hidden state' : 'New hidden state', category: 'memory', kind: 'sphere',
      tokenIndex: side === 'previous' && step > 0 ? step - 1 : step, active: side === 'current', value: source?.hiddenNorm,
      purpose: source ? `${recorded(trace)} hidden-state norm: ${source.hiddenNorm.toFixed(3)}.` : 'The initial hidden state precedes the first recorded step. It is zero in the teaching simulation; an imported trace does not record its value.',
      why: side === 'previous' ? 'This is the recurrent signal available before the current token arrives.' : 'The updated hidden state carries the result into the next token and the output.',
    }, [x, -0.65, 0.07], [x * 2, -0.74, 0.07], [0.5, 0.5, 0.5]);
  }
  a.link('hidden-previous', 'hidden-rail', 'memory');
  a.link('hidden-rail', 'hidden-current', 'memory');
  const gates: (keyof Gates)[] = gru ? ['reset', 'update'] : ['forget', 'input', 'candidate', 'output'];
  const gx = positions(gates.length, gru ? 1.65 : 2.55);
  const ge = positions(gates.length, gru ? 2.6 : 6.2);
  gates.forEach((gate, slot) => {
    const copy = gateCopy[gate];
    a.part({ id: `gate-${gate}`, label: copy.label, category: 'gates', kind: 'valve', value: summary.gates[gate], active: true,
      purpose: `${copy.purpose} ${recorded(trace)} value: ${summary.gates[gate].toFixed(3)}.`, why: copy.why, math: copy.math,
    }, [gx[slot], 0.03, 0.38], [ge[slot], 0.35, 0.38], [0.55, 0.55, 0.4]);
    a.link('current-embedding', `gate-${gate}`);
    a.link('hidden-previous', `gate-${gate}`, 'memory');
  });
  if (gru) {
    a.part({ id: 'gru-candidate', label: 'Candidate state', category: 'memory', kind: 'bundle', value: summary.gates.candidate,
      purpose: `${recorded(trace)} mean candidate: ${summary.gates.candidate.toFixed(3)}. The reset gate controls its access to old state.`,
      why: 'The update gate needs a proposed new state to mix with the old one.', math: gateCopy.reset.math,
    }, [0, 0.64, 0.1], [0, 1.3, 0.12], [1.5, 0.34, 0.76]);
    a.link('gate-reset', 'gru-candidate');
    a.link('gru-candidate', 'gate-update');
    a.link('gate-update', 'hidden-current', 'memory');
  } else {
    a.part({ id: 'cell-rail', label: 'Cell-state rail', category: 'memory', kind: 'slab', value: summary.cellNorm,
      purpose: `${recorded(trace)} cell-state norm: ${summary.cellNorm.toFixed(3)}. The rail shows the additive memory route.`,
      why: 'A controlled additive path helps useful information survive repeated updates. Gates choose preservation and writing.',
      math: 'cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ gₜ. The rail geometry is schematic, not the magnitude of cₜ.',
    }, [0, 0.66, -0.06], [0, 1.47, 0.02], [3.3, 0.12, 0.2]);
    for (const side of ['previous', 'current'] as const) {
      const x = side === 'previous' ? -2 : 2;
      const source = side === 'previous' ? trace.steps[step - 1] : summary;
      a.part({ id: `cell-${side}`, label: side === 'previous' ? 'Previous cell memory' : 'New cell memory', category: 'memory', kind: 'sphere', value: source?.cellNorm, active: side === 'current',
        purpose: source ? `${recorded(trace)} cell-state norm: ${source.cellNorm.toFixed(3)}.` : 'Initial cell memory is zero in the simulation. The imported schema does not record an initial cell vector.',
        why: 'Cell memory is retained separately from the hidden state exposed to the rest of the model.',
      }, [x, 0.66, 0.04], [x * 2, 1.47, 0.04], [0.5, 0.5, 0.5]);
    }
    a.link('cell-previous', 'gate-forget', 'memory');
    a.link('gate-forget', 'cell-rail', 'memory');
    a.link('gate-input', 'cell-rail');
    a.link('gate-candidate', 'gate-input');
    a.link('cell-rail', 'cell-current', 'memory');
    a.link('cell-current', 'gate-output');
    a.link('gate-output', 'hidden-current', 'memory');
  }
  a.output([0, 1.29, 0.12], [0, 2.54, 0.12]);
  a.link('hidden-current', 'output');
  a.flow = [`token-${step}`, 'current-embedding', ...(gru ? ['gate-reset', 'gru-candidate', 'gate-update'] : ['gate-forget', 'gate-candidate', 'gate-input', 'cell-rail', 'gate-output']), 'hidden-current', 'output'];
  return a.model(`${gru ? 'Two gates regulate a single hidden-state rail.' : 'Four tactile controls regulate a cell-state rail and a hidden-state rail.'} Memory can survive longer, but remains a fixed-size bottleneck. Attention lets the model look back directly.`);
}

function windowAround(index: number, length: number, size: number): number[] {
  const start = Math.max(0, Math.min(length - size, index - Math.floor(size / 2)));
  return sequence(start, Math.min(length, start + size));
}

function attention(options: ModelBuildOptions): ExplorerModel {
  const a = new Assembly(options);
  const { trace, config } = options;
  const decoder = clampIndex(options.decoderStep, trace.decoderTokens.length);
  const weights = trace.alignment[decoder] ?? [];
  const strongest = weights.reduce((best, value, index) => value > (weights[best] ?? -1) ? index : best, 0);
  const sources = windowAround(strongest, trace.encoderTokens.length, 8);
  const targets = windowAround(decoder, trace.decoderTokens.length, 6);
  const sx = positions(sources.length, Math.min(4.2, Math.max(0, sources.length - 1) * 0.6));
  const se = positions(sources.length, Math.min(7.8, Math.max(0, sources.length - 1) * 1.08));
  const tx = positions(targets.length, Math.min(3.6, Math.max(0, targets.length - 1) * 0.69));
  const te = positions(targets.length, Math.min(7.2, Math.max(0, targets.length - 1) * 1.4));
  const imported = trace.provenance === 'imported-trace';
  const hasAttention = imported || config.attentionEnabled;
  sources.forEach((index, slot) => {
    a.part({ id: `source-${index}`, label: trace.encoderTokens[index], category: 'tokens', kind: 'card', tokenIndex: index, active: index === strongest && hasAttention,
      purpose: `Source word ${index + 1}: “${trace.encoderTokens[index]}”. ${trace.provenance === 'educational-simulation' ? 'These conceptual word pairs are a separate Hinglish → English demonstration, not translation inference on the experiment input.' : 'This word belongs to the imported encoder sequence.'}`,
      why: 'Keeping source positions available gives a decoder a place to look back.',
    }, [sx[slot], -1.56, 0.23], [se[slot], -2.29, 0.23], [0.52, 0.42, 0.17]);
    a.part({ id: `source-state-${index}`, label: `Source state · ${index + 1}`, category: 'memory', kind: 'sphere', tokenIndex: index, value: magnitude(trace.encoderStates[index]),
      purpose: `The recorded encoder-state vector for source position ${index + 1}. Its norm is ${magnitude(trace.encoderStates[index])?.toFixed(3) ?? 'unavailable'}.`,
      why: 'Attention retrieves numerical representations of source words, rather than the text itself.',
    }, [sx[slot], -0.96, 0.14], [se[slot], -1.2, 0.14], [0.39, 0.39, 0.39]);
    a.link(`source-${index}`, `source-state-${index}`);
    a.link(`source-state-${index}`, hasAttention ? 'attention-context' : 'compression-bottleneck', hasAttention ? 'attention' : 'memory', hasAttention ? weights[index] : undefined);
  });
  a.part({ id: hasAttention ? 'attention-context' : 'compression-bottleneck', label: imported ? 'Recorded alignment' : hasAttention ? 'Weighted context' : 'Fixed-state bottleneck', category: hasAttention ? 'attention' : 'memory', kind: hasAttention ? 'bundle' : 'sphere',
    purpose: imported ? `Recorded alignment row ${decoder + 1} determines these strands, without renormalizing the visible subset. The export does not say whether the original model used these weights to compute its context.` : hasAttention ? `${recorded(trace)} alignment row ${decoder + 1} determines these strands. Each drawn weight is the exact recorded value; the visible subset is not renormalized.` : 'Attention is disabled. The demonstration reads one compressed source summary for every decoding step; it no longer retrieves individual source states.',
    why: hasAttention ? 'A fresh weighted context gives each decoder step access to different source positions.' : 'A single fixed-size summary can lose detail as the source grows. Turning on attention restores a direct retrieval path.',
    math: hasAttention ? 'contextⱼ = Σᵢ αⱼᵢ hᵢ. The full alignment row sums to 1; omitted source positions still retain their recorded mass.' : 'Teaching compression: tanh(mean(source states) × min(1, 5 / source length)).',
  }, [0, -0.2, 0.16], [0, 0.04, 0.12], hasAttention ? [2.15, 0.38, 0.87] : [0.75, 0.75, 0.75]);
  a.part({ id: 'decoder-state', label: `Decoder step · ${decoder + 1}`, category: 'memory', kind: 'slab', decoderIndex: decoder, active: true,
    purpose: `The selected target position is ${decoder + 1}: “${trace.decoderTokens[decoder] ?? ''}”. ${imported ? 'Its supplied alignment row is displayed; the decoder implementation is unrecorded.' : hasAttention ? 'Its own alignment row chooses a context.' : 'It receives the shared compressed summary.'}`,
    why: 'Decoding needs an evolving target-side representation. The teaching trace illustrates alignment; it does not execute a trained translation decoder.',
  }, [0, 0.43, 0.15], [0, 1.1, 0.15], [1.75, 0.37, 0.88]);
  targets.forEach((index, slot) => {
    a.part({ id: `decoder-${index}`, label: trace.decoderTokens[index], category: 'output', kind: 'card', decoderIndex: index, active: index === decoder,
      purpose: `Target word ${index + 1}: “${trace.decoderTokens[index]}”. Select it to inspect its recorded source-alignment row.`,
      why: 'Different output positions can retrieve different source positions, including across changes in word order.',
    }, [tx[slot], 1.13, 0.2], [te[slot], 2.28, 0.2], [0.57, 0.43, 0.17]);
    a.link('decoder-state', `decoder-${index}`);
  });
  a.link(hasAttention ? 'attention-context' : 'compression-bottleneck', 'decoder-state');
  a.flow = [`source-${strongest}`, `source-state-${strongest}`, hasAttention ? 'attention-context' : 'compression-bottleneck', 'decoder-state', `decoder-${decoder}`].filter((id) => a.parts.some((part) => part.id === id));
  const visibleMass = sources.reduce((total, index) => total + (weights[index] ?? 0), 0);
  const subset = sources.length < trace.encoderTokens.length ? ` Showing source positions ${(sources[0] ?? 0) + 1}–${(sources.at(-1) ?? 0) + 1}; ${(100 * Math.max(0, 1 - visibleMass)).toFixed(1)}% of recorded alignment mass lies outside this view.` : '';
  return a.model(`A bridge retrieves useful source states for the selected decoder word.${subset} Transformers take the next step: use attention inside the sequence itself.`);
}

function transformer(options: ModelBuildOptions): ExplorerModel {
  const a = new Assembly(options);
  const { trace, step, config } = options;
  const imported = trace.provenance === 'imported-trace';
  const weightTarget = imported || config.aggregateHeads ? 'attention-mix' : `head-${config.head}`;
  const weights = trace.attention[step] ?? [];
  const strongest = weights.reduce((best, value, index) => value > (weights[best] ?? -1) ? index : best, 0);
  // Keep the selected query and strongest actual contributor, even across a long imported sequence.
  const indices = [...new Set([step, strongest, ...sequence(Math.max(0, step - 5), step + 1), ...windowAround(step, trace.tokens.length, 8)])]
    .filter((index) => index >= 0 && index < trace.tokens.length).slice(0, 8).sort((x, y) => x - y);
  const xs = positions(indices.length, Math.min(4.05, Math.max(0, indices.length - 1) * 0.57));
  const ex = positions(indices.length, Math.min(7.7, Math.max(0, indices.length - 1) * 1.05));
  indices.forEach((index, slot) => {
    a.token(index, [xs[slot], -1.79, 0.23], [ex[slot], -2.68, 0.22]);
    a.link(`token-${index}`, 'token-embeddings');
    if (weights[index] !== undefined) a.link(`token-${index}`, weightTarget, 'attention', weights[index]);
  });
  a.part({ id: 'token-embeddings', label: 'Token embeddings', category: 'embeddings', kind: 'slab',
    purpose: 'A shared numerical representation for each token enters the stack. The geometry is schematic; this trace does not export the embedding table.',
    why: 'Attention works on vectors, so token identity must first become numerical.', math: 'xᵢ = E[tokenᵢ].',
  }, [0, -1.31, 0.02], [0, -2.02, -0.05], [2.7, 0.21, 1.18]);
  a.part({ id: 'position-encoding', label: imported ? 'Position information · unrecorded' : config.positionalEncoding ? 'Position encoding' : 'Position encoding · off', category: 'embeddings', kind: 'slab', ghost: !imported && !config.positionalEncoding,
    purpose: imported ? 'Position encoding, its type, and its enabled state are unknown: the imported schema does not record them. This plate marks the architectural idea without inferring a setting.' : config.positionalEncoding ? 'The teaching engine adds a small sinusoidal position signal. Positional vectors are not present in this schema.' : 'Position encoding is disabled in the current experiment configuration.',
    why: 'Token identity alone does not express sequence order. Position signals let attention distinguish where a token appears.',
    math: 'xᵢ = token_embeddingᵢ + position_signalᵢ (schematic).',
  }, [0, -1.01, 0.02], [0, -1.58, -0.1], [2.65, 0.14, 1.13]);
  a.part({ id: 'residual-stream', label: 'Residual stream', category: 'residual', kind: 'slab',
    purpose: 'A bypass carries representations around an attention or feed-forward sublayer. This rail is an architectural explanation, not an exported tensor.',
    why: 'Residual additions let each sublayer update an existing representation and support training deeper networks.',
    math: 'x ← x + Sublayer(x) (standard schematic). The teaching engine uses a simplified 0.3×previous-state mixing route, not a full trained Transformer block.',
  }, [-1.74, 0.1, -0.23], [-4.47, 0.16, -0.12], [0.15, 2.8, 0.2]);
  const headX = [-0.62, 0.62, -0.62, 0.62];
  const headZ = [0.43, 0.43, -0.44, -0.44];
  const headExploded = positions(4, 5.7);
  const roles = ['locality', 'repetition', 'distant-clue retrieval', 'position pattern'];
  for (let head = 0; head < 4; head++) {
    const id = `head-${head}`;
    const selected = !imported && head === config.head && !config.aggregateHeads;
    a.part({ id, label: `Attention head ${head + 1}`, category: 'attention', kind: 'head', active: selected,
      purpose: trace.provenance === 'educational-simulation' ? `This hand-designed teaching head emphasizes ${roles[head]}. ${selected ? 'The weighted strands use this head’s recorded attention row.' : config.aggregateHeads ? 'The trace currently shows the mean of four heads.' : 'Select this head to request its weights.'}` : `A schematic head in a multi-head block. The imported schema contains one attention matrix and does not identify separate per-head tensors.`,
      why: 'Multiple heads can form different information-routing patterns. These four teaching patterns are illustrative, not learned linguistic discoveries.',
      math: 'Attention(Q, K, V) = softmax(QKᵀ / √dₖ + mask)V. Q, K, V are conceptual here; their tensors are not exported.',
    }, [headX[head], -0.39, headZ[head]], [headExploded[head], -0.03, 0.1], [0.9, 0.53, 0.69]);
    const childCopy = [
      { suffix: 'q', label: 'Query', purpose: 'A query represents what the current position is looking for.', why: 'It is compared with keys to decide which source representations matter.', math: 'Q = XW_Q (architectural schematic).' },
      { suffix: 'k', label: 'Key', purpose: 'Keys describe how source positions can be matched by a query.', why: 'Separating lookup features from carried content makes matching flexible.', math: 'K = XW_K (architectural schematic).' },
      { suffix: 'v', label: 'Value', purpose: 'Values are the content combined after attention weights are chosen.', why: 'The features used for matching need not be the same features returned.', math: 'V = XW_V (architectural schematic).' },
      { suffix: 'weights', label: 'Attention weights', purpose: imported ? 'Separate head matrices, layer identity, and mask settings are unrecorded. The supplied attention matrix is displayed at the shared readout; it is not attributed to this schematic head.' : selected || config.aggregateHeads ? 'The visible strand widths use the current trace’s attention row. Q/K projections are not available in this trace.' : 'This head’s own matrix is not in the current trace. Select the head to request a teaching trace for it.', why: 'Normalized weights turn scores into a distribution over source positions.', math: 'A = softmax(QKᵀ / √dₖ + mask). A masked future position has weight 0.' },
    ];
    childCopy.forEach((copy, slot) => {
      const weightsPart = slot === 3;
      a.part({ id: `${id}-${copy.suffix}`, label: `H${head + 1} · ${copy.label}`, category: 'attention', kind: weightsPart ? 'slab' : 'card', parentId: id, detailOnly: true,
        purpose: `${copy.purpose} This piece explains a standard attention head; no Q/K/V tensor values are inferred.`, why: copy.why, math: copy.math,
      }, [headX[head] + (slot - 1) * 0.07, -0.39, headZ[head]],
      weightsPart ? [headExploded[head], 0.72, 0.1] : [headExploded[head] + (slot - 1) * 0.42, -0.79, 0.1], weightsPart ? [1.12, 0.24, 0.42] : [0.31, 0.38, 0.33]);
      a.link(id, `${id}-${copy.suffix}`, 'attention');
    });
    a.link('position-encoding', id);
    a.link(id, 'attention-mix', 'attention');
  }
  a.part({ id: 'attention-mix', label: imported ? 'Recorded attention matrix' : 'Attention readout', category: 'attention', kind: 'slab',
    purpose: trace.provenance === 'imported-trace' ? 'The weighted strands use the imported attention matrix. This schema does not record separate per-head or per-layer tensors; the stack is an architectural schematic.' : `The current trace uses ${config.aggregateHeads ? 'the mean of four illustrative heads' : `head ${config.head + 1}`} at teaching layer ${config.layer + 1}. The stacked form is schematic.`,
    why: 'Information retrieved from other positions is mixed into the current representation.',
    math: 'Standard multi-head attention concatenates head outputs and projects them. This teaching engine mixes using one selected head or their mean.',
  }, [0, 0.12, 0.03], [0, 1.25, 0.06], [2.6, 0.24, 1.17]);
  a.part({ id: 'residual-add', label: 'Residual addition', category: 'residual', kind: 'sphere',
    purpose: 'A conceptual junction combines the carried representation with a sublayer update.',
    why: 'An update can build on what is already present instead of replacing the whole representation.',
    math: 'x + update. The teaching trace uses a simplified recurrence and does not export separate residual activations.',
  }, [0, 0.54, 0.1], [2.5, 1.6, 0.1], [0.36, 0.36, 0.36]);
  a.part({ id: 'feed-forward', label: 'Feed-forward path', category: 'memory', kind: 'slab',
    purpose: 'A standard Transformer applies the same small feed-forward network separately at every token. This piece is conceptual: the teaching engine does not execute or export a separate FFN.',
    why: 'Attention moves information between positions; a feed-forward sublayer transforms features within each position.',
    math: 'FFN(x) = W₂ σ(W₁x + b₁) + b₂ (architectural schematic).',
  }, [0, 0.95, 0.03], [0, 1.9, 0.03], [2.24, 0.33, 1.04]);
  a.output([0, 1.49, 0.04], [0, 2.62, 0.04]);
  a.link('token-embeddings', 'position-encoding');
  a.link('token-embeddings', 'residual-stream', 'residual');
  a.link('attention-mix', 'residual-add');
  a.link('residual-stream', 'residual-add', 'residual');
  a.link('residual-add', 'feed-forward');
  a.link('feed-forward', 'output');
  a.flow = [`token-${step}`, 'token-embeddings', 'position-encoding', weightTarget, 'attention-mix', 'residual-add', 'feed-forward', 'output'].filter((id, index, ids) => index === 0 || id !== ids[index - 1]);
  const subset = indices.length < trace.tokens.length ? ` ${indices.length} of ${trace.tokens.length} positions are drawn, including this query and its strongest recorded contributor; strand weights are not renormalized.` : '';
  return a.model(`A compact stack lets tokens retrieve relevant positions directly. Select a head and explode it to reveal the query, key, value and weighting idea.${subset}`);
}

export function buildExplorerModel(options: ModelBuildOptions): ExplorerModel {
  if (!options.trace.steps.length || !options.trace.tokens.length) {
    return { architecture: options.trace.architecture, parts: [], links: [], flow: [], description: 'No recorded tokens or states are available.' };
  }
  const normalized = { ...options, config: { ...options.config, head: clampIndex(options.config.head, 4) }, step: clampIndex(options.step, Math.min(options.trace.tokens.length, options.trace.steps.length)) };
  switch (options.trace.architecture) {
    case 'mlp': return fixedWindow(normalized);
    case 'rnn': return recurrent(normalized);
    case 'lstm': return gated(normalized);
    case 'attention': return attention(normalized);
    case 'transformer': return transformer(normalized);
  }
}
