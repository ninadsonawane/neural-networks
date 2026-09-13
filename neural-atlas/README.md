# Neural Atlas

**How Language Models Learned to Remember** is an interactive museum of five language-model architectures and a visual companion to [ninadsonawane/neural-networks](https://github.com/ninadsonawane/neural-networks). Start with an assembled model, take it apart, and discover why each component exists. Switch to Experiment to follow the same character sequence through a fixed-context MLP, vanilla RNN, LSTM/GRU, attention bridge, and causal Transformer.

The browser runs **deterministic educational simulations**. No trained weights or notebook checkpoints are bundled, and the probability bars are not predictions from a trained language model. The live attention experiment retrieves characters from your own input. Legacy sample recordings also include a conceptual English/Hinglish word alignment; neither is trained translation. The spatial assemblies are explanatory diagrams, not literal hardware or biological neurons.

## Run and verify

From this directory:

Use Node.js 22.15+ (or another Node version supported by Vite 7). Dependencies are pinned by `package-lock.json`; use `npm ci` for reproducible installs.

```bash
npm install
npm run dev
```

Open the local URL printed by Vite. To check and preview a production build:

```bash
npm run test
npm run lint
npm run build
npm run preview
```

`npm run build` runs TypeScript checking and emits a static site to `dist/`. Deploy the contents of `dist/` to any static host. Set Vite's `base` option before building if serving beneath a repository subpath. The app has no server-side model runtime.

Six ready-to-import educational recordings live in `public/traces/`: one for every architecture and an additional Hinglish Transformer trace. Regenerate them with `npm run samples`. These files remain simulations even when replayed through the import feature.

DM Sans and IBM Plex Mono are bundled locally through Fontsource, with their SIL Open Font License notices in `public/licenses/`. No font service, external image, model download, or API key is needed at runtime.

## Learn first, then experiment

**Learn is the default.** The architecture opens as a compact assembly with a short explanation and a small control dock. Drag to orbit and scroll to zoom. Move the Assembly → Exploded slider to separate the components, then select a piece to read its purpose and why it exists. Isolate it for a closer look, return it to context, or filter semantic categories such as tokens, embeddings, memory, gates, attention, residual stream, and output. Reset reassembles the model and restores the camera, filters, and replay state. The evolution rail selects the five architectures.

**Experiment follows the signal.** Type and tokenize a short phrase, choose the cat/mouse or Hinglish example, or add a distant clue. Optionally choose a character through the hidden **Pick clue** dropdown, then use **Throw it in** to arc paper tokens into the assembly and follow their processing route in slow motion. Pause, step backward or forward, scrub, and replay. At the end, one sentence explains what reached the output. “Try the next idea” runs the same input through the next architecture. This is replay of the supplied sequence, not generated text or notebook-checkpoint inference.

Details appear when requested. Select a component and choose “Go deeper,” open Experiment settings, inspect the output probabilities, or choose Compare/About from the utility controls. The detail sheet contains:

- A compact mechanism equation and expandable model settings: MLP context 1–32, RNN recurrent multiplier, LSTM/GRU gates and clue preservation, attention-bridge source length, and Transformer layer/head/position/mask controls.
- Manual gate overrides that apply at every step and take precedence over automatic values and clue preservation. GRU remains explicitly conceptual.
- Temperature, recorded or simulated probability bars, signed state summaries, retained-index annotations, and selected source/decoder weights.
- A normalized attention matrix with exact values, keyboard cell navigation, and source/target selection. Spatial scenes show bounded subsets of the trace; the matrix retains all recorded positions.
- Comparison using the same input and shared replay store, plus source notes and JSON import/export. Imported replay hides simulation settings and simulated comparison.

The RNN backward signal is a scalar gradient-transport illustration, not a measured training gradient. Head patterns and translation alignment are deliberately constructed teaching examples. Changing a control changes the simulation; it does not train a model.

## Cinematic exploration

Huge architecture titles and camera-facing labels keep the machine recognizable. Hover lifts and illuminates a component; click isolates it with a short explanation. Double-click or Escape returns to the assembly. Scan briefly reveals nearby labels. Replay pauses at a clue-retention moment, highlights the responsible component, and brings its research card forward.

**Filter** opens the otherwise hidden **Architecture Playground**. Stage-specific controls animate context windows, sequence stacks, widths, gates, retrieval patterns, head counts, and layer stacks. Tiny, Notebook-like, Wide, Long context, and Reset presets provide starting points. Each stage remembers its own controls. Width settings illustrate capacity through geometry; numerical teaching states remain eight-dimensional, and presets are not training benchmarks. Transformer layers and heads are bounded to four each. Context/retention controls affect the educational computation.

Throw-in animation displays up to 20 physical cards with no permanent clue tray; all input positions up to the configured 80-character cap still process. Scenes show bounded subsets of larger sequences to retain space and performance. Attention ribbons use the simulated or recorded weights. Parallel Transformer animation depicts eligible attention positions, not parallel autoregressive generation.

One original archival cover accompanies each idea: Bengio (2003) for MLP, Mikolov (2010) for RNN, Graves (2013) for LSTM sequence generation, Cho (2014) for GRU, Bahdanau (2015) for attention, and Vaswani (2017) for Transformer. Graves is not credited with inventing LSTM; the detail card distinguishes the 1997 origin. Sutskever (2014) and Cho's original encoder–decoder paper appear as related reading. Cards contain original summaries and primary-source links, never reproduced PDF pages. “Why was this invented?” connects the cover to its relevant component; “Try the next idea” changes both the assembly and its paper.

## Accessibility and fallback views

The component outline exposes the same selectable parts without requiring canvas interaction. Selecting a part from the outline moves keyboard focus to its explanation. Tab navigates controls, Enter/Space activates buttons, and native range controls support arrow keys. In Experiment, outside interactive controls, Left/Right steps the timeline and Space controls replay. Escape dismisses a selection or open panel.

The detail sheet uses a native modal dialog with a focus trap, Escape dismissal, focus restoration, and a keyboard-scrollable content region. Its shortcuts do not trigger the underlying experiment. Matrix cells support arrow/Home/End navigation and announce exact values; state bars encode sign spatially and provide readable numeric values.

Choose the 2D utility for a flat interactive overview. Reduced-motion preference, unavailable WebGL, and context loss also select a 2D fallback. Component descriptions, semantic filters, selection, and isolation remain usable there. On narrow screens the layout adapts to touch controls, and comparison presents one architecture at a time at the shared timeline position.

## Component and data architecture

```text
src/
  App.tsx                         Small entry point that reexports Explorer
  Explorer.tsx                    Learn/Experiment flow, selection, rail, and dock
  explorer.css                    Responsive museum interface and overlays
  store.ts                        Zustand: replay, mode, assembly, filters, selection
  explorer/
    types.ts                      Semantic parts, categories, links, and model contract
    models.ts                     Trace → assembled/exploded architecture models
    models.test.ts                Semantic graph, layout, and recorded-value invariants
  components/
    MuseumScene.tsx               Orbitable Three.js assemblies and interactive 2D fallback
    ExplorerDetails.tsx           On-demand native modal: controls, data, compare, sources
    AttentionMatrix.tsx           Accessible, selectable attention data
    SignalInspector.tsx           Current trace values and provenance
    CompareMode.tsx               Shared-time architecture comparison
  data/
    stages.ts                     Narrative explanations and equations
    repository.ts                 Source-verified provenance records and links
  simulation/
    types.ts                      Model-agnostic Trace and TraceAdapter contract
    engine.ts                     Seeded educational traces and helpers
    adapter.ts                    Validated JSON import boundary
    engine.test.ts                Determinism and mathematical invariants
scripts/
  export-samples.ts                Regenerate six validated educational JSON recordings
public/traces/                    Ready-to-import sample recordings
```

React, TypeScript, and Vite provide the application shell. Three.js, React Three Fiber, and Drei supply spatial assemblies and orbit controls; DOM/SVG provide labels, descriptions, matrices, and the fallback view. Framer Motion handles restrained selection and panel transitions. Zustand keeps the architecture, replay, assembly slider, visibility filters, selection, and isolation consistent.

`Trace` separates model data from presentation. The engine produces educational traces, and the JSON adapter supplies recorded traces. `buildExplorerModel` translates either into an `ExplorerModel`: semantic parts with stable IDs, assembled and exploded positions, explanatory copy, categories, and directed links. `MuseumScene` renders that description without PyTorch-specific code. Geometry is schematic; embedding wafers and query/key/value pieces do not claim to show exported coordinates or learned tensors that the trace does not contain.

Identical input and settings produce identical educational traces and semantic models. Changing temperature rescales the categorical distribution; it does not train a model. Fixed character embeddings are hand-designed mathematical features, not learned repository embeddings or a PCA/UMAP export.

## Actual experiments, simulations, and future work

The following facts were checked against executable cells in the supplied source checkout. The notebooks were inspected, **not rerun or independently benchmarked**.

| Stage or data | In the supplied repository | What this website displays |
| --- | --- | --- |
| [MLP](https://github.com/ninadsonawane/neural-networks/blob/main/MLP.ipynb) | Character vocabulary 28 including `.`; default `block_size=3`, embedding size 8, hidden width 200; character/name/Hinglish dataset exploration. | A deterministic fixed-window simulation, illustrative embeddings, and normalized next-character probabilities. |
| Documented MLP loss | Source README reports approximately 1.89 for embedding 6/context 6 after 30,000 steps. This is separate from current notebook defaults and was not reproduced here. | A provenance note only. No leaderboard or claim of comparable evaluation across models. |
| [Vanilla RNN](https://github.com/ninadsonawane/neural-networks/blob/main/RNN.ipynb) | One-hot input, 50-character training windows, shared tanh recurrence, BPTT, norm-5 clipping; executable `hidden_size=300`. | A compact recurrent state and adjustable scalar gradient transport. Browser state width is illustrative. |
| [LSTM](https://github.com/ninadsonawane/neural-networks/blob/main/LSTM.ipynb) | Length-100 sequences, four explicit gates, dropout, clipping, loss/perplexity tracking; executable `hidden_size=256`. Contains checkpoint-saving code. | A compact gate/cell simulation and a clue-preservation intervention. No trained gate values. |
| GRU / encoder-decoder / seq2seq / Bahdanau attention | Listed as planned or not pushed; no corresponding notebook or trained artifact was supplied. | Faithful conceptual gate and alignment demonstrations. The English/Hinglish alignment is invented for teaching, not machine translation output. |
| [Transformer](https://github.com/ninadsonawane/neural-networks/blob/main/transformer.ipynb) | Character-level GPT code with context 8, embedding width 32, four layers/four heads, token and learned position embeddings, causal masking, residual blocks, layer normalization, and AdamW. | Small deterministic attention examples, a visible mask, head/layer exploration, and replay. No browser checkpoint inference. |
| [Self-attention block](https://github.com/ninadsonawane/neural-networks/blob/main/self-attention-block.ipynb) | Exploratory causal averaging and query/key/value cells, separate from the full Transformer notebook. | Normalized illustrative matrices and spatial token-to-token connections. |
| Model checkpoints / exported traces | No `.pt`, `.pth`, `.ckpt`, `.safetensors`, or trace JSON files were found in the supplied checkout. LSTM save/load code does not mean its output artifact was supplied. | Simulation by default; separately labeled imported recording when a user provides a valid trace. |

The brief and some old inline comments mention RNN/LSTM hidden width 100. The executable values in this checkout are 300 and 256 respectively; this site records those values rather than treating stale comments as measurements. Exploratory notebooks also contain saved error outputs, so a fresh end-to-end run is part of any real-model integration.

Scientific boundaries matter: an RNN can carry state across steps, but the notebook still trains on finite windows. LSTM gates can improve gradient flow without guaranteeing perfect memory or constant gradients. Full self-attention has quadratic attention-matrix cost in sequence length; parallel training positions do not make autoregressive generation parallel. Turning the causal mask off is a teaching intervention that reveals future input, not valid causal next-token evaluation. Attention weights describe this mechanism's mixing coefficients; they are not a complete causal explanation of model behavior.

## Import a recorded PyTorch trace

`src/simulation/types.ts` is the stable renderer contract. A version-1 `Trace` contains:

- `tokens`: token text, unique ID, sequential position, and the required pedagogical `isClue` flag.
- `steps`: per-step state vectors/norms, gate values, retained indices, logits, probabilities, and gradient summaries.
- `attention`: the currently exported self-attention matrix.
- `encoderTokens`, `decoderTokens`, `encoderStates`, and `alignment`: source/target records for the bridge scene.
- `architecture`, `input`, `vocabulary`, `schemaVersion`, `provenance`, and `metadata`: identity and provenance.

Open About from the help icon, then use **Import JSON** in the detail sheet to load a compatible recording. Import is read-only replay: the timeline, component selection, assembly, and camera remain explorable, while simulation settings cannot silently change recorded values. Return to simulation to resume model interventions. The same sheet can export the current trace or download the six educational samples. A JSON trace is an observation of a forward pass, **not a model checkpoint** and not a mechanism for generating new trained-model predictions.

The JSON adapter accepts files up to 2 MB, 1–256 tokens, and consistent state vectors up to 4,096 values. Every token needs one corresponding step. Attention/alignment rows and full-vocabulary probability distributions must be finite, nonnegative, and normalized; probabilities are sorted from highest to lowest. Recorded vector norms, dimensions, indices, and metadata are checked. The default teaching engine limits input to 80 Unicode code points, uses eight-dimensional state summaries, and has an illustrative 28-symbol output vocabulary (`a`–`z`, space, and `.`). It can visualize other input characters without claiming to have trained on them.

The adapter API is intentionally small:

```ts
import { DEFAULT_CONFIG, buildTrace } from './src/simulation/engine';
import { loadTraceJson, JsonTraceAdapter } from './src/simulation/adapter';

// Educational data, useful for learning the schema and testing a round trip.
const sample = buildTrace('the cat that chased the mouse was ...', 'transformer', DEFAULT_CONFIG);
const json = JSON.stringify(sample);
const recording = loadTraceJson(json, 'teaching-example.json');

// An alternative to direct import, using the common async TraceAdapter interface.
const adapter = new JsonTraceAdapter(json, 'teaching-example.json');
const trace = await adapter.load(recording.input, recording.architecture, DEFAULT_CONFIG);
```

`EducationalTraceAdapter` implements the same interface. `JsonTraceAdapter` always returns a copy of its recorded trace; it does not recompute new settings or input. Import labels the result “Imported trace” and states that trained-model provenance has not been verified.

To create a real recording:

1. Reproduce a notebook run and preserve the exact character vocabulary/order, checkpoint identifier, dataset description, and model configuration. Repair exploratory cells as necessary before claiming reproducibility.
2. In an evaluation forward pass, capture the actual per-character hidden/cell states, gates, output logits, and any attention weights needed by the scene. For the Transformer, capture each head's softmax weights inside `Head.forward`; they are not returned by the current notebook's public `forward` method.
3. Convert tensors with `tensor.detach().cpu().tolist()`. Export probabilities using a stable softmax at a documented temperature. Norm summaries must match the recorded vectors. Use actual gradient measurements from a separate backward pass if gradient data is included; do not substitute the site's scalar multiplier and label it measured.
4. Map the recorded arrays to `Trace` in chronological token order. Use `schemaVersion: 1`, `provenance: 'imported-trace'`, the correct architecture, and a `metadata.source` describing the checkpoint/export. Include the checkpoint identity in `metadata.description` too, because UI import identifies its source using the imported filename. The contract also requires visualization annotations (`isClue`, `retainedIndices`, `clueStrength`) and all six gate summary keys; describe inapplicable fields and any nonmeasured annotations explicitly in `metadata.description`. Gradient entries must be nonnegative and have `step.index + 1` values. This version does not support per-field provenance, so use a trace only when its displayed data can be described honestly.
5. Validate the JSON with the adapter before displaying it. Start by round-tripping the app's sample trace export to learn the exact schema. Its values remain educational simulation data even after importing; a successful validation checks structure, not model authenticity.

The current schema is a compact replay format: it stores one selected attention matrix rather than an entire layer/head tensor archive. Export a selected layer/head with its identity in metadata, or extend the adapter contract deliberately if the full recording needs live layer/head selection. There is no automatic conversion from a raw `.pt` file and no browser PyTorch runtime.

## Validation scope

`npm run test` runs `src/simulation/engine.test.ts`, `src/explorer/models.test.ts`, and `src/explorer/cinema.test.ts`. The cinematic suite checks processing routes, topology bounds, custom attention, context limits, and imported-recording preservation. The simulation suite checks determinism, model-mechanism invariants, probabilities, and the JSON boundary. The explorer suite checks stable semantic IDs, valid graph links, bounded geometry, separated exploded parts, and agreement between displayed values and their trace. `npm run lint` checks source quality; `npm run build` checks types and the production bundle.

Interface verification should cover orbit/zoom, assembled/exploded views, component selection/isolation, semantic filters, keyboard outline navigation, Learn/Experiment transitions, replay and same-input progression, dialog focus return, import round trips, and responsive/2D fallbacks. These checks establish that the teaching instrument works; they do not validate the source notebooks' experimental claims.

## Interaction reference

[Human Atlas](https://human-atlas-seven.vercel.app) was studied as an interaction reference for this museum redesign; its [source repository](https://github.com/ashemag/human-atlas) is credited here. Neural Atlas uses an independent implementation and original procedural geometry. No Human Atlas code, assets, or models were copied.
