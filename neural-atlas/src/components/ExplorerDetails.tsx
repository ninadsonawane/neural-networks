import { useEffect, useId, useRef, useState } from 'react';
import { ArrowDownToLine, ArrowUpRight, Check, ChevronDown, Upload, X } from 'lucide-react';
import type { Architecture, Gates, SimulationConfig, Trace } from '../simulation/types';
import { loadTraceJson } from '../simulation/adapter';
import { STAGES } from '../data/stages';
import { REPOSITORY_FACTS, REPO_URL } from '../data/repository';
import SignalInspector from './SignalInspector';
import AttentionMatrix from './AttentionMatrix';
import CompareMode from './CompareMode';
import './ExplorerDetails.css';

interface ExplorerDetailsProps {
  open: boolean;
  onClose: () => void;
  tab: 'mechanism' | 'probabilities' | 'compare' | 'about';
  trace: Trace;
  step: number;
  config: SimulationConfig;
  onConfig: (partial: Partial<SimulationConfig>) => void;
  onStep: (step: number) => void;
  decoderStep: number;
  onDecoderStep: (step: number) => void;
  onImport: (trace: Trace | null) => void;
  onExplore: (stage: Architecture) => void;
}

function DetailRange({ label, value, min, max, step = 1, onChange, suffix = '' }: {
  label: string; value: number; min: number; max: number; step?: number; onChange: (value: number) => void; suffix?: string;
}) {
  return <label className="detail-range"><span>{label}<output>{step < 1 ? value.toFixed(2) : value}{suffix}</output></span><input type="range" min={min} max={max} step={step} value={value} onChange={event => onChange(Number(event.target.value))}/></label>;
}

function DetailSwitch({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return <label className="detail-switch"><span>{label}</span><input type="checkbox" checked={checked} onChange={event => onChange(event.target.checked)}/><span className="detail-switch-track" aria-hidden="true"><i/></span></label>;
}

const SAMPLE_TRACES = [
  ['mlp', 'Fixed window'], ['rnn', 'Recurrent state'], ['lstm', 'Gated memory'],
  ['attention', 'Attention bridge'], ['transformer', 'Transformer'], ['transformer-hinglish', 'Transformer · Hinglish'],
];
const TITLES = { mechanism: 'Inside the mechanism', probabilities: 'Read the signal', compare: 'Five ways to remember', about: 'Behind the atlas' };

export default function ExplorerDetails({ open, onClose, tab, trace, step, config, onConfig, onStep, decoderStep, onDecoderStep, onImport, onExplore }: ExplorerDetailsProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const returnFocus = useRef<HTMLElement | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [importStatus, setImportStatus] = useState('');
  const [reading, setReading] = useState(false);
  const id = useId();
  const imported = trace.provenance === 'imported-trace';
  const stage = STAGES.find(item => item.id === trace.architecture) ?? STAGES[0];
  const activeStep = Math.max(0, Math.min(step, trace.steps.length - 1));
  const state = trace.steps[activeStep];
  const gateNames: (keyof Gates)[] = config.gatedVariant === 'gru' ? ['update', 'reset'] : ['forget', 'input', 'candidate', 'output'];
  const manualGates = Object.keys(config.gateOverrides ?? {}).length > 0;
  const bridge = trace.architecture === 'attention';

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open && !dialog.open) {
      returnFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
      dialog.showModal();
    } else if (!open && dialog.open) {
      dialog.close();
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = previousOverflow; };
  }, [open]);

  useEffect(() => () => {
    if (returnFocus.current?.isConnected) returnFocus.current.focus({ preventScroll: true });
  }, []);

  const restoreFocus = () => {
    if (returnFocus.current?.isConnected) returnFocus.current.focus({ preventScroll: true });
    if (open) onClose();
  };

  const exportTrace = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(trace, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = `neural-atlas-${trace.architecture}-trace.json`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  const importFile = async (file?: File) => {
    if (!file) return;
    setReading(true);
    setImportStatus('Reading trace…');
    try {
      if (file.size > 2 * 1024 * 1024) throw new Error('Choose a JSON trace no larger than 2 MB.');
      const recording = loadTraceJson(await file.text(), file.name);
      onImport(recording);
      setImportStatus(`Loaded ${file.name}. Recorded values are ready to replay; their source has not been verified.`);
    } catch (error) {
      setImportStatus(error instanceof Error ? error.message : 'The trace could not be read.');
    } finally {
      setReading(false);
    }
  };

  const configuration = <div className="detail-control-stack">
    {stage.id === 'mlp' && <><DetailRange label="Context window" value={config.contextLength} min={3} max={8} onChange={value => onConfig({ contextLength: value })} suffix=" characters"/><p className="detail-help">Only these recent characters enter the prediction. A wider hidden layer cannot recover input outside the window.</p></>}
    {stage.id === 'rnn' && <><DetailRange label="Recurrent multiplier" value={config.recurrentMultiplier} min={.1} max={1.5} step={.05} onChange={value => onConfig({ recurrentMultiplier: value })}/><div className="detail-measure"><span>First-step backward signal</span><strong>{(state?.gradients[0] ?? 1).toExponential(2)}</strong><small>After {activeStep} hops · proxy r<sup>d</sup></small></div><p className="detail-help">Below 1, repeated multiplication shrinks the signal. Above 1, it amplifies it. This scalar illustration omits the full recurrent Jacobian and tanh derivatives.</p></>}
    {stage.id === 'lstm' && <><div className="detail-segmented" aria-label="Gated memory variant"><button type="button" aria-pressed={config.gatedVariant === 'lstm'} onClick={() => onConfig({ gatedVariant: 'lstm', gateOverrides: undefined })}>LSTM</button><button type="button" aria-pressed={config.gatedVariant === 'gru'} onClick={() => onConfig({ gatedVariant: 'gru', gateOverrides: undefined })}>GRU · conceptual</button></div><DetailSwitch label="Preserve the early clue" checked={config.pinClue} onChange={value => onConfig({ pinClue: value })}/><DetailSwitch label="Override gate values" checked={manualGates} onChange={value => onConfig({ gateOverrides: value ? Object.fromEntries(gateNames.map(gate => [gate, state.gates[gate]])) : undefined })}/>{manualGates ? <div className="detail-gate-sliders">{gateNames.map(gate => <DetailRange key={gate} label={`${gate[0].toUpperCase()}${gate.slice(1)} gate`} value={config.gateOverrides?.[gate] ?? state.gates[gate]} min={gate === 'candidate' ? -1 : 0} max={1} step={.01} onChange={value => onConfig({ gateOverrides: { ...config.gateOverrides, [gate]: value } })}/>)}</div> : <dl className="detail-gate-values">{gateNames.map(gate => <div key={gate}><dt>{gate}</dt><dd>{state.gates[gate].toFixed(2)}</dd></div>)}</dl>}<p className="detail-help">{manualGates ? 'Manual values apply at every step and override automatic gates and clue preservation. Turn overrides off to restore the teaching recurrence.' : config.gatedVariant === 'gru' ? 'A GRU has two gates and one hidden state. The update gate here weights the new candidate; 1 − update preserves the old state. GRU is planned repository work.' : 'The forget gate controls the old cell state; input and candidate write new content; output controls its readout. Preservation helps memory without guaranteeing perfect gradients.'}</p></>}
    {stage.id === 'attention' && <><DetailRange label="Source sequence length" value={config.sourceLength} min={4} max={14} onChange={value => { onConfig({ sourceLength: value }); onDecoderStep(0); }} suffix=" tokens"/><DetailSwitch label="Open the attention bridge" checked={config.attentionEnabled} onChange={value => onConfig({ attentionEnabled: value })}/><p className="detail-help">{config.attentionEnabled ? 'Each target reads a new weighted mixture of source states. The alignment is a conceptual Hinglish-to-English example.' : 'The source is compressed into one fixed context. Its simulated capacity falls as the sequence grows; this is an illustration of the bottleneck, not measured translation quality.'}</p></>}
    {stage.id === 'transformer' && <><div className="detail-select-row"><label>Layer<select value={config.layer} onChange={event => onConfig({ layer: Number(event.target.value) })}>{[0, 1, 2, 3].map(layer => <option key={layer} value={layer}>{layer + 1}</option>)}</select></label><label>Attention head<select value={config.head} onChange={event => onConfig({ head: Number(event.target.value) })}>{['Local neighbors', 'Repeated characters', 'Distant clue', 'Position pattern'].map((head, index) => <option key={head} value={index}>{index + 1} · {head}</option>)}</select></label></div><DetailSwitch label="Average all four heads" checked={config.aggregateHeads} onChange={value => onConfig({ aggregateHeads: value })}/><DetailSwitch label="Include position information" checked={config.positionalEncoding} onChange={value => onConfig({ positionalEncoding: value })}/><DetailSwitch label="Mask future positions" checked={config.causalMask} onChange={value => onConfig({ causalMask: value })}/><p className="detail-help">Four hand-designed patterns demonstrate different retrieval routes. These roles are teaching choices, not learned attention heads.</p>{!config.causalMask && <p className="detail-callout">Future characters are visible. This intervention is not valid causal next-token evaluation.</p>}</>}
    <div className="detail-temperature"><DetailRange label="Temperature" value={config.temperature} min={.2} max={2} step={.1} onChange={value => onConfig({ temperature: value })}/><p className="detail-help">Lower values concentrate the distribution; higher values spread it out. The bars remain educational probabilities.</p></div>
  </div>;

  return <dialog className="detail-sheet" ref={dialogRef} aria-labelledby={`${id}-title`} aria-describedby={`${id}-intro`} onKeyDown={event => event.stopPropagation()} onCancel={event => { event.preventDefault(); onClose(); }} onClose={restoreFocus} onClick={event => { if (event.target === event.currentTarget) onClose(); }}>
    <div className="detail-shell">
      <header className="detail-header"><div><span className="detail-eyebrow">NEURAL ATLAS / A CLOSER LOOK</span><h2 id={`${id}-title`}>{TITLES[tab]}</h2></div><button type="button" className="detail-close" aria-label="Close details and return to the experiment" onClick={onClose} autoFocus><X size={21}/></button></header>
      <div className="detail-body" role="region" aria-label={`${TITLES[tab]} content`} tabIndex={0}>
        <p className="detail-intro" id={`${id}-intro`}>{tab === 'about' ? 'A visual companion to from-scratch PyTorch experiments, with a clear account of what is real and what is illustrated.' : tab === 'compare' ? 'One shared character sequence, viewed through five different mechanisms.' : `${stage.name} · ${imported ? 'recorded trace; source unverified' : 'deterministic educational simulation'}`}</p>

        {tab === 'mechanism' && <div className="detail-mechanism">
          <h3 className="detail-section-title">{stage.title}</h3><p className="detail-prose">{stage.mechanism}</p>
          <div className="detail-equation" role="math" aria-label={!imported && stage.id === 'lstm' && config.gatedVariant === 'gru' ? 'The new hidden state is one minus the update gate times the previous state, plus the update gate times the candidate.' : stage.equation}>{!imported && stage.id === 'lstm' && config.gatedVariant === 'gru' ? 'hₜ = (1 − zₜ) ⊙ hₜ₋₁ + zₜ ⊙ h̃ₜ' : stage.equation}</div>
          <p className="detail-help">{!imported && stage.id === 'lstm' && config.gatedVariant === 'gru' ? 'z weights the new candidate; 1 − z retains the old state. The reset gate changes how the old state enters the candidate.' : stage.mathNote}</p>
          <dl className="detail-explanation"><div><dt>What changed</dt><dd>{stage.fixed}</dd></div><div><dt>What remains difficult</dt><dd>{stage.limitation}</dd></div></dl>
          {imported ? <div className="detail-callout"><strong>Recorded replay</strong><p>Configuration is fixed by the supplied recording. Its export does not describe every model setting, so this sheet shows canonical architecture equations and the recorded arrays.</p><button type="button" className="detail-text-button" onClick={() => onImport(null)}>Return to educational simulation <ArrowUpRight size={14}/></button></div> : <details className="detail-disclosure"><summary>Change the model settings <ChevronDown size={16}/></summary>{configuration}</details>}
          {(stage.id === 'transformer' || bridge) && <details className="detail-disclosure detail-matrix"><summary>{bridge ? 'Inspect source-to-target alignment' : 'Inspect exact attention weights'} <ChevronDown size={16}/></summary><AttentionMatrix matrix={bridge ? trace.alignment : trace.attention} rows={bridge ? trace.decoderTokens : trace.tokens.map(token => token.text)} columns={bridge ? trace.encoderTokens : trace.tokens.map(token => token.text)} selectedRow={bridge ? Math.min(decoderStep, trace.decoderTokens.length - 1) : activeStep} onSelectRow={bridge ? onDecoderStep : onStep} label={bridge ? imported ? 'Recorded source alignment' : 'Conceptual translation alignment' : 'Self-attention weights'} causal={!imported && !bridge && config.causalMask}/>{imported && <p className="detail-help">Values are shown as recorded. Mask, layer, and head settings are not inferred from this schema.</p>}{bridge && !imported && !config.attentionEnabled && <p className="detail-help">The closed bridge shows uniform preview weights. Open it to see target-specific retrieval.</p>}</details>}
        </div>}

        {tab === 'probabilities' && <div className="detail-probabilities"><SignalInspector trace={trace} step={activeStep} xray={false} decoderStep={decoderStep}/><p className="detail-help detail-bottom-note">{imported ? 'Imported values are a replay of a supplied recording. Structural validation does not establish that a trained model produced them.' : 'These values come from a small deterministic teaching engine. They show the effect of the mechanism, rather than a trained model’s language ability.'}</p></div>}

        {tab === 'compare' && (imported ? <div className="detail-callout"><strong>This recording contains one architecture.</strong><p>Return to the educational simulation to replay the same input across all five models.</p><button type="button" className="detail-button" onClick={() => onImport(null)}>Return to simulation</button></div> : <div className="detail-compare"><CompareMode input={trace.input} step={activeStep} config={config} onExplore={architecture => { onClose(); onExplore(architecture); }}/></div>)}

        {tab === 'about' && <div className="detail-about">
          <div className="detail-provenance"><span className="detail-provenance-dot"/><div><strong>{imported ? 'Imported recording' : 'Educational simulation'}</strong><p>{imported ? trace.metadata.description : 'No trained checkpoint is running. The probabilities, embeddings, states, gradients, and attention patterns are generated by a deterministic teaching engine.'}</p></div></div>
          <a className="detail-repository-link" href={REPO_URL} target="_blank" rel="noreferrer">ninadsonawane / neural-networks <ArrowUpRight size={16}/></a>
          <h3 className="detail-section-title">The source notebooks</h3><div className="detail-facts">{REPOSITORY_FACTS.map(fact => <article key={fact.architecture}><div><h4>{fact.architecture}</h4><span>{fact.status}</span></div><p>{fact.details}</p>{fact.file && <a href={`${REPO_URL}/blob/main/${fact.file}`} target="_blank" rel="noreferrer">{fact.file} <ArrowUpRight size={12}/></a>}</article>)}</div>
          <div className="detail-import-section"><h3 className="detail-section-title">Bring a recorded trace</h3><p className="detail-prose">Export the current values or import a compatible JSON recording. A trace replays observed arrays; it does not contain a runnable model.</p><div className="detail-file-actions"><button type="button" className="detail-button" onClick={exportTrace}><ArrowDownToLine size={15}/> Export this trace</button><button type="button" className="detail-button detail-button-quiet" disabled={reading} onClick={() => fileRef.current?.click()}><Upload size={15}/>{reading ? 'Reading…' : 'Import JSON'}</button><input ref={fileRef} className="detail-file-input" type="file" accept=".json,application/json" aria-label="Import a recorded trace JSON file" onChange={event => { const file = event.target.files?.[0]; event.target.value = ''; void importFile(file); }}/></div><p className="detail-help">Maximum 2 MB · version-1 Trace schema · complete normalized probabilities and consistent tensor dimensions. The README documents the PyTorch export adapter.</p><p className="detail-import-status" role="status" aria-live="polite">{importStatus}</p>{imported && <button type="button" className="detail-text-button" onClick={() => { onImport(null); setImportStatus('Returned to educational simulation.'); }}><Check size={14}/> Return to simulation</button>}</div>
          <details className="detail-disclosure"><summary>Download sample recordings <ChevronDown size={16}/></summary><p className="detail-help">All six files contain educational simulation data. Download one and import it to try the replay workflow.</p><div className="detail-sample-links">{SAMPLE_TRACES.map(([file, label]) => <a key={file} href={`${import.meta.env.BASE_URL}traces/${file}.json`} download={`${file}.json`}>{label}<ArrowDownToLine size={14}/></a>)}</div></details>
          <p className="detail-help detail-bottom-note">Repository code was inspected, not retrained for this website. GRU and attention bridge stages are conceptual. The documented MLP loss near 1.89 is one configuration’s observation, not a comparable model ranking.</p>
        </div>}
      </div>
      <footer className="detail-footer"><span>{imported ? 'Recorded values · source unverified' : 'Illustrated mechanisms · simulated values'}</span><button type="button" onClick={onClose}>Back to the experiment <span aria-hidden="true">↗</span></button></footer>
    </div>
  </dialog>;
}
