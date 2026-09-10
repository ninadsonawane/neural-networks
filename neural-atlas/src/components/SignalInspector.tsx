import { Activity, ArrowUpRight, ChevronDown } from 'lucide-react';
import type { Trace } from '../simulation/types';
import { displayToken } from '../simulation/engine';

interface SignalInspectorProps {
  trace: Trace;
  step: number;
  xray: boolean;
  decoderStep?: number;
}

const tokenName = (token: string) => token === ' ' ? 'space' : token === '\n' ? 'newline' : token === '\t' ? 'tab' : token;
const compactNumber = (value: number) => Math.abs(value) >= 10000 || (value !== 0 && Math.abs(value) < .001) ? value.toExponential(2) : value.toFixed(3);

export default function SignalInspector({ trace, step, xray, decoderStep }: SignalInspectorProps) {
  const state = trace.steps[step];
  if (!state) return null;
  const imported = trace.provenance === 'imported-trace';
  const isBridge = trace.architecture === 'attention';
  const decoderIndex = Math.max(0, Math.min(decoderStep ?? step % Math.max(1, trace.decoderTokens.length), trace.decoderTokens.length - 1));
  const retained = isBridge ? trace.encoderTokens.join(' ') : state.retainedIndices.map(i => trace.tokens[i]?.text ?? '').join('');
  const retainedLabel = isBridge ? 'ENCODER SOURCE TOKENS' : imported ? 'RECORDED RETAINED INDICES' : trace.architecture === 'mlp' ? 'ACCESSIBLE CONTEXT' : 'RETENTION PROXY';
  const retainedNote = isBridge ? `${trace.encoderTokens.length} ${imported ? 'recorded' : 'conceptual'} source tokens` : imported ? `${state.retainedIndices.length} positions annotated by the exporter` : trace.architecture === 'mlp' ? `${state.retainedIndices.length} positions in the current window` : `${state.retainedIndices.length} positions above the display threshold`;
  const chartValues = state.hidden.slice(0, 32);
  const chartScale = state.hidden.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 0) || 1;
  const connectionWeights = isBridge ? trace.alignment[decoderIndex] : trace.attention[step];
  const connectionTokens = isBridge ? trace.encoderTokens : trace.tokens.map(token => token.text);

  return <aside className="inspector" aria-label="Signal inspector">
    <div className="inspector-heading"><Activity size={14}/><span>Signal inspector</span><span className="live-dot"/></div>
    <div className="inspector-token"><span className="micro">CURRENT CHARACTER · SHARED TIMELINE</span><div><strong title={tokenName(trace.tokens[step].text)}><span aria-hidden="true">{displayToken(trace.tokens[step].text)}</span><span className="sr-only">{tokenName(trace.tokens[step].text)}</span></strong><span>position {String(step + 1).padStart(2, '0')} <ArrowUpRight size={15}/></span></div></div>
    <div className="prediction-heading"><span>{imported ? 'RECORDED NEXT SYMBOL' : 'ILLUSTRATIVE NEXT CHARACTER'}</span><span>p(x)</span></div>
    <div className="probabilities" aria-label={`Top five ${imported ? 'recorded' : 'illustrative'} next-symbol probabilities`}>
      {state.probabilities.slice(0, 5).map((p, i) => <div className="probability" key={p.token} title={`${tokenName(p.token)}: ${p.probability} probability`}>
        <span className="prob-token"><span aria-hidden="true">{displayToken(p.token)}</span><span className="sr-only">{tokenName(p.token)}</span></span>
        <div className="prob-track" aria-hidden="true"><div style={{ width: `${p.probability * 100}%`, opacity: 1 - i * .13 }}/></div>
        <span>{(p.probability * 100).toFixed(1)}<small>%</small><span className="sr-only">; exact probability {p.probability}</span></span>
      </div>)}
    </div>
    <p className="tiny muted">Top 5 of {trace.vocabulary.length} symbols · {imported ? 'recorded distribution; source unverified' : 'illustrative distribution'}</p>
    <div className="retained"><span className="micro">{retainedLabel}</span><code>{retained || '∅'}</code><span className="tiny muted">{retainedNote}</span></div>
    {(trace.architecture === 'rnn' || trace.architecture === 'lstm' || xray) && <details className="state-details" open={xray}>
      <summary>Inside the state <ChevronDown size={13}/></summary>
      <div className="norm-row"><span title={`Hidden-state Euclidean norm ${state.hiddenNorm}`}>‖h‖ {compactNumber(state.hiddenNorm)}</span><span title={`Cell-state Euclidean norm ${state.cellNorm}`}>‖c‖ {compactNumber(state.cellNorm)}</span></div>
      <div className="state-bars" role="img" aria-label={`Signed hidden-state bars. Positive values extend up, negative values down. Scale ${chartScale}. ${chartValues.map((value, index) => `h ${index}: ${value}`).join('; ')}`} style={{ backgroundImage: 'linear-gradient(to bottom, transparent 49%, #526a75 49%, #526a75 51%, transparent 51%)' }}>
        {chartValues.map((value, index) => <div key={index} title={`h[${index}] = ${value}`} style={{ flex: 1, minWidth: 1, maxWidth: 14 }}><i style={{ position: 'absolute', height: `${Math.abs(value) / chartScale * 46}%`, bottom: value >= 0 ? '50%' : undefined, top: value < 0 ? '50%' : undefined, background: value < 0 ? 'var(--orange)' : 'var(--cyan)' }}/></div>)}
      </div>
      <p className="tiny muted" style={{ fontSize: 8 }}>±{compactNumber(chartScale)} scale · above/below zero{state.hidden.length > 32 ? ` · first 32 of ${state.hidden.length}` : ''}</p>
      <details><summary style={{ fontSize: 9, marginTop: 8 }}>Read state values</summary><code tabIndex={0} aria-label="Hidden state values" style={{ display: 'block', maxHeight: 120, overflow: 'auto', overflowWrap: 'anywhere', whiteSpace: 'pre-wrap', fontSize: 9, lineHeight: 1.7, marginTop: 8 }}>{state.hidden.map((value, index) => `h[${index}] = ${value}`).join('\n')}</code></details>
      {trace.architecture === 'lstm' && <div className="gate-readout">{Object.entries(state.gates).map(([key, value]) => <span key={key}>{key}<b>{value.toFixed(2)}</b></span>)}</div>}
    </details>}
    {(trace.architecture === 'transformer' || isBridge) && <div className="selected-weights">
      {isBridge && <><span className="micro">SELECTED DECODER TOKEN</span><div><span>{trace.decoderTokens[decoderIndex]} <small>at {decoderIndex + 1}</small></span></div><p className="tiny muted" style={{ fontSize: 8 }}>{imported ? 'Recorded alignment' : 'Conceptual word alignment'} · selected independently of the character stream</p></>}
      <span className="micro">{isBridge ? 'STRONGEST SOURCE WEIGHTS' : 'STRONGEST ATTENTION WEIGHTS'}</span>
      {connectionWeights?.map((weight, i) => ({ weight, i })).sort((a, b) => b.weight - a.weight).slice(0, 3).map(value => <div key={value.i} title={`Source ${value.i + 1}: ${tokenName(connectionTokens[value.i])}, probability ${value.weight}`}><span>{displayToken(connectionTokens[value.i])} <small>at {value.i + 1}</small></span><code>{(value.weight * 100).toFixed(1)}%<span className="sr-only">; exact probability {value.weight}</span></code></div>)}
    </div>}
    <div className="inspector-foot"><span className="square-dot"/>{imported ? 'Imported trace · source unverified' : 'Simulated values. Real principles.'}</div>
  </aside>;
}
