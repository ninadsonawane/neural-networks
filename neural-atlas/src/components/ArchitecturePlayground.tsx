import { X, RotateCcw } from 'lucide-react';
import { PART_CATEGORIES } from '../explorer/types';
import type { PartCategory } from '../explorer/types';
import { PRESETS, type Sandbox } from '../explorer/cinema';
import type { Architecture } from '../simulation/types';

export default function ArchitecturePlayground({stage,value,onChange,onPreset,visible,onToggle,onClose,changed,imported,gru,onVariant}:{stage:Architecture;value:Sandbox;onChange:(key:keyof Sandbox,value:number)=>void;onPreset:(name:string)=>void;visible:PartCategory[];onToggle:(category:PartCategory)=>void;onClose:()=>void;changed:string;imported:boolean;gru:boolean;onVariant:(gru:boolean)=>void}) {
  const controls:Record<Architecture,[keyof Sandbox,string,number,number,number,string][]>={
    mlp:[['context','Context window',1,32,1,'Recent input reaches the dense path.'],['embedding','Embedding dimension',8,128,8,'Wider embedding wafers represent more channels.'],['hidden','Hidden width',16,512,16,'Wider blocks cost more space; they cannot recover excluded input.']],
    rnn:[['sequence','Sequence length',8,80,1,'Use this many characters from your input; longer routes have more handoffs.'],['hidden','Hidden-state width',16,512,16,'A wider capsule represents more capacity, not perfect recall.'],['retention','Recurrent retention',.1,1.1,.01,'Repeated multiplication weakens or amplifies the carried clue.']],
    lstm:[['hidden','Hidden width',16,512,16,'Change the illustrated memory capacity.'],['forget','Memory keep tendency',.05,.995,.005,'Higher values keep the old state; lower values let it fade.'],['sequence','Sequence length',8,80,1,'More updates put the same clue through more gates.']],
    attention:[['source','Source length',4,80,1,'Only this many earlier source positions are eligible at each step.'],['sharpness','Attention sharpness',.2,4,.1,'Sharper scores concentrate ribbons on fewer source positions.'],['head','Selected head / pattern',0,3,1,'Choose one of four illustrative retrieval patterns, not trained heads.']],
    transformer:[['layers','Layer count',1,4,1,'More repeated blocks transform representations in stages.'],['heads','Head count',1,4,1,'More parallel retrieval structures share the same input.'],['embedding','Embedding dimension',8,128,8,'Wider token representations make the stack broader.'],['context','Context length',1,32,1,'Tokens outside this attention window have no direct retrieval path.']],
  };
  return <aside className="architecture-playground glass-surface" aria-label="Architecture Playground">
    <header><div><span>TAKE THE IDEA APART</span><h2>Architecture Playground</h2></div><button className="museum-icon" aria-label="Close Architecture Playground" onClick={onClose}><X size={18}/></button></header>
    <p className="playground-note">Conceptual architecture sandbox. Shapes illustrate capacity; the teaching state stays eight-dimensional. This is not a training benchmark.</p>
    {imported?<p className="playground-note">Recorded geometry and arrays are preserved. Parameters are unavailable for imported traces.</p>:<>
      {stage==='lstm'&&<div className="playground-presets"><button aria-pressed={!gru} onClick={()=>onVariant(false)}>LSTM</button><button aria-pressed={gru} onClick={()=>onVariant(true)}>GRU</button></div>}
      <div className="playground-presets">{Object.keys(PRESETS).map(name=><button key={name} onClick={()=>onPreset(name)}>{name}</button>)}<button onClick={()=>onPreset('Reset')}><RotateCcw size={11}/> Reset</button></div>
      {controls[stage].map(([key,label,min,max,step,help])=><label className="playground-range" key={key}><span>{label}<output>{key==='head'?value[key]+1:Number(value[key].toFixed(3))}</output></span><input aria-label={label} type="range" min={min} max={max} step={step} value={value[key]} onChange={e=>onChange(key,Number(e.target.value))}/><small>{help}</small></label>)}
      <p className="playground-change" role="status">{changed||'Move a control. Watch the structure respond.'}</p>
    </>}
    <details className="playground-filters"><summary>Component filters</summary>{PART_CATEGORIES.map(c=><label key={c.id}><i style={{background:c.color}}/>{c.label}<input type="checkbox" checked={visible.includes(c.id)} onChange={()=>onToggle(c.id)}/></label>)}</details>
  </aside>;
}
