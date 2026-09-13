import type { Architecture, SimulationConfig, Trace } from '../simulation/types';
import type { ExplorerModel, ExplorerPart, Vec3 } from './types';

export interface Sandbox { context: number; embedding: number; hidden: number; sequence: number; retention: number; forget: number; source: number; sharpness: number; head: number; heads: number; layers: number }
export const DEFAULT_SANDBOX: Sandbox = { context: 8, embedding: 32, hidden: 128, sequence: 80, retention: .72, forget: .96, source: 32, sharpness: 1, head: 0, heads: 4, layers: 4 };
export const PRESETS: Record<string, Partial<Sandbox>> = {
  Tiny: { context: 3, embedding: 8, hidden: 16, sequence: 16, source: 8, heads: 1, head: 0, layers: 1 },
  'Notebook-like': { context: 8, embedding: 32, hidden: 256, sequence: 50, source: 16, heads: 4, layers: 4 },
  Wide: { embedding: 128, hidden: 512, heads: 4, layers: 4 },
  'Long context': { context: 32, sequence: 80, source: 80, retention: .96, forget: .995 },
};
export const INTUITION: Record<Architecture,string> = { mlp:'A small window. Everything else falls away.', rnn:'Carries the past. Squeezes it through one state.', lstm:'Gates decide what survives the journey.', attention:'Look back. Bring the useful clue forward.', transformer:'Many paths. One shared view of the past.' };
export const MOMENT: Record<Architecture,string> = { mlp:'Earlier clue: out of reach.', rnn:'One state must carry everything.', lstm:'Gates protect what matters.', attention:'Look back at the useful clue.', transformer:'Direct paths replace long chains.' };
export const ADDED: Record<Architecture,string> = { mlp:'A fixed window gathers nearby input.', rnn:'Added: a state to carry the past.', lstm:'Added: gates and a protected memory route.', attention:'Added: a direct route back to source.', transformer:'Added: parallel heads and residual paths.' };
export function responsiblePart(stage: Architecture, gru = false) { return ({mlp:'fixed-window',rnn:'recurrent-update',lstm:gru?'gate-update':'cell-rail',attention:'attention-context',transformer:'attention-mix'})[stage]; }
export function sandboxConfig(base: SimulationConfig, s: Sandbox, stage: Architecture): SimulationConfig {
  return {...base,contextLength:s.context,recurrentMultiplier:s.retention,sourceLength:s.source,head:stage==='attention'?s.head:Math.min(s.head,s.heads-1),headCount:s.heads,layer:s.layers-1,attentionContext:s.context,attentionSharpness:s.sharpness,customAttention:true,
    ...(stage==='lstm'?{gateOverrides:base.gatedVariant==='gru'?{...base.gateOverrides,update:1-s.forget}:{...base.gateOverrides,forget:s.forget}}:{})};
}
export function momentAt(trace: Trace, context: number) {
  const clue=trace.tokens.findIndex(t=>t.isClue);
  return Math.min(trace.tokens.length-1,Math.max(1,clue+(trace.architecture==='mlp'?context:Math.min(10,Math.ceil(trace.tokens.length*.55)))));
}
export function processingRoute(model: ExplorerModel, step: number) {
  const ids=new Set(model.parts.map(p=>p.id));
  const stage=model.architecture;
  const route=stage==='mlp'?[`token-${step}`,`embedding-${step}`,'window-join','dense-projection','dense-state','output']:
    stage==='rnn'?[`token-${step}`,`embedding-${step}`,`state-${step}`,'output']:
    stage==='lstm'?[`token-${step}`,'current-embedding',...(ids.has('cell-rail')?['gate-input','cell-rail','cell-current','gate-output']:['gate-reset','gru-candidate','gate-update']),'hidden-current','output']:model.flow;
  return route.filter((id,index)=>ids.has(id)&&route.indexOf(id)===index);
}
export function cinematicModel(base: ExplorerModel, s: Sandbox, trace: Trace, step: number, reveal=false): ExplorerModel {
  const model:ExplorerModel={...base,parts:base.parts.map(p=>({...p,position:[...p.position],explodedPosition:[...p.explodedPosition],scale:[...p.scale]})),links:[...base.links],flow:[...base.flow]};
  const simulated=trace.provenance==='educational-simulation';
  const labelMap:Record<string,string>={'window-join':'Embedding','dense-state':'Hidden state hₜ','output':'Output','current-embedding':'Embedding','cell-rail':'Cell state cₜ','hidden-current':'Hidden state hₜ','gate-forget':'Forget gate','gate-input':'Input gate','gate-output':'Output gate','gate-update':'Update gate','gate-reset':'Reset gate','token-embeddings':'Embedding','attention-mix':'Attention','attention-context':'Attention','encoder-readout':'Embedding'};
  model.parts.forEach(p=>{
    p.sceneLabel=labelMap[p.id]??(p.id.endsWith('-q')?'Query':p.id.endsWith('-k')?'Key':p.id.endsWith('-v')?'Value':p.id===`state-${step}`?'Hidden state hₜ':undefined);
    if(p.kind==='card'&&p.category==='tokens'){p.color='#d9d2bd';p.clue=p.tokenIndex!==undefined&&!!trace.tokens[p.tokenIndex]?.isClue;}
    if(simulated&&trace.metadata.description.includes('User-character source')){
      if(p.id.startsWith('source-')&&p.category==='tokens')p.purpose=`Source character ${(p.tokenIndex??0)+1}: “${p.label}”, from your own input. This teaching alignment retrieves eligible earlier characters.`;
      if(p.decoderIndex!==undefined){p.purpose=`Output position ${p.decoderIndex+1} in your sequence. Its normalized alignment chooses a source context; this is replay, not generated text.`;if(p.decoderIndex===step&&p.category==='output')p.sceneLabel='Output';}
    }
    if(base.architecture==='rnn'&&p.id.startsWith('state-')&&p.tokenIndex!==undefined)p.echo=Math.min(1,trace.steps[step].gradients[p.tokenIndex]??0);
    if(base.architecture==='transformer'&&p.category==='tokens'&&p.tokenIndex!==undefined)p.signal=trace.attention[step]?.[p.tokenIndex]??0;
    if(simulated){
      const factor=p.category==='embeddings'?Math.sqrt(s.embedding/32):p.category==='memory'&&!p.id.includes('rail')?Math.sqrt(s.hidden/128):1;
      p.scale=[p.scale[0]*Math.min(1.5,Math.max(.6,factor)),p.scale[1],p.scale[2]*Math.min(1.35,Math.max(.6,factor))];
      if(p.category==='memory'||p.category==='embeddings')p.cells=Math.min(49,Math.max(4,Math.round((p.category==='memory'?s.hidden:s.embedding)/8)));
      if(p.id==='fixed-window')p.scale[0]*=1+s.context/48;
    }
  });
  const input=model.parts.find(p=>p.id===`token-${step}`)??model.parts.find(p=>p.category==='tokens'&&!p.ghost);
  if(input)input.sceneLabel='Input tokens';
  if(simulated&&(base.architecture==='rnn'||base.architecture==='lstm')){
    for(let i=0;i<Math.ceil(trace.tokens.length/8);i++)model.parts.push({id:`sequence-deck-${i}`,label:`Input bundle ${i+1}`,sceneLabel:i===0?`${trace.tokens.length} input tokens`:undefined,category:'tokens',kind:'slab',position:[-2.6,-1.9+i*.065,-.5],explodedPosition:[-4.5,-2.5+i*.16,-.2],scale:[.6,.045,.46],color:'#acae9b',purpose:`This small stack stands for input positions ${i*8+1}–${Math.min(trace.tokens.length,i*8+8)}.`,why:'Longer input adds more sequential updates. The active token follows its individual route.'});
  }
  if(base.architecture==='mlp')model.parts.push({id:'memory-horizon',label:'Memory horizon',category:'memory',kind:'slab',position:[0,-1.55,-.9],explodedPosition:[-3.9,-1.6,-.5],scale:[2.8,.018,.7],ghost:true,color:'#af9771',purpose:'Earlier cards fall behind the fixed input window.',why:'Excluded input has no route into this prediction.'});
  if(model.architecture==='transformer'&&simulated){
    model.parts=model.parts.filter(p=>{const match=(p.parentId??p.id).match(/^head-(\d+)$/);return !match||Number(match[1])<s.heads;});
    if(reveal){
      const head=`head-${Math.min(s.head,s.heads-1)}`;
      const parent=model.parts.find(p=>p.id===head)!;
      ['q','k','v','weights'].forEach((suffix,i)=>{const child=model.parts.find(p=>p.id===`${head}-${suffix}`);if(child){child.detailOnly=false;child.position=[parent.position[0]+(i-1.5)*.57,parent.position[1]-.55,1.2];child.explodedPosition=[(i-1.5)*.85,-.8,1.4];}});
      model.links.push({id:'query-beam',from:'position-encoding',to:`${head}-q`,kind:'attention'},{id:'key-beam',from:'position-encoding',to:`${head}-k`,kind:'attention'},{id:'value-beam',from:'position-encoding',to:`${head}-v`,kind:'attention'},{id:'query-weights',from:`${head}-q`,to:`${head}-weights`,kind:'attention'},{id:'key-weights',from:`${head}-k`,to:`${head}-weights`,kind:'attention'},{id:'value-weights',from:`${head}-v`,to:`${head}-weights`,kind:'attention'},{id:'weights-readout',from:`${head}-weights`,to:'attention-mix',kind:'attention'});
      const index=model.flow.indexOf(head);if(index>=0)model.flow.splice(index,1,`${head}-q`,`${head}-weights`);
    }
    const layers:ExplorerPart[]=[];
    for(let i=1;i<s.layers;i++)layers.push({id:`layer-${i}`,label:`Transformer layer ${i+1}`,sceneLabel:i===s.layers-1?`${s.layers} layers`:undefined,kind:'slab',category:'attention',position:[0,1.05+i*.23,-.14],explodedPosition:[0,2.05+i*.48,-.15],scale:[2.1,.14,.95],color:'#557a83',purpose:'A repeated attention and feed-forward block in this conceptual stack.',why:'Each layer can build a new representation from earlier transformations.'});
    const output=model.parts.find(p=>p.id==='output');if(output){output.position[1]+=s.layers*.23;output.explodedPosition[1]+=s.layers*.48;}
    model.parts.push(...layers);
    layers.forEach((p,i)=>model.links.push({id:`layer-flow-${i}`,from:i?layers[i-1].id:'feed-forward',to:p.id,kind:'flow'}));
    if(layers.length){model.links=model.links.filter(l=>!(l.from==='feed-forward'&&l.to==='output'));model.links.push({id:'stack-output',from:layers.at(-1)!.id,to:'output',kind:'flow'});model.flow.splice(model.flow.indexOf('output'),0,...layers.map(p=>p.id));}
  }
  if(model.architecture==='lstm'&&trace.tokens.some(t=>t.isClue)){
    const rail=model.parts.find(p=>p.id==='cell-rail'||p.id==='hidden-rail');
    if(rail)model.parts.push({id:'clue-bookmark',label:'Clue bookmark',sceneLabel:'Clue',category:'memory',kind:'card',position:[rail.position[0]-1.1,rail.position[1]+.16,rail.position[2]+.25],explodedPosition:[rail.explodedPosition[0]-1.1,rail.explodedPosition[1]+.2,rail.explodedPosition[2]+.25],scale:[.17,.48,.035],color:'#e5cc88',value:trace.steps[step].clueStrength,clue:true,purpose:'Brightness follows the recorded clue-retention scalar at this step.',why:'This bookmark makes preservation visible. It is a teaching proxy, not proof that a model understands a clue.'});
  }
  const existing=new Set(model.parts.map(p=>p.id));model.links=model.links.filter(l=>existing.has(l.from)&&existing.has(l.to));
  model.flow=processingRoute(model,step);
  return model;
}
export const semanticSeed=(part:ExplorerPart):Vec3=>part.category==='tokens'?[0,-2.4,0]:part.category==='output'?[0,1.8,0]:[0,0,0];
