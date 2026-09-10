import { Component, Suspense, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Line, RoundedBox } from '@react-three/drei';
import { CanvasTexture, Group, Mesh, SRGBColorSpace } from 'three';
import type { SimulationConfig, Trace } from '../simulation/types';
import './ArchitectureScene.css';

export interface ArchitectureSceneProps {
  trace: Trace; step: number; config: SimulationConfig; reducedMotion: boolean;
  focusMode: boolean; bptt: boolean; trainingPass: boolean; onSelectStep: (index: number) => void;
  decoderStep?: number; onSelectDecoderStep?: (index: number) => void;
}
type Point = [number, number, number];
type Node = { id: string; p: Point; label: string; color: string; labelColor?: string; kind: 'tile' | 'orb' | 'gate' | 'capsule'; opacity?: number; selected?: boolean; token?: number; decoder?: boolean; size?: number };
type Edge = { id: string; points: Point[]; color: string; weight?: number; opacity?: number; dashed?: boolean };
type Label = { id: string; p: Point; text: string; color?: string; size?: number };
type Region = { id: string; x: number; y: number; w: number; h: number; color: string; dashed?: boolean; label?: string };
type SceneSpec = { nodes: Node[]; edges: Edge[]; labels: Label[]; regions: Region[]; caption: string; state: string };
const C = { cyan: '#67e4dc', ivory: '#e4e9df', teal: '#3f8d8c', dim: '#294b54', amber: '#e7b776', muted: '#789597', violet: '#a4a5df' };
const pt = (x: number, y: number, z = 0): Point => [x, y, z];
const readable = (s: string) => s === ' ' ? '␣' : s === '\n' ? '↵' : s === '\t' ? '⇥' : s;
const clamp = (n: number, a = 0, b = 1) => Math.max(a, Math.min(b, n));
const spaced = (i: number, count: number, width = 8.3) => count <= 1 ? 0 : (i / (count - 1) - .5) * Math.min(width, (count - 1) * .66);
function arc(a: Point, b: Point, rise = .7): Point[] {
  return Array.from({ length: 25 }, (_, i) => { const t = i / 24; return pt(a[0] * (1-t) + b[0]*t, a[1]*(1-t)+b[1]*t + Math.sin(Math.PI*t)*rise, a[2]*(1-t)+b[2]*t + Math.sin(Math.PI*t)*.09); });
}
function arrowPoints(edge:Edge):Point[]|null {
  if(!/^(relay|highway|grad)/.test(edge.id)) return null;
  const end=edge.points[edge.points.length-1],before=edge.points[edge.points.length-2];
  const dx=end[0]-before[0],dy=end[1]-before[1],length=Math.hypot(dx,dy)||1;
  const ux=dx/length,uy=dy/length,s=.08;
  return [pt(end[0]-ux*s-uy*s*.55,end[1]-uy*s+ux*s*.55,end[2]),end,pt(end[0]-ux*s+uy*s*.55,end[1]-uy*s-ux*s*.55,end[2])];
}

function makeScene({ trace, step, config, bptt, trainingPass, decoderStep }: Pick<ArchitectureSceneProps,'trace'|'step'|'config'|'bptt'|'trainingPass'|'decoderStep'>): SceneSpec {
  const active = Math.max(0, Math.min(step, trace.tokens.length - 1));
  const summary = trace.steps[active];
  const nodes: Node[] = [], edges: Edge[] = [], labels: Label[] = [], regions: Region[] = [];
  let caption = '', state = '';
  const recorded=trace.provenance==='imported-trace';
  const first = Math.max(0, Math.min(active - 10, trace.tokens.length - 14));
  const tokens = trace.tokens.slice(first, first + 14);
  const distantClue=trace.tokens.find(t=>t.isClue&&t.position<first);
  if(trace.architecture==='transformer'&&distantClue) tokens.splice(0,1,distantClue);
  const x = (i: number) => spaced(i, tokens.length);
  const addLabel = (id: string, px: number, py: number, text: string, color = C.muted, size = .12) => labels.push({ id, p: pt(px, py, .1), text, color, size });
  const addTile = (i: number, y: number, opacity = 1) => {
    const token = tokens[i], selected = token.position === active;
    nodes.push({ id: `t${token.position}`, p: pt(x(i), y, selected ? .16 : 0), label: readable(token.text), color: selected || trainingPass ? C.cyan : token.position > active ? C.dim : C.teal, labelColor: selected||trainingPass?C.cyan:token.position>active?C.muted:C.ivory, kind: 'tile', token: token.position, selected, opacity });
    addLabel(`p${i}`, x(i), y - .5, String(token.position + 1).padStart(2, '0'), token.isClue ? C.amber : C.muted, .1);
  };
  if (trace.architecture === 'mlp') {
    const retained = new Set(summary?.retainedIndices ?? []);
    const visible = tokens.map((t,i) => retained.has(t.position) ? i : -1).filter(i => i >= 0);
    tokens.forEach((t,i) => addTile(i, -.95, t.position > active ? .32 : retained.has(t.position) ? 1 : .45));
    if (visible.length) {
      const left = x(visible[0]) - .32, right = x(visible[visible.length - 1]) + .32;
      regions.push({ id: 'window', x: left, y: -.95, w: right-left, h: .79, color: C.cyan, label: recorded?`${retained.size}-token recorded context`:`${config.contextLength}-token context window` });
    }
    for (let layer = 0; layer < 3; layer++) {
      const count = layer === 2 ? 3 : 5;
      for (let n=0; n<count; n++) {
        const nx = (layer - 1)*1.5, ny = .36+n*.25-(count-1)*.125;
        nodes.push({ id: `h${layer}-${n}`, p: pt(nx, ny, .1 + layer*.08), label: '', kind:'orb', color: layer === 2 ? C.cyan : C.teal, size: .075 });
        if (layer > 0) for (let prev=0; prev<5; prev++) edges.push({ id:`dense${layer}-${n}-${prev}`, points:[pt((layer-2)*1.5, .36+prev*.25-.5,.1+(layer-1)*.08),pt(nx,ny,.1+layer*.08)], color:C.teal, opacity:.23, weight:.65 });
      }
    }
    visible.forEach((i) => edges.push({id:`input${i}`,points:arc(pt(x(i),-.56),pt(-1.5,.0,.1),.05),color:C.cyan,opacity:.35,weight:.8}));
    addLabel('input-label', -3.1, -.16, 'CONCATENATE', C.muted, .105);
    addLabel('dense-label', .02, 1.3, 'DENSE LAYERS  →  NEXT TOKEN', C.ivory, .11);
    const clueIndex = tokens.findIndex(t => t.isClue && !retained.has(t.position) && t.position <= active);
    if (clueIndex >= 0) {
      regions.push({ id:'lost-clue', x:x(clueIndex)-.31,y:-.95,w:.62,h:.78,color:C.amber,dashed:true });
      addLabel('lost-label', Math.max(-3.4,x(clueIndex)), -.15, 'OUT OF REACH',C.amber,.095);
    } else {
      const clue=trace.tokens.find(t=>t.isClue&&!retained.has(t.position)&&t.position<=active);
      if(clue) {
        nodes.push({id:'pinned-clue',p:pt(-3.6,.72),kind:'tile',label:readable(clue.text),color:C.amber,opacity:.65,token:clue.position});
        regions.push({id:'pinned-clue-outline',x:-3.94,y:.72,w:.68,h:.81,color:C.amber,dashed:true});
        edges.push({id:'unreachable-clue',points:[pt(-3.19,.72),pt(-2.7,.72)],color:C.amber,dashed:true,opacity:.55});
        addLabel('pinned-clue-label',-3.6,1.43,'CLUE OUT OF REACH',C.amber,.09);
      }
    }
    const embeddingChars=[...new Set(tokens.filter(t=>t.position<=active).map(t=>t.text))].slice(-8);
    const projection=(character:string):Point=>{
      const code=character.toLowerCase().codePointAt(0)??32;
      return pt(3.32+(.65*Math.sin(code*.173)+.35*Math.cos(code*.071))*.63,.41+(.65*Math.sin(code*2*.173)+.35*Math.cos(code*.071+1.7))*.39,.12);
    };
    edges.push({id:'embedding-axis-x',points:[pt(2.49,.38),pt(4.17,.38)],color:C.teal,opacity:.19});
    edges.push({id:'embedding-axis-y',points:[pt(3.32,-.16),pt(3.32,.95)],color:C.teal,opacity:.19});
    embeddingChars.forEach((character,i)=>{
      const point=projection(character),isActive=character===trace.tokens[active]?.text;
      nodes.push({id:`embedding${i}`,p:point,kind:'orb',label:'',color:isActive?C.cyan:C.teal,size:isActive?.045:.03});
      addLabel(`embedding-char${i}`,point[0]+.13,point[1]+.1,readable(character),isActive?C.cyan:C.muted,.09);
    });
    addLabel('embedding-title',3.32,1.25,'EMBEDDING SPACE',C.ivory,.095);
    addLabel('embedding-disclaimer',3.32,1.03,'ILLUSTRATIVE · 2D PROJECTION',C.muted,.075);
    caption = recorded?'The highlighted tokens are the recorded retained context for this step.':'Only the most recent context-window tokens enter the dense network. Older tokens fall outside its receptive field.';
    state = `${visible.length} visible in window`;
  } else if (trace.architecture === 'rnn') {
    tokens.forEach((t,i) => {
      const done = t.position <= active;
      addTile(i,-1.03,done ? .95 : .28);
      nodes.push({id:`h${i}`,p:pt(x(i),.12,.16),kind:'orb',label: `h${t.position+1}`,color:t.position === active ? C.cyan : done ? C.teal : C.dim,size:.21,opacity:done ? 1 : .35,selected:t.position===active});
      edges.push({id:`feed${i}`,points:[pt(x(i),-.65),pt(x(i),-.11,.16)],color:done?C.teal:C.dim,opacity:.65});
      if(i) edges.push({id:`relay${i}`,points:[pt(x(i-1)+.24,.12,.16),pt(x(i)-.24,.12,.16)],color:done?C.cyan:C.dim,opacity:done?.7:.35,weight:1.6});
      if(bptt && i && done) {
        const magnitude=Math.abs(summary?.gradients[t.position] ?? Math.pow(config.recurrentMultiplier,active-t.position));
        const strength=clamp(magnitude),width=magnitude>1?3+Math.min(7,Math.log10(magnitude)*2):.5+strength*2.5;
        edges.push({id:`grad${i}`,points:arc(pt(x(i),.43,.13),pt(x(i-1),.43,.13),.4),color:strength<.18||magnitude>1.1?C.amber:C.violet,opacity:.13+strength*.87,weight:width});
      }
    });
    addLabel('shared',0,1.24,bptt?'BACKPROPAGATION THROUGH TIME  ←':'ONE STATE. REUSED AT EVERY STEP.',bptt?C.violet:C.ivory,.12);
    addLabel('relay-label',0,-.43,'INPUT + PREVIOUS STATE',C.muted,.105);
    caption = bptt ? `${recorded?'Recorded backward values control':'The r-to-the-distance teaching proxy controls'} the backward path. Thin amber paths fade; thick amber paths amplify.${recorded?'':' This simulation does not compute the full recurrent Jacobian.'}` : 'Each input updates a shared hidden state. The glowing relay carries a compressed history forward.';
    state = bptt ? 'backward signal visible' : 'shared recurrent weights';
  } else if (trace.architecture === 'lstm') {
    const gru = !recorded&&config.gatedVariant === 'gru';
    const displayed = tokens.slice(Math.max(0, tokens.findIndex(t=>t.position===active)-3), Math.max(0, tokens.findIndex(t=>t.position===active)-3)+5);
    const gates = summary?.gates;
    const cellX=(i:number)=>(i-(displayed.length-1)/2)*1.74;
    displayed.forEach((t,i) => {
      const nx=cellX(i), done=t.position<=active, selected=t.position===active;
      nodes.push({id:`gt${i}`,p:pt(nx,-1.16),label:readable(t.text),kind:'tile',color:selected?C.cyan:done?C.teal:C.dim,token:t.position,selected,opacity:done?1:.3});
      nodes.push({id:`gh${i}`,p:pt(nx,-.02,.15),label:`h${t.position+1}`,kind:'orb',color:selected?C.cyan:C.teal,size:.19,opacity:done?1:.3});
      nodes.push({id:`cell${i}`,p:pt(nx,1,.2),label:gru?'h':'c',kind:'capsule',color:selected?C.cyan:C.teal,size:.34,opacity:done?1:.3});
      edges.push({id:`cell-feed${i}`,points:[pt(nx,-.8),pt(nx,-.22,.15)],color:C.teal,opacity:done?.65:.2});
      edges.push({id:`cell-out${i}`,points:[pt(nx, .8,.2),pt(nx,.18,.15)],color:done?C.cyan:C.dim,opacity:.7});
      if(i) {
        const retention=clamp(summary?.gradients[t.position]??0);
        edges.push({id:`highway${i}`,points:[pt(cellX(i-1)+.36,1,.2),pt(nx-.36,1,.2)],color:done?C.cyan:C.dim,weight:done?1+retention*3:1,opacity:done?.22+retention*.7:.25});
      }
      if(selected) {
        const names = gru ? ['r','z'] : ['f','i','g̃','o'];
        const values = gru ? [gates?.reset??0,gates?.update??0] : [gates?.forget??0,gates?.input??0,gates?.candidate??0,gates?.output??0];
        names.forEach((name,j) => {
          const gx=nx+(j-(names.length-1)/2)*.41;
          nodes.push({id:`gate${j}`,p:pt(gx,.49,.3),label:name,kind:'gate',color:values[j]<.35?C.amber:C.cyan,size:.16});
          addLabel(`gatevalue${j}`,gx,.22,values[j].toFixed(2),C.ivory,.085);
        });
      }
      addLabel(`gindex${i}`,nx,-1.64,String(t.position+1).padStart(2,'0'),C.muted,.1);
    });
    const clueStrength=clamp(summary?.clueStrength??0),activeCell=displayed.findIndex(t=>t.position===active);
    if(activeCell>=0&&trace.tokens.some(t=>t.isClue&&t.position<=active)) {
      nodes.push({id:'clue-memory',p:pt(cellX(activeCell),1.32,.3),kind:'orb',label:'',color:C.amber,size:.045+clueStrength*.06,opacity:.15+clueStrength*.85});
      addLabel('clue-memory-label',2.7,1.64,`CLUE ${Math.round(clueStrength*100)}% RETAINED`,C.amber,.088);
    }
    addLabel('highway-title',-1.1,1.64,recorded?'RECORDED MEMORY STATES':gru?'GATED HIDDEN-STATE HIGHWAY':'CELL-STATE HIGHWAY',C.ivory,.11);
    addLabel('gates-title',0,-.55,gru?'RESET  ·  UPDATE':'FORGET  ·  WRITE  ·  READ',C.muted,.105);
    caption=gru?'Reset and update gates control how new input changes the hidden state. The GRU uses one memory stream.':'A separate cell state carries memory along the upper highway. Forget, input, and output gates regulate what is kept, written, and read.';
    state=recorded?'recorded states and gates':gru?'2 gates · 1 memory stream':'3 gates · 2 memory streams';
  } else if (trace.architecture === 'attention') {
    const target=Math.min(trace.decoderTokens.length-1,Math.max(0,decoderStep??active%Math.max(1,trace.decoderTokens.length)));
    const targetWeights=trace.alignment[target]??[];
    const strongestSource=targetWeights.reduce((best,weight,index)=>weight>(targetWeights[best]??-1)?index:best,0);
    const encoderStart=Math.max(0,Math.min(strongestSource-7,trace.encoderTokens.length-16));
    const decoderStart=Math.max(0,Math.min(target-7,trace.decoderTokens.length-16));
    const encoder=trace.encoderTokens.slice(encoderStart,encoderStart+16);
    const decoder=trace.decoderTokens.slice(decoderStart,decoderStart+16);
    const targetLocal=target-decoderStart;
    const sourceSubset=encoder.length<trace.encoderTokens.length;
    const targetSubset=decoder.length<trace.decoderTokens.length;
    const enabled=recorded||config.attentionEnabled;
    const wordX=(i:number,count:number)=>(i-(count-1)/2)*Math.min(1.02,9.2/Math.max(1,count));
    const wordWidth=(count:number)=>Math.min(.82,9.2/Math.max(1,count)-.07);
    encoder.forEach((t,i) => {
      const sourceIndex=encoderStart+i,nx=wordX(i,encoder.length);
      nodes.push({id:`enc${sourceIndex}`,p:pt(nx,1.05),label:readable(t),kind:'tile',color:C.teal,size:wordWidth(encoder.length)});
      nodes.push({id:`encstate${sourceIndex}`,p:pt(nx,.45,.14),label:'',kind:'orb',color:C.teal,size:.08});
      edges.push({id:`encfeed${sourceIndex}`,points:[pt(nx,.8),pt(nx,.52,.14)],color:C.teal,opacity:.55});
      if(enabled) {
        const weight=targetWeights[sourceIndex]??1/Math.max(1,trace.encoderTokens.length);
        edges.push({id:`align${sourceIndex}`,points:arc(pt(nx,.35,.14),pt(wordX(targetLocal,decoder.length),-.55,.2),.03),color:weight>.13?C.cyan:C.teal,weight:.5+weight*9,opacity:.15+weight*.85});
      } else edges.push({id:`compress${sourceIndex}`,points:[pt(nx,.35,.14),pt(0,-.05,.15)],color:C.amber,opacity:.28,weight:.8});
    });
    decoder.forEach((t,i) => {
      const targetIndex=decoderStart+i;
      nodes.push({id:`dec${targetIndex}`,p:pt(wordX(i,decoder.length),-1,.1),label:readable(t),kind:'tile',color:targetIndex===target?C.cyan:targetIndex<target?C.teal:C.dim,selected:targetIndex===target,token:targetIndex,decoder:true,opacity:targetIndex<=target?1:.38,size:wordWidth(decoder.length)});
    });
    if(!enabled) {
      nodes.push({id:'bottleneck',p:pt(0,-.04,.3),label:'context',kind:'capsule',color:C.amber,size:.62});
      edges.push({id:'bottleneck-out',points:[pt(0,-.24,.2),pt(wordX(targetLocal,decoder.length),-.67,.1)],color:C.amber,weight:2});
      for(let i=0;i<encoder.length;i++) nodes.push({id:`pressure${i}`,p:pt(-.51+i*.073,-.3,.31),label:'',kind:'orb',color:C.amber,size:.018,opacity:.45+encoder.length/32});
      addLabel('pressure-label',2.65,-.05,`${encoder.length} STATES → 1 VECTOR`,C.amber,.11);
    }
    addLabel('encoder-title',0,1.7,sourceSubset?`SOURCE ${encoderStart+1}–${encoderStart+encoder.length} / ${trace.encoderTokens.length} · SUBSET`:'ENCODER · SOURCE MEMORY',C.ivory,.11);
    addLabel('decoder-title',0,-1.7,targetSubset?`TARGET ${decoderStart+1}–${decoderStart+decoder.length} / ${trace.decoderTokens.length}`:'DECODER · TARGET SEQUENCE',C.ivory,.11);
    caption=enabled?'The active decoder token reads a weighted blend of encoder states. Brighter, thicker strands carry more attention.':'All encoder states must compress into one fixed context vector. Switch attention on to reopen direct paths to the source.';
    state=recorded?'recorded source alignment':enabled?'dynamic source alignment':'fixed context bottleneck';
    if(sourceSubset) {
      const visibleMass=targetWeights.slice(encoderStart,encoderStart+encoder.length).reduce((sum,weight)=>sum+weight,0);
      caption+=` Showing source positions ${encoderStart+1} through ${encoderStart+encoder.length}, centered near the strongest contributor, carrying ${(visibleMass*100).toFixed(1)}% of the attention weight. Subset weights are not renormalized; the full matrix below contains all source positions.`;
      state='source subset · full matrix below';
    }
    if(targetSubset) caption+=` Decoder positions ${decoderStart+1} through ${decoderStart+decoder.length} are shown around selected target ${target+1}.`;
  } else {
    const selected=tokens.findIndex(t=>t.position===active);
    const sx=x(Math.max(0,selected));
    const causal=recorded?trace.attention.every((row,q)=>row.every((weight,k)=>k<=q||weight===0)):config.causalMask;
    tokens.forEach((t,i)=>{
      const future=t.position>active, allowed=!causal||!future;
      addTile(i,-.87,trainingPass?1:future?.27:1);
      nodes.push({id:`qkv${i}`,p:pt(x(i),-.08,.16),label:i===selected?'Q':'K·V',kind:'capsule',color:i===selected?C.cyan:future&&causal?C.dim:C.teal,size:.25,opacity:future&&causal?.28:.85});
      edges.push({id:`projection${i}`,points:[pt(x(i),-.51),pt(x(i),-.24,.16)],color:C.teal,opacity:.4});
      const weight=trace.attention[active]?.[t.position]??0;
      if(allowed&&i!==selected) edges.push({id:`attention${i}`,points:arc(pt(sx,.04,.18),pt(x(i),.04,.18),Math.min(1.42,.25+Math.abs(sx-x(i))*.18)),color:weight>.12?C.cyan:C.teal,weight:.6+weight*7,opacity:.13+weight*.8});
      if(config.positionalEncoding&&!recorded) addLabel(`posencode${i}`,x(i),-1.67,`+ p${t.position}`,C.teal,.085);
    });
    if(distantClue) addLabel('omitted-context',x(0),.57,'EARLY CLUE ···',C.amber,.085);
    if(causal&&selected<tokens.length-1&&selected>=0) {
      const bx=(sx+x(selected+1))/2;
      edges.push({id:'mask-boundary',points:[pt(bx,-1.31,.05),pt(bx,1.53,.05)],color:C.amber,dashed:true,weight:1,opacity:.65});
      addLabel('mask-label',Math.min(3.6,bx+.82),1.6,recorded?'FUTURE WEIGHTS = 0':'FUTURE MASKED',C.amber,.105);
    }
    addLabel('head-label',-1.2,1.64,recorded?'RECORDED ATTENTION MATRIX':config.aggregateHeads?'COMBINED ATTENTION HEADS':`LAYER ${config.layer+1} · HEAD ${config.head+1}`,C.ivory,.11);
    caption=recorded?'Arcs show the imported attention matrix. Head, layer, and positional-encoding metadata were not supplied.':trainingPass?'Training computes representations at all token positions together. The causal mask still controls which earlier positions each token can read.':causal?'The selected query reads keys and values from its available past. The dashed boundary blocks every future token.':'With the causal mask removed, the query can read future tokens. This reveals why causal language models require masking.';
    state=recorded?'recorded attention weights':trainingPass?'parallel training positions':causal?'causal self-attention':'future positions accessible';
  }
  return {nodes,edges,labels,regions,caption,state};
}

function useLabelTexture(text: string, color: string, big = false) {
  const result = useMemo(() => {
    const canvas=document.createElement('canvas');
    const ctx=canvas.getContext('2d');
    const font=`${big?500:600} 96px ${big?'ui-monospace, monospace':'system-ui, sans-serif'}`;
    if(ctx)ctx.font=font;
    canvas.width=Math.max(128,Math.ceil((ctx?.measureText(text).width??text.length*58)+28));canvas.height=128;
    if(ctx) { ctx.clearRect(0,0,canvas.width,canvas.height); ctx.fillStyle=color; ctx.textAlign='center';ctx.textBaseline='middle';ctx.font=font;ctx.fillText(text,canvas.width/2,canvas.height/2); }
    const texture=new CanvasTexture(canvas);texture.colorSpace=SRGBColorSpace;return {texture,aspect:canvas.width/canvas.height};
  },[text,color,big]);
  useEffect(()=>()=>result.texture.dispose(),[result]);
  return result;
}
function LabelMesh({ label }: {label:Label}) {
  const {texture,aspect}=useLabelTexture(label.text,label.color??C.muted,true);
  const height=Math.max(.22,(label.size??.12)*2.35);
  return <mesh position={label.p}><planeGeometry args={[height*aspect,height]} /><meshBasicMaterial map={texture} transparent depthWrite={false} toneMapped={false} /></mesh>;
}
function NodeMesh({ node, onSelect, onSelectDecoder, reducedMotion }: {node:Node;onSelect:(n:number)=>void;onSelectDecoder?:(n:number)=>void;reducedMotion:boolean}) {
  const {texture,aspect}=useLabelTexture(node.label,node.labelColor??(node.kind==='tile'&&!node.selected&&node.color===C.teal?C.ivory:node.color));
  const group=useRef<Group>(null); const glow=useRef<Mesh>(null);
  useFrame(({clock})=>{if(glow.current&&!reducedMotion){const s=1+Math.sin(clock.elapsedTime*2)*.055;glow.current.scale.setScalar(s);} });
  const size=node.size??.47, width=node.kind==='capsule'?size*2:node.kind==='orb'?size*2:size;
  const height=node.kind==='capsule'?size*.73:node.kind==='orb'?size*2:node.kind==='gate'?size*2:.61;
  const opacity=node.opacity??1;
  const textHeight=Math.min(node.kind==='tile'?.44:height*.95,width*.92/aspect),textWidth=textHeight*aspect;
  return <group ref={group} position={node.p} onClick={node.token===undefined?undefined:(event)=>{event.stopPropagation();(node.decoder?(onSelectDecoder??onSelect):onSelect)(node.token!);}} onPointerOver={node.token===undefined?undefined:()=>{document.body.style.cursor='pointer';}} onPointerOut={()=>{document.body.style.cursor='auto';}}>
    {node.selected&&<mesh ref={glow} position={[0,0,-.035]}><planeGeometry args={[width+.2,height+.2]} /><meshBasicMaterial color={C.cyan} transparent opacity={.09} depthWrite={false} /></mesh>}
    {node.kind==='orb'?<mesh><sphereGeometry args={[size,16,12]}/><meshStandardMaterial color={node.color} roughness={.28} metalness={.55} emissive={node.color} emissiveIntensity={node.selected?.42:.12} transparent opacity={opacity}/></mesh>:<RoundedBox args={[width,height,.105]} radius={.045} smoothness={3}><meshStandardMaterial color={node.selected?'#164747':node.kind==='gate'?'#234743':'#143035'} roughness={.4} metalness={.3} emissive={node.color} emissiveIntensity={node.selected?.25:.045} transparent opacity={opacity}/></RoundedBox>}
    {node.kind!=='orb'&&<Line points={[pt(-width/2,-height/2,.061),pt(width/2,-height/2,.061),pt(width/2,height/2,.061),pt(-width/2,height/2,.061),pt(-width/2,-height/2,.061)]} color={node.color} lineWidth={node.selected?1.5:.65} transparent opacity={opacity*(node.selected?1:.5)} />}
    {node.label&&<mesh position={[0,0,node.kind==='orb'?size+.015:.065]}><planeGeometry args={[textWidth,textHeight]}/><meshBasicMaterial map={texture} transparent opacity={opacity} depthWrite={false} toneMapped={false}/></mesh>}
  </group>;
}
function CameraFit() {
  const {camera,size}=useThree();
  useEffect(()=>{const distance=Math.max(4.5,10.2/(size.width/size.height))/(2*Math.tan(Math.PI/10));camera.position.set(0,distance*.17,distance);camera.lookAt(0,0,0);camera.updateProjectionMatrix();},[camera,size]);
  return null;
}
function Scene({spec,onSelect,onSelectDecoder,reducedMotion}:{spec:SceneSpec;onSelect:(n:number)=>void;onSelectDecoder?:(n:number)=>void;reducedMotion:boolean}) {
  return <>
    <CameraFit/><ambientLight intensity={1.1}/><directionalLight position={[-4,5,6]} intensity={2} color="#c7f8f2"/><pointLight position={[4,-1,3]} intensity={8} color="#4fd1c5"/>
    <group position={[0,.08,0]}>
      {Array.from({length:19},(_,i)=><Line key={`gx${i}`} points={[pt((i-9)*.52,-1.93,-1.7),pt((i-9)*.52,-1.93,1.5)]} color="#315457" transparent opacity={.22} lineWidth={.5}/>)}
      {Array.from({length:8},(_,i)=><Line key={`gz${i}`} points={[pt(-4.8,-1.93,(i-4)*.46),pt(4.8,-1.93,(i-4)*.46)]} color="#315457" transparent opacity={.22} lineWidth={.5}/>)}
      {spec.regions.map(r=><group key={r.id}><Line points={[pt(r.x,r.y-r.h/2,.04),pt(r.x+r.w,r.y-r.h/2,.04),pt(r.x+r.w,r.y+r.h/2,.04),pt(r.x,r.y+r.h/2,.04),pt(r.x,r.y-r.h/2,.04)]} color={r.color} lineWidth={1} dashed={r.dashed} dashSize={.07} gapSize={.05} />{r.label&&<LabelMesh label={{id:r.id,p:pt(r.x+r.w/2,r.y-r.h/2-.52,.1),text:r.label.toUpperCase(),color:r.color,size:.1}}/>}</group>)}
      {spec.edges.map(edge=>{const arrow=arrowPoints(edge);return <group key={edge.id}><Line points={edge.points} color={edge.color} lineWidth={edge.weight??1} transparent opacity={edge.opacity??.8} dashed={edge.dashed} dashSize={.08} gapSize={.06}/>{arrow&&<Line points={arrow} color={edge.color} lineWidth={edge.weight??1} transparent opacity={edge.opacity??.8}/>}</group>;})}
      {spec.nodes.map(node=><NodeMesh key={node.id} node={node} onSelect={onSelect} onSelectDecoder={onSelectDecoder} reducedMotion={reducedMotion}/>)}
      {spec.labels.map(label=><LabelMesh key={label.id} label={label}/>)}
    </group>
  </>;
}

function FlatScene({spec,onSelect,onSelectDecoder}:{spec:SceneSpec;onSelect:(n:number)=>void;onSelectDecoder?:(n:number)=>void}) {
  const px=(v:number)=>500+v*99, py=(v:number)=>203-v*76;
  return <svg className="architecture-flat" viewBox="0 0 1000 400" role="img" aria-label={spec.caption}>
    <defs><pattern id="scene-grid" width="49.5" height="38" patternUnits="userSpaceOnUse"><path d="M 49.5 0 L 0 0 0 38" fill="none" stroke="#234246" strokeWidth=".65" opacity=".5"/></pattern><radialGradient id="scene-glow"><stop stopColor="#184342" stopOpacity=".42"/><stop offset="1" stopColor="#0a171a" stopOpacity="0"/></radialGradient></defs>
    <rect width="1000" height="400" fill="url(#scene-grid)"/><ellipse cx="500" cy="210" rx="450" ry="170" fill="url(#scene-glow)"/>
    {spec.regions.map(r=><g key={r.id}><rect x={px(r.x)} y={py(r.y+r.h/2)} width={r.w*99} height={r.h*76} rx="6" fill={r.color} fillOpacity=".03" stroke={r.color} strokeOpacity=".7" strokeDasharray={r.dashed?'5 5':undefined}/>{r.label&&<text x={px(r.x+r.w/2)} y={py(r.y-r.h/2-.5)} fill={r.color} className="flat-label">{r.label.toUpperCase()}</text>}</g>)}
    {spec.edges.map(e=>{const arrow=arrowPoints(e);return <g key={e.id} fill="none" stroke={e.color} strokeWidth={e.weight??1} opacity={e.opacity??.8}><polyline points={e.points.map(p=>`${px(p[0])},${py(p[1])}`).join(' ')} strokeDasharray={e.dashed?'5 5':undefined}/>{arrow&&<polyline points={arrow.map(p=>`${px(p[0])},${py(p[1])}`).join(' ')}/>}</g>;})}
    {spec.nodes.map(n=>{
      const size=n.size??.47,w=(n.kind==='capsule'?size*2:n.kind==='orb'?size*2:size)*99,h=(n.kind==='capsule'?size*.73:n.kind==='orb'?size*2:n.kind==='gate'?size*2:.61)*76;
      return <g key={n.id} opacity={n.opacity??1} className={n.token===undefined?'':'flat-token'} onClick={n.token===undefined?undefined:()=>(n.decoder?(onSelectDecoder??onSelect):onSelect)(n.token!)}>
        {n.selected&&<rect x={px(n.p[0])-w/2-7} y={py(n.p[1])-h/2-7} width={w+14} height={h+14} rx="9" fill={C.cyan} fillOpacity=".07" stroke={C.cyan} strokeOpacity=".18"/>}
        {n.kind==='orb'?<ellipse cx={px(n.p[0])} cy={py(n.p[1])} rx={w/2} ry={h/2} fill="#173b3c" stroke={n.color} strokeWidth={n.selected?2:1}/>:<><rect x={px(n.p[0])-w/2+2} y={py(n.p[1])-h/2+4} width={w} height={h} rx="5" fill="#061316" stroke="#1e363b"/><rect x={px(n.p[0])-w/2} y={py(n.p[1])-h/2} width={w} height={h} rx="5" fill={n.selected?'#184744':'#10292d'} stroke={n.color} strokeOpacity={n.selected?1:.5}/></>}
        <text x={px(n.p[0])} y={py(n.p[1])+1} fill={n.labelColor??(n.kind==='tile'&&!n.selected&&n.color===C.teal?C.ivory:n.color)} className={n.kind==='tile'?'flat-token-label':'flat-node-label'} style={n.kind==='tile'&&n.label.length>1?{fontSize:Math.min(19,w*.88/(n.label.length*.6))}:undefined}>{n.label}</text>
      </g>;
    })}
    {spec.labels.map(l=><text key={l.id} x={px(l.p[0])} y={py(l.p[1])} fill={l.color??C.muted} className="flat-label" style={{fontSize:Math.max(11,(l.size??.12)*108)}}>{l.text}</text>)}
  </svg>;
}
class SceneBoundary extends Component<{children:ReactNode;fallback:ReactNode},{failed:boolean}> {
  state={failed:false};
  static getDerivedStateFromError(){return {failed:true};}
  render(){return this.state.failed?this.props.fallback:this.props.children;}
}
function hasWebGL(){try{const canvas=document.createElement('canvas');const gl=canvas.getContext('webgl2')||canvas.getContext('webgl');if(!gl)return false;gl.getExtension('WEBGL_lose_context')?.loseContext();return true;}catch{return false;}}

export function ArchitectureScene(props:ArchitectureSceneProps) {
  const {trace,step,reducedMotion,focusMode,onSelectStep,onSelectDecoderStep,decoderStep,trainingPass,config,bptt}=props;
  const [webgl]=useState(hasWebGL),[contextLost,setContextLost]=useState(false);
  const spec=useMemo(()=>makeScene({trace,step,config,bptt,trainingPass,decoderStep}),[trace,step,config,bptt,trainingPass,decoderStep]);
  const flat=focusMode||reducedMotion||!webgl||contextLost;
  const fallback=<FlatScene spec={spec} onSelect={onSelectStep} onSelectDecoder={onSelectDecoderStep}/>;
  const current=trace.tokens[Math.max(0,Math.min(step,trace.tokens.length-1))];
  const isAttention=trace.architecture==='attention';
  const selectedDecoder=Math.min(trace.decoderTokens.length-1,Math.max(0,decoderStep??step%Math.max(1,trace.decoderTokens.length)));
  return <div className={`architecture-scene${flat?' is-flat':''}`}>
    <div className="scene-frame-topline"><span className="scene-live-dot"/><span>{spec.state}</span><span className="scene-dimension">{flat?'2D DIAGRAM':'SPATIAL VIEW'}</span></div>
    <div className="scene-viewport" aria-label={`${trace.metadata.label} interactive architecture diagram`}>
      {flat?fallback:<SceneBoundary fallback={fallback}><Suspense fallback={fallback}><Canvas dpr={[1,1.5]} camera={{position:[0,1.2,7],fov:36,near:.1,far:50}} gl={{alpha:true,antialias:true,powerPreference:'low-power'}} onCreated={({gl})=>{gl.domElement.addEventListener('webglcontextlost',()=>setContextLost(true),{once:true});}}><Scene spec={spec} onSelect={onSelectStep} onSelectDecoder={onSelectDecoderStep} reducedMotion={reducedMotion}/></Canvas></Suspense></SceneBoundary>}
    </div>
    <div className="scene-bottomline"><span className="scene-key"><i/> active state</span><span className="scene-key history"><i/> retained history</span><span className="scene-position">{isAttention?'target':'t'} = {String((isAttention?selectedDecoder:step)+1).padStart(2,'0')} <span>/ {isAttention?trace.decoderTokens.length:trace.tokens.length}</span></span></div>
    <p className="scene-accessible-description" aria-live="polite">{spec.caption} {isAttention?`Selected decoder target ${selectedDecoder+1}: ${trace.decoderTokens[selectedDecoder]??'none'}.`:`Selected position ${step+1}: ${current?readable(current.text):'none'}.`}</p>
    <div className={`scene-token-selectors${isAttention?' is-word':''}`} aria-label={isAttention?'Select decoder target':'Select token position'}>
      {isAttention?trace.decoderTokens.map((token,i)=><button key={i} type="button" className={i===selectedDecoder?'selected':''} aria-pressed={i===selectedDecoder} aria-label={`Select decoder target ${i+1}: ${token}`} onClick={()=>(onSelectDecoderStep??onSelectStep)(i)}>{token}</button>):trace.tokens.map((token,i)=><button key={token.id} type="button" className={i===step?'selected':''} aria-pressed={i===step} aria-label={`Select token ${i+1}: ${token.text===' '?'space':token.text}`} onClick={()=>onSelectStep(i)}>{readable(token.text)}</button>)}
    </div>
  </div>;
}
export default ArchitectureScene;
