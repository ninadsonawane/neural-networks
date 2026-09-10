import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { ArrowRight, ArrowUpRight, Atom, Box, Check, ChevronDown, ChevronLeft, ChevronRight, CircleHelp, Eye, Filter, Focus, GitCompareArrows, Layers3, ListTree, Move3D, Pause, Play, RotateCcw, Settings2, Sparkles, X } from 'lucide-react';
import { useLab, PROMPTS } from './store';
import { buildTrace } from './simulation/engine';
import { buildExplorerModel } from './explorer/models';
import { PART_CATEGORIES } from './explorer/types';
import type { Architecture, Trace } from './simulation/types';
import { STAGES } from './data/stages';

const MuseumScene = lazy(()=>import('./components/MuseumScene'));
const ExplorerDetails = lazy(()=>import('./components/ExplorerDetails'));
type DetailTab = 'mechanism'|'probabilities'|'compare'|'about';
const INTUITIONS:Record<Architecture,string>={mlp:'It only sees the last few characters. Everything before the window is out of reach.',rnn:'It can carry the past, but must squeeze it through one state at every step.',lstm:'Gates decide what survives the journey.',attention:'It can look back at a useful source instead of compressing everything into one vector.',transformer:'Every character can directly look at relevant earlier characters.'};
const TITLES:Record<Architecture,[string,string]>={mlp:['A small window.','A hard limit.'],rnn:['Carry a little','of the past.'],lstm:['Remember what','matters.'],attention:['The art of','looking back.'],transformer:['Connection is','a kind of memory.']};

export default function Explorer() {
  const lab=useLab(); const reducedMotion=!!useReducedMotion();
  const [imported,setImported]=useState<Trace|null>(null),[decoderStep,setDecoderStep]=useState(0);
  const [popover,setPopover]=useState<'filters'|'outline'|null>(null),[detail,setDetail]=useState<DetailTab|null>(null);
  const [announcement,setAnnouncement]=useState(''),[hoveredId,setHoveredId]=useState<string|null>(null),[showExamples,setShowExamples]=useState(false);
  const selectionHeading=useRef<HTMLHeadingElement>(null);
  const effectiveInput=useMemo(()=>Array.from(lab.challenge?`zara kept the key. ${lab.input}`:lab.input).slice(0,80).join(''),[lab.input,lab.challenge]);
  const simulated=useMemo(()=>buildTrace(effectiveInput,lab.stage,lab.config),[effectiveInput,lab.stage,lab.config]);
  const trace=imported??simulated,step=Math.min(lab.step,trace.steps.length-1);
  const stage=STAGES.find(s=>s.id===trace.architecture)!;
  const model=useMemo(()=>buildExplorerModel({trace,step,config:lab.config,decoderStep}),[trace,step,lab.config,decoderStep]);
  const selected=model.parts.find(p=>p.id===lab.selectedId)??null,hovered=model.parts.find(p=>p.id===hoveredId);
  const presentCategories=PART_CATEGORIES.filter(c=>model.parts.some(p=>p.category===c.id));
  const filteredParts=model.parts.filter(p=>lab.visibleCategories.includes(p.category));
  const complete=lab.hasRun&&!lab.playing&&step===trace.tokens.length-1,dirty=lab.draft!==lab.input;
  const nextStage=STAGES[(STAGES.findIndex(s=>s.id===stage.id)+1)%STAGES.length];
  const transition={duration:reducedMotion?0:.26};

  useEffect(()=>{if(!lab.playing)return;const timer=window.setInterval(()=>{const current=useLab.getState();if(current.step>=trace.steps.length-1)current.setPlaying(false);else current.setStep(current.step+1);},reducedMotion?420:300);return()=>clearInterval(timer);},[lab.playing,trace.steps.length,reducedMotion]);
  useEffect(()=>setDecoderStep(step%Math.max(1,trace.decoderTokens.length)),[step,trace.decoderTokens.length]);
  useEffect(()=>{const handler=(event:KeyboardEvent)=>{
    if(event.key==='Escape'&&!detail){lab.selectPart(null);setPopover(null);setShowExamples(false);}
    if(event.defaultPrevented||detail||(event.target as HTMLElement).closest('input,textarea,select,button,a,summary,[role="button"],[role="dialog"]'))return;
    if(lab.mode==='experiment'&&event.code==='Space'){event.preventDefault();if(!lab.hasRun||step===trace.tokens.length-1){if(!imported&&dirty)lab.tokenize();lab.startRun();}else lab.setPlaying(!lab.playing);}
    if(lab.mode==='experiment'&&(event.key==='ArrowLeft'||event.key==='ArrowRight')){event.preventDefault();lab.setPlaying(false);lab.setStep(Math.max(0,Math.min(trace.tokens.length-1,step+(event.key==='ArrowRight'?1:-1))));}
  };window.addEventListener('keydown',handler);return()=>window.removeEventListener('keydown',handler);},[detail,lab,step,trace.tokens.length,imported,dirty]);

  function selectPart(id:string|null,fromOutline=false){
    lab.selectPart(id);setPopover(null);
    if(id){
      const part=model.parts.find(p=>p.id===id);
      if(part?.decoderIndex!==undefined)setDecoderStep(part.decoderIndex);
      const headId=part?.parentId??part?.id;
      const headMatch=headId?.match(/^head-(\d+)$/);
      if(!imported&&headMatch)lab.setConfig({head:Number(headMatch[1]),aggregateHeads:false});
      if(part?.detailOnly&&lab.explosion<.15)lab.setExplosion(1);
      setAnnouncement(`${part?.label??'Component'} selected. Details are open.`);
      if(fromOutline)requestAnimationFrame(()=>selectionHeading.current?.focus());
    }
  }
  function switchStage(id:Architecture){setImported(null);lab.setStage(id);setPopover(null);setHoveredId(null);setDetail(null);setAnnouncement(`${STAGES.find(s=>s.id===id)?.name} selected.`);}
  function changeMode(mode:'learn'|'experiment'){lab.setMode(mode);setPopover(null);setShowExamples(false);setHoveredId(null);}
  function tokenize(){lab.tokenize();setAnnouncement(`${Array.from(lab.draft||' ').length} characters ready. Press Run to follow the signal.`);}
  function run(){if(!imported&&dirty)lab.tokenize();setPopover(null);setShowExamples(false);lab.startRun();setAnnouncement(imported?'Replaying the recorded trace.':'Running an educational simulation of your character sequence.');}
  function tryNext(){switchStage(nextStage.id);lab.setMode('experiment');lab.startRun();setAnnouncement(`${nextStage.name}. Replaying the same input with a new way to remember.`);}
  function reset(){lab.resetExplorer();setHoveredId(null);setPopover(null);setShowExamples(false);setAnnouncement('Architecture assembled. View and filters reset.');}
  function onImport(next:Trace|null){setImported(next);lab.resetExplorer();lab.setMode('experiment');setDetail(null);setAnnouncement(next?'Imported recording ready to replay. Source authenticity is not verified.':'Returned to educational simulation.');}
  function outputCaption(){
    if(imported)return 'These are the recorded values from your imported trace.';
    if(stage.id==='mlp')return `Only the final ${Math.min(lab.config.contextLength,trace.tokens.length)} characters reached the output. Earlier context was outside the window.`;
    if(stage.id==='rnn')return 'The state crossed the whole sequence. Each update compressed the past again.';
    if(stage.id==='lstm')return lab.config.pinClue?'The gates protected the early clue along the memory lane.':'The gates chose what to carry forward. Pin a clue to explore preservation.';
    if(stage.id==='attention')return lab.config.attentionEnabled?'Each target could reach back to its weighted source context.':'The source had to fit into a single capsule. Open the bridge to give each target a direct route back.';
    return 'Attention gave distant characters a direct connection. This replays your sequence; it does not generate new text.';
  }

  return <div className={`museum-app ${lab.mode==='experiment'?'is-experiment':''} ${selected?'has-selection':''} ${reducedMotion?'reduce-motion':''}`}>
    <a className="skip-link" href="#component-outline" onClick={()=>setPopover('outline')}>Explore the accessible component outline</a>
    <header className="museum-header">
      <a className="museum-brand" href="#" onClick={event=>{event.preventDefault();reset();changeMode('learn');}} aria-label="Neural Atlas home"><Atom size={28} strokeWidth={1.15}/><span>neural<span>atlas</span><b>.</b></span></a>
      <span className="museum-header-note">AN EXPLORABLE HISTORY OF MEMORY</span>
      <div className="museum-header-actions"><span className={`museum-provenance ${imported?'is-recorded':''}`}><i/>{imported?'Imported recording':'Educational simulation'}</span><button className="museum-icon" aria-label="About Neural Atlas and data sources" title="About this atlas" onClick={()=>setDetail('about')}><CircleHelp size={18}/></button></div>
    </header>
    <nav className={`museum-rail ${lab.railExpanded?'expanded':''}`} aria-label="Evolution of model memory">
      <button className="rail-expander museum-icon" aria-label={lab.railExpanded?'Collapse evolution rail':'Expand evolution rail'} aria-expanded={lab.railExpanded} onClick={()=>lab.setRailExpanded(!lab.railExpanded)}>{lab.railExpanded?<ChevronLeft size={15}/>:<ChevronRight size={15}/>}</button>
      <span className="museum-rail-caption">FIVE WAYS TO REMEMBER</span>
      {STAGES.map(s=><button key={s.id} className={`museum-stage ${s.id===stage.id?'current':''}`} aria-label={`${s.number} ${s.name}, ${s.year}`} aria-current={s.id===stage.id?'step':undefined} title={`${s.name} · ${s.year}`} onClick={()=>switchStage(s.id)}><span>{s.number}</span><div>{s.name}<small>{s.year}</small></div></button>)}
      <span className="rail-end">2003 — 2017</span>
    </nav>
    <main className="museum-main" aria-label={`${stage.name} architecture explorer`}>
      <div className="museum-canvas-area"><Suspense fallback={<div className="museum-loading"><Atom size={32} strokeWidth={1}/><span>Assembling the architecture</span></div>}><MuseumScene model={model} explosion={lab.explosion} selectedId={selected?.id??null} isolated={lab.isolated&&!!selected} visibleCategories={lab.visibleCategories} mode={lab.mode} playing={lab.playing} hasRun={lab.hasRun} step={step} resetKey={lab.cameraReset} reducedMotion={reducedMotion} flat={lab.focusMode} onSelect={selectPart} onHover={setHoveredId}/></Suspense></div>
      <div className="museum-intro">
        <div className="museum-eyebrow"><span>{stage.number} / 05</span><span>{stage.model} <i/> {stage.year}</span></div>
        <AnimatePresence mode="wait"><motion.div key={stage.id} initial={{opacity:0,y:reducedMotion?0:7}} animate={{opacity:1,y:0}} exit={{opacity:0,y:reducedMotion?0:-5}} transition={transition}><h1>{TITLES[stage.id][0]}<br/><em>{TITLES[stage.id][1]}</em></h1><p>{stage.id==='mlp'&&!imported?`It only sees the last ${lab.config.contextLength} characters. Everything before the window is out of reach.`:imported?INTUITIONS[stage.id].replaceAll('characters','tokens').replaceAll('character','token'):INTUITIONS[stage.id]}</p></motion.div></AnimatePresence>
        <label className="museum-mobile-stage"><span className="sr-only">Choose architecture</span><select aria-label="Choose architecture" value={stage.id} onChange={e=>switchStage(e.target.value as Architecture)}>{STAGES.map(s=><option value={s.id} key={s.id}>{s.number} · {s.name}</option>)}</select><ChevronDown size={13}/></label>
      </div>
      <div className="museum-tools" aria-label="Explorer utilities"><button className={`museum-icon ${popover==='outline'?'on':''}`} title="Component outline" aria-label="Open accessible component outline" aria-expanded={popover==='outline'} onClick={()=>setPopover(popover==='outline'?null:'outline')}><ListTree size={17}/></button><button className={`museum-icon ${lab.focusMode?'on':''}`} aria-label="Use accessible 2D view" aria-pressed={lab.focusMode} title="2D view" onClick={()=>lab.setFocusMode(!lab.focusMode)}><Eye size={17}/></button><button className="museum-icon" title="Compare the five ideas" aria-label="Compare architectures" onClick={()=>setDetail('compare')}><GitCompareArrows size={17}/></button>{lab.mode==='experiment'&&<button className="museum-icon" title="Experiment settings" aria-label="Open experiment settings" onClick={()=>setDetail('mechanism')}><Settings2 size={17}/></button>}</div>
      {!filteredParts.length&&<div className="museum-empty"><Layers3 size={26}/><p>Every layer is hidden.</p><button onClick={lab.showAllCategories}>Show all components</button></div>}
      {hovered&&!selected&&<div className="museum-hover" aria-hidden="true"><span className="hover-dot"/>{hovered.label}<small>Click to discover</small></div>}
      <AnimatePresence>{selected&&<motion.aside key="selection" className="museum-selection glass-surface" aria-label="Selected component" initial={{opacity:0,y:reducedMotion?0:12}} animate={{opacity:1,y:0}} exit={{opacity:0,y:reducedMotion?0:8}} transition={transition}>
        <div className="selection-kicker"><span>{PART_CATEGORIES.find(c=>c.id===selected.category)?.label}</span><button className="museum-icon" aria-label="Close component details" onClick={()=>selectPart(null)}><X size={15}/></button></div><h2 ref={selectionHeading} tabIndex={-1}>{selected.label}</h2><p>{selected.purpose}</p><div className="selection-why"><span>WHY IT EXISTS</span><p>{selected.why}</p></div><div className="selection-actions"><button className="museum-primary" aria-pressed={lab.isolated} onClick={()=>lab.setIsolated(!lab.isolated)}><Focus size={14}/>{lab.isolated?'Show in context':'Isolate component'}</button><button className="museum-text" onClick={()=>setDetail(selected.category==='output'?'probabilities':'mechanism')}>Go deeper <ArrowUpRight size={13}/></button></div>{selected.resource&&<a className="selection-resource" href={selected.resource} target="_blank" rel="noreferrer">From the notebook <ArrowUpRight size={11}/></a>}
      </motion.aside>}</AnimatePresence>
      <AnimatePresence>{popover&&<motion.aside className={`museum-popover glass-surface ${popover==='outline'?'component-outline':'filter-popover'}`} id={popover==='outline'?'component-outline':undefined} aria-label={popover==='outline'?'Architecture component outline':'Semantic component filters'} initial={{opacity:0,y:reducedMotion?0:8}} animate={{opacity:1,y:0}} exit={{opacity:0,y:reducedMotion?0:8}} transition={transition}>
        <div className="popover-title"><span>{popover==='outline'?'Inside this architecture':'Layers of the idea'}</span><button className="museum-icon" aria-label="Close explorer panel" onClick={()=>setPopover(null)}><X size={15}/></button></div>
        {popover==='filters'?<><p>Keep the parts you want to explore.</p>{presentCategories.map(category=><label className="filter-option" key={category.id}><i style={{background:category.color}}/><span>{category.label}</span><input type="checkbox" checked={lab.visibleCategories.includes(category.id)} onChange={()=>lab.toggleCategory(category.id)}/></label>)}<button className="museum-text" onClick={lab.showAllCategories}>Show all layers <RotateCcw size={12}/></button></>:<><p>The same selectable pieces, without the canvas.</p><div className="outline-content">{presentCategories.map(category=><details key={category.id} open><summary><span style={{background:category.color}}/>{category.label}<small>{model.parts.filter(p=>p.category===category.id).length}</small></summary><ul>{model.parts.filter(p=>p.category===category.id).map(part=><li key={part.id}><button aria-pressed={selected?.id===part.id} onClick={()=>{if(!lab.visibleCategories.includes(part.category))lab.toggleCategory(part.category);selectPart(part.id,true);}}><span>{part.label}</span><ArrowUpRight size={12}/></button></li>)}</ul></details>)}</div></>}
      </motion.aside>}</AnimatePresence>
      {lab.mode==='experiment'&&<div className={`museum-experiment ${selected?'with-selection':''}`}>
        <AnimatePresence>{complete&&!selected&&<motion.div className="experiment-result glass-surface" initial={{opacity:0,y:reducedMotion?0:10}} animate={{opacity:1,y:0}} exit={{opacity:0}} transition={transition}><div><span className="result-glyph"><Check size={17}/></span><span className="result-eyebrow">{imported?'RECORDING REPLAYED':'THE SEQUENCE HAS ARRIVED'}</span><button className="museum-text" onClick={()=>setDetail('probabilities')}>See probabilities <ArrowUpRight size={12}/></button></div><p>{outputCaption()}</p>{!imported&&<button className="museum-primary" onClick={tryNext}>{stage.id==='transformer'?'Revisit the first idea':'Try the next idea'}<ArrowRight size={15}/></button>}</motion.div>}</AnimatePresence>
        <div className="experiment-command glass-surface"><form className="experiment-input-row" onSubmit={event=>{event.preventDefault();tokenize();}}><span className="input-chevron">›</span><input aria-label="Your experiment text" maxLength={160} spellCheck={false} value={imported?.input??lab.draft} disabled={!!imported} onChange={event=>{lab.setDraft(Array.from(event.target.value).slice(0,80).join(''));lab.setPlaying(false);}} placeholder="Give the model something to remember…"/><button type="submit" disabled={!!imported} className="tokenize-quiet">Tokenize</button><span className="command-divider"/><button type="button" className={`${!complete&&!selected?'museum-primary':'museum-text'} run-control`} onClick={()=>lab.playing?lab.setPlaying(false):lab.hasRun&&!complete?lab.setPlaying(true):run()}>{lab.playing?<Pause size={14}/>:complete?<RotateCcw size={14}/>:<Play size={14}/>}<span>{lab.playing?'Pause':complete?'Replay':lab.hasRun?'Resume':'Run'}</span></button></form>
          <div className="experiment-caption"><button className="museum-text" onClick={()=>setShowExamples(!showExamples)} aria-expanded={showExamples}>{trace.tokens.length} {imported?'tokens':'characters'}<ChevronDown size={11}/></button>{lab.hasRun?<div className="experiment-scrub"><button className="museum-icon" aria-label={imported?"Previous token":"Previous character"} disabled={step===0} onClick={()=>{lab.setPlaying(false);lab.setStep(step-1);}}><ChevronLeft size={12}/></button><input aria-label="Simulation timeline" type="range" min={0} max={Math.max(1,trace.tokens.length-1)} value={step} disabled={trace.tokens.length===1} onChange={e=>{lab.setPlaying(false);lab.setStep(Number(e.target.value));}}/><button className="museum-icon" aria-label={imported?"Next token":"Next character"} disabled={step===trace.tokens.length-1} onClick={()=>{lab.setPlaying(false);lab.setStep(step+1);}}><ChevronRight size={12}/></button><span>{String(step+1).padStart(2,'0')} / {trace.tokens.length}</span></div>:<span>{dirty?'Tokenize your changes, or press Run.':imported?'Recorded values · source unverified':'One character per tile. Follow the signal.'}</span>}</div>
          {showExamples&&<div className="experiment-examples"><span>TRY A SENTENCE</span>{PROMPTS.slice(0,2).map((prompt,i)=><button key={prompt} disabled={!!imported} onClick={()=>{lab.setInput(prompt);setShowExamples(false);setAnnouncement('Example tokenized. Press Run.');}}>{i===0?'The cat & the mouse':'A little Hinglish'}<ArrowUpRight size={11}/></button>)}<label><input type="checkbox" disabled={!!imported} checked={lab.challenge} onChange={e=>lab.setChallenge(e.target.checked)}/> Add a distant clue</label></div>}
        </div>
      </div>}
      <div className="museum-bottom"><div className="museum-dock glass-surface" aria-label="Main explorer controls">
        <div className="museum-modes" aria-label="Experience mode"><button aria-pressed={lab.mode==='learn'} onClick={()=>changeMode('learn')}><Box size={15}/><span>Learn</span></button><button aria-pressed={lab.mode==='experiment'} onClick={()=>changeMode('experiment')}><Sparkles size={15}/><span>Experiment</span></button></div><span className="dock-divider"/>
        <div className="assembly-control"><div><button onClick={()=>lab.setExplosion(0)} aria-label="Assemble architecture">Assembly</button><span>{Math.round(lab.explosion*100)}%</span><button onClick={()=>lab.setExplosion(1)} aria-label="Explode architecture">Exploded</button></div><input aria-label="Assembly to exploded view" aria-valuetext={`${Math.round(lab.explosion*100)} percent exploded`} type="range" min={0} max={100} value={Math.round(lab.explosion*100)} onChange={e=>lab.setExplosion(Number(e.target.value)/100)}/></div><span className="dock-divider"/>
        <button className={`dock-action ${popover==='filters'?'on':''}`} aria-label="Filter components" aria-expanded={popover==='filters'} onClick={()=>setPopover(popover==='filters'?null:'filters')}><Filter size={16}/><span>Filter</span>{presentCategories.some(c=>!lab.visibleCategories.includes(c.id))&&<i/>}</button><button className="dock-action" aria-label="Reset architecture and camera" onClick={reset}><RotateCcw size={16}/><span>Reset</span></button>
      </div><div className="museum-hints"><span><Move3D size={12}/>{lab.focusMode||reducedMotion?<>Select a component <i/> Slide to separate layers</>:<>Drag to rotate <i/> Scroll to zoom <i/> Click to discover</>}</span><button onClick={()=>setDetail('about')}>A field guide to neural memory <ArrowUpRight size={11}/></button></div></div>
      <span className="sr-only" role="status" aria-live="polite">{announcement}</span>
    </main>
    <Suspense fallback={null}>{detail&&<ExplorerDetails open={true} onClose={()=>setDetail(null)} tab={detail} trace={trace} step={step} config={lab.config} onConfig={lab.setConfig} onStep={lab.setStep} decoderStep={decoderStep} onDecoderStep={setDecoderStep} onImport={onImport} onExplore={switchStage}/>}</Suspense>
  </div>;
}
