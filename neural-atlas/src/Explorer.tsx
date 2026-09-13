import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'framer-motion';
import { ArrowRight, ArrowUpRight, Atom, Box, Check, ChevronDown, ChevronLeft, ChevronRight, CircleHelp, Eye, Filter, Focus, GitCompareArrows, Layers3, ListTree, Move3D, Pause, Play, RotateCcw, Settings2, Sparkles, X } from 'lucide-react';
import { useLab, PROMPTS } from './store';
import { buildTrace } from './simulation/engine';
import { buildExplorerModel } from './explorer/models';
import { PART_CATEGORIES } from './explorer/types';
import type { Architecture, Trace } from './simulation/types';
import { STAGES } from './data/stages';
import { ADDED, DEFAULT_SANDBOX, INTUITION, MOMENT, PRESETS, cinematicModel, momentAt, responsiblePart, sandboxConfig, type Sandbox } from './explorer/cinema';
import { paperFor } from './data/papers';
import ArchitecturePlayground from './components/ArchitecturePlayground';
import PaperOrigin from './components/PaperOrigin';

const MuseumScene = lazy(()=>import('./components/MuseumScene'));
const ExplorerDetails = lazy(()=>import('./components/ExplorerDetails'));
type DetailTab = 'mechanism'|'probabilities'|'compare'|'about';

export default function Explorer() {
  const lab=useLab(); const reducedMotion=!!useReducedMotion();
  const [imported,setImported]=useState<Trace|null>(null),[decoderStep,setDecoderStep]=useState(0);
  const [popover,setPopover]=useState<'filters'|'outline'|null>(null),[detail,setDetail]=useState<DetailTab|null>(null);
  const [announcement,setAnnouncement]=useState(''),[hoveredId,setHoveredId]=useState<string|null>(null),[showExamples,setShowExamples]=useState(false);
  const selectionHeading=useRef<HTMLHeadingElement>(null);
  const [sandbox,setSandbox]=useState<Sandbox>({...DEFAULT_SANDBOX,context:3});
  const stageSettings=useRef<Partial<Record<Architecture,Sandbox>>>({});
  const [changed,setChanged]=useState(''),[scan,setScan]=useState(false),[paperFront,setPaperFront]=useState(false),[paperOpen,setPaperOpen]=useState(false);
  const [launching,setLaunching]=useState(false),[moment,setMoment]=useState(false),[runId,setRunId]=useState(0),[speed,setSpeed]=useState(1800),[added,setAdded]=useState('');
  const [clueChoice,setClueChoice]=useState<{input:string;index:number}|null>(null);
  const [showCluePicker,setShowCluePicker]=useState(false);
  const momentSeen=useRef(false),sceneArea=useRef<HTMLDivElement>(null),paperThread=useRef<SVGLineElement>(null);
  const effectiveInput=useMemo(()=>Array.from(lab.challenge?`zara kept the key. ${lab.input}`:lab.input).slice(0,sandbox.sequence).join(''),[lab.input,lab.challenge,sandbox.sequence]);
  const clueIndex=clueChoice?.input===effectiveInput?clueChoice.index:Math.max(0,Array.from(effectiveInput).findIndex(char=>/[\p{L}\p{N}]/u.test(char)));
  const activeConfig=useMemo(()=>({...sandboxConfig(lab.config,sandbox,lab.stage),clueIndex}),[lab.config,sandbox,lab.stage,clueIndex]);
  const simulated=useMemo(()=>buildTrace(effectiveInput,lab.stage,activeConfig),[effectiveInput,lab.stage,activeConfig]);
  const trace=imported??simulated,step=Math.min(lab.step,trace.steps.length-1);
  const stage=STAGES.find(s=>s.id===trace.architecture)!;
  const model=useMemo(()=>cinematicModel(buildExplorerModel({trace,step,config:activeConfig,decoderStep}),sandbox,trace,step,lab.hasRun),[trace,step,activeConfig,decoderStep,sandbox,lab.hasRun]);
  const gru=!imported&&lab.config.gatedVariant==='gru';
  const architectureTitle=stage.id==='lstm'?(imported?'GATED MEMORY':gru?'GRU':'LSTM'):stage.id.toUpperCase();
  const paper=paperFor(stage.id,gru),targetId=responsiblePart(stage.id,gru);
  const selected=model.parts.find(p=>p.id===lab.selectedId)??null,hovered=model.parts.find(p=>p.id===hoveredId);
  const presentCategories=PART_CATEGORIES.filter(c=>model.parts.some(p=>p.category===c.id));
  const filteredParts=model.parts.filter(p=>lab.visibleCategories.includes(p.category));
  const complete=lab.hasRun&&!lab.playing&&!launching&&!moment&&step===trace.tokens.length-1,dirty=lab.draft!==lab.input;
  const nextStage=STAGES[(STAGES.findIndex(s=>s.id===stage.id)+1)%STAGES.length];
  const transition={duration:reducedMotion?0:.26};

  useEffect(()=>{
    if(!lab.playing||paperOpen)return;
    const timer=window.setTimeout(()=>{
      if(launching){setLaunching(false);return;}
      if(moment){setMoment(false);setPaperFront(false);return;}
      if(!imported&&!momentSeen.current&&step>=momentAt(trace,sandbox.context)){momentSeen.current=true;setMoment(true);setPaperFront(true);return;}
      if(step>=trace.steps.length-1)useLab.getState().setPlaying(false);else useLab.getState().setStep(step+1);
    },launching?(reducedMotion?100:1900):moment?2600:speed);
    return()=>clearTimeout(timer);
  },[lab.playing,step,launching,moment,runId,paperOpen,imported,trace,sandbox.context,speed,reducedMotion,lab.setPlaying,lab.setStep]);
  useEffect(()=>{if(!added)return;const timer=window.setTimeout(()=>setAdded(''),4200);return()=>clearTimeout(timer);},[added]);
  useEffect(()=>setDecoderStep(step%Math.max(1,trace.decoderTokens.length)),[step,trace.decoderTokens.length]);
  useEffect(()=>{const handler=(event:KeyboardEvent)=>{
    if(event.key==='Escape'&&!detail&&!paperOpen){setShowCluePicker(false);lab.selectPart(null);lab.setExplosion(0);setPopover(null);setShowExamples(false);setPaperFront(false);}
    if(event.defaultPrevented||detail||paperOpen||(event.target as HTMLElement).closest('input,textarea,select,button,a,summary,[role="button"],[role="dialog"]'))return;
    if(lab.mode==='experiment'&&event.code==='Space'){event.preventDefault();if(!lab.hasRun||step===trace.tokens.length-1){if(!imported&&dirty)lab.tokenize();momentSeen.current=false;setMoment(false);setLaunching(!imported);setRunId(id=>id+1);lab.startRun();}else lab.setPlaying(!lab.playing);}
    if(lab.mode==='experiment'&&(event.key==='ArrowLeft'||event.key==='ArrowRight')){event.preventDefault();lab.setPlaying(false);lab.setStep(Math.max(0,Math.min(trace.tokens.length-1,step+(event.key==='ArrowRight'?1:-1))));}
  };window.addEventListener('keydown',handler);return()=>window.removeEventListener('keydown',handler);},[detail,lab,step,trace.tokens.length,imported,dirty,paperOpen]);

  function selectPart(id:string|null,fromOutline=false){
    lab.selectPart(id);lab.setIsolated(!!id);lab.setPlaying(false);setPopover(null);setMoment(false);
    if(id){
      const part=model.parts.find(p=>p.id===id);
      if(part?.decoderIndex!==undefined)setDecoderStep(part.decoderIndex);
      const headId=part?.parentId??part?.id;
      const headMatch=headId?.match(/^head-(\d+)$/);
      if(!imported&&headMatch){setSandbox(s=>({...s,head:Number(headMatch[1])}));lab.setConfig({aggregateHeads:false});}
      if(part?.detailOnly&&lab.explosion<.15)lab.setExplosion(1);
      setAnnouncement(`${part?.label??'Component'} selected. Details are open.`);
      if(fromOutline)requestAnimationFrame(()=>selectionHeading.current?.focus());
    }
  }
  function switchStage(id:Architecture){if(!imported)stageSettings.current[lab.stage]=sandbox;setSandbox({...stageSettings.current[id]??{...DEFAULT_SANDBOX,context:id==='mlp'?3:32},sequence:sandbox.sequence});setImported(null);lab.setStage(id);setMoment(false);setLaunching(false);setPaperFront(false);setPopover(null);setHoveredId(null);setDetail(null);setAdded(ADDED[id]);setAnnouncement(`${STAGES.find(s=>s.id===id)?.name} selected.`);}
  function changeMode(mode:'learn'|'experiment'){setMoment(false);setLaunching(false);lab.setMode(mode);setPopover(null);setShowExamples(false);setHoveredId(null);}
  function tokenize(){lab.tokenize();setAnnouncement(`${Array.from(lab.draft||' ').length} characters ready. Press Throw it in to follow the signal.`);}
  function run(){setShowCluePicker(false);if(!imported&&dirty)lab.tokenize();setPopover(null);setShowExamples(false);momentSeen.current=false;setMoment(false);setLaunching(!imported);setPaperFront(false);setRunId(id=>id+1);lab.startRun();setAnnouncement(imported?'Replaying the recorded trace.':'Throwing your characters into an educational simulation.');}
  function tryNext(){switchStage(nextStage.id);lab.setMode('experiment');momentSeen.current=false;setLaunching(true);setRunId(id=>id+1);lab.startRun();setAnnouncement(`${nextStage.name}. Replaying the same input with a new way to remember.`);}
  function reset(){lab.resetExplorer();setMoment(false);setLaunching(false);setPaperFront(false);setHoveredId(null);setPopover(null);setShowExamples(false);setAnnouncement('Architecture assembled. View and filters reset.');}
  function changeParameter(key:keyof Sandbox,value:number){setSandbox(s=>({...s,[key]:value,...(key==='heads'?{head:Math.min(s.head,value-1)}:{})}));lab.setDraft(lab.draft);lab.selectPart(null);setMoment(false);setLaunching(false);setChanged(`${key==='forget'?'Memory keep tendency':key[0].toUpperCase()+key.slice(1)} → ${value}. ${{context:'The reachable input window changes.',embedding:'Wafers widen to represent more channels.',hidden:'State capacity changes; excluded clues stay excluded.',sequence:'More input means more processing steps.',retention:'The carried clue now fades or grows at this rate.',forget:'The valves change how much old state survives.',source:'The eligible retrieval window changes.',sharpness:'Ribbons concentrate on fewer clues as sharpness rises.',head:'A different retrieval pattern is highlighted.',heads:'Parallel retrieval branches enter or leave the assembly.',layers:'Repeated processing blocks enter or leave the stack.'}[key]}`);}
  function preset(name:string){setSandbox(s=>name==='Reset'?{...DEFAULT_SANDBOX,context:stage.id==='mlp'?3:8}:{...s,...PRESETS[name]});lab.resetExplorer();setMoment(false);setLaunching(false);setChanged(`${name} proportions applied. Notebook-like is illustrative; source notebooks have different dimensions.`);}
  function returnAssembly(){lab.selectPart(null);lab.setExplosion(0);setPaperFront(false);}
  function positionPaperThread(x:number,y:number){const area=sceneArea.current,thread=paperThread.current;if(!area||!thread)return;const parent=area.parentElement!.getBoundingClientRect(),rect=area.getBoundingClientRect();const mark=area.parentElement?.querySelector('.paper-cover mark')?.getBoundingClientRect();if(mark){thread.setAttribute('x1',String(mark.left-parent.left));thread.setAttribute('y1',String(mark.top+mark.height/2-parent.top));}thread.setAttribute('x2',String(x+rect.left-parent.left));thread.setAttribute('y2',String(y+rect.top-parent.top));}
  function onImport(next:Trace|null){setMoment(false);setLaunching(false);setPaperFront(false);setImported(next);lab.resetExplorer();lab.setMode('experiment');setDetail(null);setAnnouncement(next?'Imported recording ready to replay. Source authenticity is not verified.':'Returned to educational simulation.');}
  function outputCaption(){
    if(imported)return 'These are the recorded values from your imported trace.';
    if(stage.id==='mlp'&&trace.tokens.length<=sandbox.context)return 'Every character fit inside the window. Longer input would leave earlier clues behind.';
    if(stage.id==='mlp')return `Only the final ${Math.min(sandbox.context,trace.tokens.length)} characters reached the output. Earlier context was outside the window.`;
    if(stage.id==='rnn')return 'The state crossed the whole sequence. Each update compressed the past again.';
    if(stage.id==='lstm')return trace.steps[step].clueStrength>.25?'The gates kept a visible trace of the clue. Change memory keep tendency to test its survival.':'The clue faded along the gated path. Increase memory keep tendency to protect more of it.';
    if(stage.id==='attention')return lab.config.attentionEnabled?'Each target could reach back to its weighted source context.':'The source had to fit into a single capsule. Open the bridge to give each target a direct route back.';
    return trace.steps[step].clueStrength>0?'The chosen clue had a direct attention path to this position. This is sequence replay, not generated text.':'The chosen clue had no weight at this position. Try a longer context to make earlier positions reachable.';
  }

  return <div className={`museum-app stage-${lab.stage} ${lab.mode==='experiment'?'is-experiment':''} ${selected?'has-selection':''} ${reducedMotion?'reduce-motion':''}`}>
    <a className="skip-link" href="#component-outline" onClick={()=>setPopover('outline')}>Explore the accessible component outline</a>
    <header className="museum-header">
      <a className="museum-brand" href="#" onClick={event=>{event.preventDefault();reset();changeMode('learn');}} aria-label="Neural Atlas home"><Atom size={28} strokeWidth={1.15}/><span>neural<span>atlas</span><b>.</b></span></a>
      <div className="museum-header-actions"><a className="explore-code" href="https://github.com/ninadsonawane/neural-networks" target="_blank" rel="noopener noreferrer" title="Open the from-scratch notebooks">Explore the code<ArrowUpRight size={13} aria-hidden="true"/></a><span className={`museum-provenance ${imported?'is-recorded':''}`}><i/>{imported?'Imported recording':'Educational simulation'}</span><button className="museum-icon" aria-label="About Neural Atlas and data sources" title="About this atlas" onClick={()=>setDetail('about')}><CircleHelp size={18}/></button></div>
    </header>
    <nav className={`museum-rail ${lab.railExpanded?'expanded':''}`} aria-label="Evolution of model memory">
      <button className="rail-expander museum-icon" aria-label={lab.railExpanded?'Collapse evolution rail':'Expand evolution rail'} aria-expanded={lab.railExpanded} onClick={()=>lab.setRailExpanded(!lab.railExpanded)}>{lab.railExpanded?<ChevronLeft size={15}/>:<ChevronRight size={15}/>}</button>
      <span className="museum-rail-caption">FIVE WAYS TO REMEMBER</span>
      {STAGES.map(s=><button key={s.id} className={`museum-stage ${s.id===stage.id?'current':''}`} aria-label={`${s.number} ${s.name}, ${s.year}`} aria-current={s.id===stage.id?'step':undefined} title={`${s.name} · ${s.year}`} onClick={()=>switchStage(s.id)}><span>{s.number}</span><div>{s.name}<small>{s.year}</small></div></button>)}
      <span className="rail-end">2003 — 2017</span>
    </nav>
    <main className="museum-main" aria-label={`${stage.name} architecture explorer`}>
      <div className="museum-canvas-area" ref={sceneArea}><Suspense fallback={<div className="museum-loading"><Atom size={32} strokeWidth={1}/><span>Assembling the architecture</span></div>}><MuseumScene model={model} explosion={lab.explosion} selectedId={selected?.id??null} isolated={lab.isolated&&!!selected} visibleCategories={lab.visibleCategories} mode={lab.mode} playing={lab.playing&&!paperOpen} hasRun={lab.hasRun} step={step} resetKey={lab.cameraReset} reducedMotion={reducedMotion} flat={lab.focusMode} onSelect={selectPart} onHover={setHoveredId} duration={speed} launching={launching} inputCards={trace.tokens.map(t=>t.text)} momentId={moment?targetId:null} additionId={added?targetId:null} clueStrength={trace.steps[step].clueStrength} scan={scan} paperTarget={targetId} paperFront={paperFront||moment} onAnchor={positionPaperThread} onReturn={returnAssembly}/></Suspense></div>
      <div className="museum-intro">
        <div className="museum-eyebrow"><span>{stage.number} / 05</span><span>{architectureTitle} <i/> {paper.year}</span></div>
        <AnimatePresence mode="wait"><motion.div key={architectureTitle} initial={{opacity:0,y:reducedMotion?0:7}} animate={{opacity:1,y:0}} exit={{opacity:0,y:reducedMotion?0:-5}} transition={transition}><h1 className={`architecture-title title-${stage.id}`}>{architectureTitle}</h1><p className="architecture-intuition">{stage.id==='lstm'&&gru?'Two gates. One compact memory path.':INTUITION[stage.id]}</p></motion.div></AnimatePresence>
        {stage.id==='lstm'&&!imported&&<div className="gated-variant"><button aria-pressed={!gru} onClick={()=>{lab.setConfig({gatedVariant:'lstm'});lab.selectPart(null);setAdded('Separate cell memory and readout.');}}>LSTM</button><button aria-pressed={gru} onClick={()=>{lab.setConfig({gatedVariant:'gru'});lab.selectPart(null);setAdded('One state. Two gates.');}}>GRU</button></div>}
        <label className="museum-mobile-stage"><span className="sr-only">Choose architecture</span><select aria-label="Choose architecture" value={stage.id} onChange={e=>switchStage(e.target.value as Architecture)}>{STAGES.map(s=><option value={s.id} key={s.id}>{s.number} · {s.name}</option>)}</select><ChevronDown size={13}/></label>
      </div>
      <div className="museum-tools" aria-label="Explorer utilities"><button className={`museum-icon ${scan?'on':''}`} aria-label="Scan mode" aria-pressed={scan} title="Scan to reveal labels" onClick={()=>setScan(!scan)}><Focus size={17}/></button><button className={`museum-icon ${popover==='outline'?'on':''}`} title="Component outline" aria-label="Open accessible component outline" aria-expanded={popover==='outline'} onClick={()=>setPopover(popover==='outline'?null:'outline')}><ListTree size={17}/></button><button className={`museum-icon ${lab.focusMode?'on':''}`} aria-label="Use accessible 2D view" aria-pressed={lab.focusMode} title="2D view" onClick={()=>lab.setFocusMode(!lab.focusMode)}><Eye size={17}/></button><button className="museum-icon" title="Compare the five ideas" aria-label="Compare architectures" onClick={()=>setDetail('compare')}><GitCompareArrows size={17}/></button>{lab.mode==='experiment'&&<button className="museum-icon" title="Experiment settings" aria-label="Open experiment settings" onClick={()=>setDetail('mechanism')}><Settings2 size={17}/></button>}</div>
      {!filteredParts.length&&<div className="museum-empty"><Layers3 size={26}/><p>Every layer is hidden.</p><button onClick={lab.showAllCategories}>Show all components</button></div>}
      {hovered&&!selected&&<div className="museum-hover" aria-hidden="true"><span className="hover-dot"/>{hovered.label}<small>Click to discover</small></div>}
      <AnimatePresence>{selected&&<motion.aside key="selection" className="museum-selection glass-surface" aria-label="Selected component" initial={{opacity:0,y:reducedMotion?0:12}} animate={{opacity:1,y:0}} exit={{opacity:0,y:reducedMotion?0:8}} transition={transition}>
        <div className="selection-kicker"><span>FOCUS / {PART_CATEGORIES.find(c=>c.id===selected.category)?.label}</span><button className="museum-icon" aria-label="Close component details" onClick={returnAssembly}><X size={15}/></button></div><h2 ref={selectionHeading} tabIndex={-1}>{selected.sceneLabel??selected.label}</h2><p>{selected.why}</p><div className="selection-actions"><button className="museum-primary" aria-pressed={lab.isolated} onClick={()=>lab.setIsolated(!lab.isolated)}><Focus size={14}/>{lab.isolated?'Show in context':'Focus component'}</button><button className="museum-text" onClick={()=>setDetail(selected.category==='output'?'probabilities':'mechanism')}>Go deeper <ArrowUpRight size={13}/></button></div>
      </motion.aside>}</AnimatePresence>
      {popover==='filters'&&<ArchitecturePlayground stage={stage.id} value={sandbox} onChange={changeParameter} onPreset={preset} visible={lab.visibleCategories} onToggle={lab.toggleCategory} onClose={()=>setPopover(null)} changed={changed} imported={!!imported} gru={gru} onVariant={isGru=>{lab.setConfig({gatedVariant:isGru?'gru':'lstm'});lab.selectPart(null);}}/>}
      <AnimatePresence>{popover==='outline'&&<motion.aside className="museum-popover glass-surface component-outline" id="component-outline" aria-label="Architecture component outline" initial={{opacity:0,y:reducedMotion?0:8}} animate={{opacity:1,y:0}} exit={{opacity:0,y:reducedMotion?0:8}} transition={transition}>
        <div className="popover-title"><span>{popover==='outline'?'Inside this architecture':'Layers of the idea'}</span><button className="museum-icon" aria-label="Close explorer panel" onClick={()=>setPopover(null)}><X size={15}/></button></div>
        <><p>The same selectable pieces, without the canvas.</p><div className="outline-content">{presentCategories.map(category=><details key={category.id} open><summary><span style={{background:category.color}}/>{category.label}<small>{model.parts.filter(p=>p.category===category.id).length}</small></summary><ul>{model.parts.filter(p=>p.category===category.id).map(part=><li key={part.id}><button aria-pressed={selected?.id===part.id} onClick={()=>{if(!lab.visibleCategories.includes(part.category))lab.toggleCategory(part.category);selectPart(part.id,true);}}><span>{part.label}</span><ArrowUpRight size={12}/></button></li>)}</ul></details>)}</div></>
      </motion.aside>}</AnimatePresence>
      <svg className={`origin-thread ${paperFront||moment?'visible':''}`} aria-hidden="true"><line ref={paperThread} x1="85%" y1="52%" x2="60%" y2="50%"/></svg>
      {!selected&&popover!=='filters'&&<PaperOrigin paper={paper} front={paperFront||moment} open={paperOpen} onFront={()=>setPaperFront(!paperFront)} onOpen={()=>{setPaperFront(true);setPaperOpen(true);}} onClose={()=>setPaperOpen(false)} reduced={reducedMotion}/>}
      <AnimatePresence>{(moment||added)&&!selected&&<motion.div className="intuition-moment" role="status" initial={{opacity:0,y:reducedMotion?0:12}} animate={{opacity:1,y:0}} exit={{opacity:0}}><span>{moment?'THE IDEA, MADE VISIBLE':'THE NEXT IDEA'}</span><strong>{moment?(stage.id==='mlp'&&trace.steps[step].clueStrength>0?'The clue still fits the window.':stage.id==='lstm'&&trace.steps[step].clueStrength<.25?'Closed gates let the clue fade.':MOMENT[stage.id]):added}</strong></motion.div>}</AnimatePresence>
      {lab.mode==='experiment'&&<div className={`museum-experiment ${selected?'with-selection':''}`}>
        {lab.hasRun&&<div className="cinema-transport"><button className="museum-icon" aria-label="Replay from start" onClick={run}><RotateCcw size={13}/></button><button className="museum-text" aria-label="Change slow motion speed" onClick={()=>setSpeed(speed===1800?3000:1800)}>{speed===1800?'Slow motion':'Extra slow'} · {speed/1000}s / token</button><span>{launching?'Casting the cards…':moment?'Intuition pause':`Clue ${trace.tokens.find(t=>t.isClue)?.text??'·'} · ${Math.round(trace.steps[step].clueStrength*100)}% ${imported?'recorded':'teaching signal'}`}</span></div>}
        <AnimatePresence>{complete&&!selected&&<motion.div className="experiment-result glass-surface" initial={{opacity:0,y:reducedMotion?0:10}} animate={{opacity:1,y:0}} exit={{opacity:0}} transition={transition}><div><span className="result-glyph"><Check size={17}/></span><span className="result-eyebrow">{imported?'RECORDING REPLAYED':'THE SEQUENCE HAS ARRIVED'}</span><button className="museum-text" onClick={()=>setDetail('probabilities')}>See probabilities <ArrowUpRight size={12}/></button></div><p>{outputCaption()}</p>{!imported&&<button className="museum-primary" onClick={tryNext}>{stage.id==='transformer'?'Revisit the first idea':'Try the next idea'}<ArrowRight size={15}/></button>}</motion.div>}</AnimatePresence>
        <div className="experiment-command glass-surface"><form className="experiment-input-row" onSubmit={event=>{event.preventDefault();tokenize();}}><span className="input-chevron">›</span><input aria-label="Your experiment text" maxLength={160} spellCheck={false} value={imported?.input??lab.draft} disabled={!!imported} onChange={event=>{setShowCluePicker(false);lab.setDraft(Array.from(event.target.value).slice(0,80).join(''));lab.setPlaying(false);}} placeholder="Give the model something to remember…"/><button type="submit" disabled={!!imported} className="tokenize-quiet">Tokenize</button><span className="command-divider"/><button type="button" className={`${!complete&&!selected?'museum-primary':'museum-text'} run-control`} onClick={()=>lab.playing?lab.setPlaying(false):lab.hasRun&&!complete?lab.setPlaying(true):run()}>{lab.playing?<Pause size={14}/>:complete?<RotateCcw size={14}/>:<Play size={14}/>}<span>{lab.playing?'Pause':complete?'Replay':lab.hasRun?'Resume':'Throw it in'}</span></button></form>
          <div className="experiment-caption">{!imported&&<div className="clue-picker"><button type="button" className="museum-text" aria-expanded={showCluePicker} aria-controls="clue-picker-panel" onClick={()=>{if(dirty)lab.tokenize();setShowCluePicker(!showCluePicker);}}>Pick clue<ChevronDown size={11}/></button>{showCluePicker&&<div id="clue-picker-panel" className="clue-picker-panel" onKeyDown={event=>{if(event.key==='Escape'){event.stopPropagation();setShowCluePicker(false);}}}><label htmlFor="clue-choice">Follow a character</label><select id="clue-choice" value={clueIndex} onChange={event=>{setClueChoice({input:effectiveInput,index:Number(event.target.value)});lab.setDraft(lab.draft);setMoment(false);setLaunching(false);setShowCluePicker(false);}}>{Array.from(effectiveInput).map((char,index)=><option key={index} value={index}>{index+1} · {char===' '?'space':char} — {Array.from(effectiveInput).slice(Math.max(0,index-5),index+10).join('')}</option>)}</select></div>}</div>}<button className="museum-text" onClick={()=>setShowExamples(!showExamples)} aria-expanded={showExamples}>{trace.tokens.length} {imported?'tokens':'characters'}<ChevronDown size={11}/></button>{lab.hasRun?<div className="experiment-scrub"><button className="museum-icon" aria-label={imported?"Previous token":"Previous character"} disabled={step===0} onClick={()=>{lab.setPlaying(false);lab.setStep(step-1);}}><ChevronLeft size={12}/></button><input aria-label="Simulation timeline" type="range" min={0} max={Math.max(1,trace.tokens.length-1)} value={step} disabled={trace.tokens.length===1} onChange={e=>{lab.setPlaying(false);lab.setStep(Number(e.target.value));}}/><button className="museum-icon" aria-label={imported?"Next token":"Next character"} disabled={step===trace.tokens.length-1} onClick={()=>{lab.setPlaying(false);lab.setStep(step+1);}}><ChevronRight size={12}/></button><span>{String(step+1).padStart(2,'0')} / {trace.tokens.length}</span></div>:<span>{dirty?'Throw your new characters into the machine.':imported?'Recorded values · source unverified':'Throw the tiles. Follow the clue.'}</span>}</div>
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
    <Suspense fallback={null}>{detail&&<ExplorerDetails open={true} onClose={()=>setDetail(null)} tab={detail} trace={trace} step={step} config={activeConfig} onConfig={partial=>{lab.setConfig(partial);setSandbox(s=>({...s,...(partial.contextLength!==undefined?{context:partial.contextLength}:{}),...(partial.recurrentMultiplier!==undefined?{retention:partial.recurrentMultiplier}:{}),...(partial.sourceLength!==undefined?{source:partial.sourceLength}:{}),...(partial.head!==undefined?{head:partial.head}:{}),...(partial.layer!==undefined?{layers:partial.layer+1}:{}),...(partial.gateOverrides?.forget!==undefined?{forget:partial.gateOverrides.forget}:{})}));}} onStep={lab.setStep} decoderStep={decoderStep} onDecoderStep={setDecoderStep} onImport={onImport} onExplore={switchStage}/>}</Suspense>
  </div>;
}
