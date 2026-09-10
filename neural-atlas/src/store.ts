import { create } from 'zustand';
import { DEFAULT_CONFIG } from './simulation/engine';
import type { Architecture, SimulationConfig } from './simulation/types';
import { PART_CATEGORIES } from './explorer/types';
import type { ExplorerMode, PartCategory } from './explorer/types';
export const PROMPTS = ['the cat that chased the mouse was ...', 'chai thandi thi par yaad garam hai', 'anna, arjun, anaya, aarav'];
interface LabState {
  mode: ExplorerMode; explosion: number; selectedId: string|null; isolated: boolean;
  visibleCategories: PartCategory[]; railExpanded: boolean; cameraReset: number; hasRun: boolean;
  setMode:(mode:ExplorerMode)=>void; setExplosion:(explosion:number)=>void;
  selectPart:(selectedId:string|null)=>void; setIsolated:(isolated:boolean)=>void;
  toggleCategory:(category:PartCategory)=>void; showAllCategories:()=>void;
  setRailExpanded:(railExpanded:boolean)=>void; resetExplorer:()=>void; startRun:()=>void;
  input: string; draft: string; stage: Architecture; step: number; playing: boolean;
  config: SimulationConfig; challenge: boolean; density: 'story' | 'xray'; focusMode: boolean;
  setDraft: (draft:string)=>void; tokenize: ()=>void; setInput:(input:string)=>void;
  setStage:(stage:Architecture)=>void; setStep:(step:number)=>void;
  setPlaying:(playing:boolean)=>void; setConfig:(config:Partial<SimulationConfig>)=>void;
  setChallenge:(challenge:boolean)=>void; setDensity:(density:'story'|'xray')=>void; setFocusMode:(focusMode:boolean)=>void;
}
export const useLab = create<LabState>((set)=>({
  mode:'learn',explosion:0,selectedId:null,isolated:false,
  visibleCategories:PART_CATEGORIES.map(c=>c.id),railExpanded:false,cameraReset:0,hasRun:false,
  setMode:(mode)=>set({mode,playing:false,selectedId:null,isolated:false}),
  setExplosion:(explosion)=>set({explosion:Math.max(0,Math.min(1,explosion))}),
  selectPart:(selectedId)=>set({selectedId,isolated:false}),setIsolated:(isolated)=>set({isolated}),
  toggleCategory:(category)=>set(s=>({visibleCategories:s.visibleCategories.includes(category)?s.visibleCategories.filter(c=>c!==category):[...s.visibleCategories,category],selectedId:null,isolated:false})),
  showAllCategories:()=>set({visibleCategories:PART_CATEGORIES.map(c=>c.id)}),
  setRailExpanded:(railExpanded)=>set({railExpanded}),
  resetExplorer:()=>set(s=>({explosion:0,selectedId:null,isolated:false,visibleCategories:PART_CATEGORIES.map(c=>c.id),playing:false,hasRun:false,step:0,cameraReset:s.cameraReset+1})),
  startRun:()=>set({step:0,playing:true,hasRun:true,selectedId:null,isolated:false}),
  input:PROMPTS[0], draft:PROMPTS[0], stage:'mlp', step:15, playing:false,
  config:{...DEFAULT_CONFIG,head:0,layer:0,attentionEnabled:true}, challenge:false, density:'story', focusMode:false,
  setDraft:(draft)=>set({draft,hasRun:false,playing:false}), tokenize:()=>set(s=>({input:Array.from(s.draft || ' ').slice(0,80).join(''),draft:Array.from(s.draft || ' ').slice(0,80).join(''),step:0,playing:false,hasRun:false,selectedId:null,isolated:false})),
  setInput:(input)=>set({input,draft:input,step:0,playing:false,hasRun:false,selectedId:null,isolated:false}), setStage:(stage)=>set({stage,playing:false,hasRun:false,selectedId:null,isolated:false}),
  setStep:(step)=>set({step:Math.max(0,step)}), setPlaying:(playing)=>set({playing}),
  setConfig:(config)=>set(s=>({config:{...s.config,...config}})),
  setChallenge:(challenge)=>set({challenge,step:0,playing:false,hasRun:false,selectedId:null,isolated:false}),
  setDensity:(density)=>set({density}), setFocusMode:(focusMode)=>set({focusMode}),
}));
