import type { Architecture, SimulationConfig, Trace } from '../simulation/types';

export type PartCategory = 'tokens' | 'embeddings' | 'memory' | 'gates' | 'attention' | 'residual' | 'output';
export type ExplorerMode = 'learn' | 'experiment';
export type Vec3 = [number, number, number];
export type PartKind = 'card' | 'slab' | 'sphere' | 'valve' | 'ring' | 'head' | 'bundle';
export interface ExplorerPart {
  id: string;
  label: string;
  category: PartCategory;
  kind: PartKind;
  position: Vec3;
  explodedPosition: Vec3;
  scale: Vec3;
  color?: string;
  purpose: string;
  why: string;
  math?: string;
  resource?: string;
  tokenIndex?: number;
  decoderIndex?: number;
  parentId?: string;
  value?: number;
  ghost?: boolean;
  active?: boolean;
  detailOnly?: boolean;
  sceneLabel?: string;
  cells?: number;
  clue?: boolean;
  echo?: number;
  signal?: number;
}
export interface ExplorerLink {
  id: string;
  from: string;
  to: string;
  weight?: number;
  kind: 'flow' | 'memory' | 'attention' | 'gradient' | 'residual';
}
export interface ExplorerModel {
  architecture: Architecture;
  parts: ExplorerPart[];
  links: ExplorerLink[];
  flow: string[];
  description: string;
}
export interface ModelBuildOptions {
  trace: Trace;
  step: number;
  config: SimulationConfig;
  decoderStep: number;
}
export const PART_CATEGORIES: {id:PartCategory;label:string;color:string}[] = [
  {id:'tokens',label:'Tokens',color:'#cbd9e5'},
  {id:'embeddings',label:'Embeddings',color:'#97c8ca'},
  {id:'memory',label:'Memory & state',color:'#a5e8ca'},
  {id:'gates',label:'Gates',color:'#d4c996'},
  {id:'attention',label:'Attention',color:'#83d7de'},
  {id:'residual',label:'Residual stream',color:'#b4b2dd'},
  {id:'output',label:'Output',color:'#bce9dc'},
];
