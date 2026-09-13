import test from 'node:test';
import assert from 'node:assert/strict';
import { buildTrace, DEFAULT_CONFIG } from '../simulation/engine';
import { loadTraceJson } from '../simulation/adapter';
import { buildExplorerModel } from './models';
import { cinematicModel, DEFAULT_SANDBOX, momentAt, sandboxConfig } from './cinema';
import type { Architecture } from '../simulation/types';

test('cinematic routes follow real diagram edges and topology remains bounded',()=>{
  for(const architecture of ['mlp','rnn','lstm','attention','transformer'] as Architecture[])for(const heads of [1,4]){
    const settings={...DEFAULT_SANDBOX,heads,head:heads-1,layers:heads};const config=sandboxConfig(DEFAULT_CONFIG,settings,architecture);
    const trace=buildTrace('a distant clue reaches the end',architecture,config),step=15;
    const base=buildExplorerModel({trace,step,config,decoderStep:step%trace.decoderTokens.length||0});const model=cinematicModel(base,settings,trace,step,true);
    const ids=new Set(model.parts.map(p=>p.id));assert.equal(ids.size,model.parts.length);
    assert.ok(model.parts.length<130);
    for(const part of model.parts)assert.ok([...part.position,...part.scale,...part.explodedPosition].every(Number.isFinite));
    for(const link of model.links)assert.ok(ids.has(link.from)&&ids.has(link.to));
    for(let i=1;i<model.flow.length;i++)assert.ok(model.links.some(l=>l.from===model.flow[i-1]&&l.to===model.flow[i]),`${architecture}: ${model.flow[i-1]} → ${model.flow[i]}`);
    if(architecture==='transformer'){assert.equal(model.parts.filter(p=>p.kind==='head').length,heads);assert.equal(model.parts.filter(p=>p.id.startsWith('layer-')).length,heads-1);assert.ok(model.parts.some(p=>p.sceneLabel==='Query'&&!p.detailOnly));}
  }
});
test('user-character attention uses custom text, chosen clue, source limits, and normalized deterministic weights',()=>{
  const config={...DEFAULT_CONFIG,customAttention:true,sourceLength:4,clueIndex:2,attentionSharpness:2};
  const trace=buildTrace('abcd efgh','attention',config);
  assert.deepEqual(trace.encoderTokens,Array.from('abcd efgh'));assert.ok(trace.tokens[2].isClue);
  assert.deepEqual(trace,buildTrace('abcd efgh','attention',config));
  trace.alignment.forEach((row,i)=>{assert.ok(Math.abs(row.reduce((a,b)=>a+b,0)-1)<1e-10);row.forEach((weight,j)=>{if(j>i||j<i-3)assert.equal(weight,0);});});
  assert.ok(trace.alignment[3][2]>trace.alignment[3][1]);assert.equal(trace.steps.at(-1)!.clueStrength,0);
  assert.equal(loadTraceJson(JSON.stringify(trace),'custom').steps.length,trace.steps.length);
});
test('Transformer context has zero paths outside its selected window',()=>{
  const trace=buildTrace('abcdefghijk','transformer',{...DEFAULT_CONFIG,attentionContext:3});
  trace.attention.forEach((row,i)=>row.forEach((w,j)=>{if(j>i||j<i-2)assert.equal(w,0);}));
});
test('MLP long windows change reachable input and moment stays within the trace',()=>{
  const short=buildTrace('abcdefghijklmnopqrstuvwxyz','mlp',{...DEFAULT_CONFIG,contextLength:3});
  const long=buildTrace(short.input,'mlp',{...DEFAULT_CONFIG,contextLength:32});
  assert.equal(short.steps.at(-1)!.retainedIndices.length,3);assert.equal(long.steps.at(-1)!.retainedIndices.length,26);
  assert.notDeepEqual(short.steps.at(-1)!.hidden,long.steps.at(-1)!.hidden);
  assert.ok(momentAt(long,32)<long.steps.length);
});
test('cinematic imported rendering ignores sandbox dimensions and never invents head internals',()=>{
  const trace=loadTraceJson(JSON.stringify(buildTrace('the clue returns','transformer',DEFAULT_CONFIG)),'recording');
  const base=buildExplorerModel({trace,step:9,config:DEFAULT_CONFIG,decoderStep:0});
  const tiny=cinematicModel(base,{...DEFAULT_SANDBOX,hidden:16,embedding:8,heads:1,layers:1},trace,9,true);
  const wide=cinematicModel(base,{...DEFAULT_SANDBOX,hidden:512,embedding:128,heads:4,layers:4},trace,9,true);
  assert.deepEqual(tiny,wide);
});
