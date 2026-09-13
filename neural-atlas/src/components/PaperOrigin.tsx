import { useEffect, useRef } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { ArrowUpRight, X } from 'lucide-react';
import type { ResearchPaper } from '../data/papers';

export default function PaperOrigin({paper,front,open,onFront,onOpen,onClose,reduced}:{paper:ResearchPaper;front:boolean;open:boolean;onFront:()=>void;onOpen:()=>void;onClose:()=>void;reduced:boolean}) {
  const dialog=useRef<HTMLDialogElement>(null);
  useEffect(()=>{if(open&&!dialog.current?.open)dialog.current?.showModal();if(!open&&dialog.current?.open)dialog.current.close();},[open]);
  return <>
    <div className={`paper-origin ${front?'paper-forward':''}`}>
      <AnimatePresence mode="wait"><motion.button key={paper.id} className="paper-cover" onClick={onOpen} aria-label={`Read about ${paper.title}`} initial={reduced?false:{opacity:0,rotateY:-80,x:40}} animate={{opacity:1,rotateY:0,x:0}} exit={reduced?{opacity:0}:{opacity:0,rotateY:85,x:40}} transition={{duration:reduced?0:.85}}>
        <span className="paper-archive">NEURAL MEMORY / RESEARCH ARCHIVE</span><span className="paper-year">{paper.year}</span><strong>{paper.title}</strong><span className="paper-authors">{paper.authors}</span><span className="paper-rule"/><mark>{paper.contribution}</mark><span className="paper-foot">Idea summary · click to read <ArrowUpRight size={12}/></span>
      </motion.button></AnimatePresence>
      <button className="paper-why museum-text" onClick={onFront}>Why was this invented? <ArrowUpRight size={12}/></button>
    </div>
    <dialog className="paper-dialog" ref={dialog} onCancel={e=>{e.preventDefault();onClose();}} onClick={e=>{if(e.target===e.currentTarget)onClose();}} aria-label="Research paper details">
      <button autoFocus className="paper-close museum-icon" aria-label="Close paper details" onClick={onClose}><X size={18}/></button><span className="paper-archive">THE IDEA BEHIND THE MACHINE · {paper.year}</span><h2>{paper.title}</h2><p>{paper.authors}</p><p className="paper-contribution">{paper.contribution}</p>{paper.note&&<small>{paper.note}</small>}<a className="paper-read" href={paper.url} target="_blank" rel="noreferrer">Read the paper ↗</a>{paper.related&&<details><summary>Related research</summary>{paper.related.map(p=><a href={p.url} key={p.url} target="_blank" rel="noreferrer">{p.title} ↗</a>)}</details>}
    </dialog>
  </>;
}
