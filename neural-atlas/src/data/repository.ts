export const REPO_URL = 'https://github.com/ninadsonawane/neural-networks';

export const REPOSITORY_FACTS: {
  architecture: string;
  status: 'Notebook experiment' | 'Planned';
  details: string;
  file: string | null;
}[] = [
  {
    architecture: 'Fixed-context MLP',
    status: 'Notebook experiment',
    details: 'Character vocabulary: 28 including “.”. The supplied notebook sets context 3, embedding 8, and hidden width 200. Loss ≈1.89 is a separately documented context-6 / embedding-6 run, not a cross-model benchmark.',
    file: 'MLP.ipynb',
  },
  {
    architecture: 'Vanilla RNN',
    status: 'Notebook experiment',
    details: 'One-hot character input, 50-character training windows, shared tanh recurrence, BPTT, and gradient clipping at norm 5. Executable code sets hidden width 300; older inline comments say 100.',
    file: 'RNN.ipynb',
  },
  {
    architecture: 'LSTM',
    status: 'Notebook experiment',
    details: '100-character sequences, explicit input/forget/candidate/output gates, dropout, clipped gradients, and loss/perplexity tracking. Executable code sets hidden width 256. Checkpoint-saving code exists; no checkpoint file was supplied.',
    file: 'LSTM.ipynb',
  },
  {
    architecture: 'GRU & attention bridge',
    status: 'Planned',
    details: 'GRU, encoder–decoder, seq2seq, and Bahdanau attention are planned in the repository. Their gates, compression demo, and English/Hinglish word alignment here are conceptual simulations, with no measured translation results.',
    file: null,
  },
  {
    architecture: 'Causal Transformer',
    status: 'Notebook experiment',
    details: 'Character-level GPT experiment with context 8, embedding 32, four layers, four heads, token + learned position embeddings, a causal mask, residual blocks, and AdamW. Self-attention is also explored in a separate notebook. This site runs a teaching simulation.',
    file: 'transformer.ipynb',
  },
];
