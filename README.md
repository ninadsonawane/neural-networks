Neural Networks From Scratch — Language Modeling Experiments

A practical re-implementation of foundational neural language modeling papers (2003–2017).

This repository documents a complete journey through the core evolution of neural language models — from Bengio’s MLP (2003) to RNNs (2010), LSTMs (2013), GRUs (2014), and the Transformer (2017).

Each model is implemented from first principles using only PyTorch basics (no high-level abstractions), with a running emphasis on:

what each paper actually introduced

what problems it tried to solve

what breaks in practice

how loss behaves

how far each architecture can model long-range dependencies

1. Repository Structure
neural-networks/
│
├── data/                  # Tiny text dataset for experiments
├── mlp/                   # Bengio et al. 2003 implementation
├── rnn/                   # Mikolov RNNLM + BPTT
├── lstm/                  # Graves 2013 LSTM
├── gru/                   # Cho et al. GRU encoder-decoder
├── transformer/           # Scaled dot-product attention + multi-head + FFN
│
├── utils/                 # batching, text encoding, weight init, metrics
└── README.md              # (You are reading it)


Every directory contains:

model.py

train.py

generate.py (where applicable)

2. Running the Models

All models follow the same pattern:

Install
pip install torch numpy tqdm

Train
python train.py --epochs 20000 --lr 1e-3

Generate text (RNN/LSTM/GRU/Transformer)
python generate.py --prompt "the"

Inputs

raw text → tokenized to integer IDs

batches created with sliding window or sequential mini-batching

default vocab: all characters present in the dataset

Outputs

next-token probability distribution

sampling uses:

argmax

multinomial sampling

temperature scaling

Weights

all models save:

checkpoints/model.pt


load via:

torch.load('checkpoints/model.pt')

3. Paper-by-Paper Implementations

Below is the core logic of each paper and what the implementation tries to replicate.

Bengio et al., 2003 — Neural Probabilistic Language Model (MLP)

Paper: A Neural Probabilistic Language Model (2003)
Goal: Beat n-gram models using a simple feedforward network with learned word embeddings.

Key Concepts Implemented

learned embeddings

fixed context window

feedforward network predicting next token

cross-entropy loss

Observed Loss (My Experiments)
Configuration	Loss Trend
emb=2, ctx=2	2.262 → 2.416 → 2.46 (unstable)
emb=6	2.48 → 1.96
emb=16, ctx=8	~2.5 (no improvement)
emb=6, ctx=6 (30k steps)	2.5 → 1.89
with BatchNorm	stable, smoother convergence
Why MLP fails

fixed window → no long-range memory

cannot hallucinate or track state

deeper MLPs still do not fix context limitation

This is exactly what Bengio reported: MLPs can't carry sequence structure.

Mikolov (2010) — Recurrent Neural Network Language Model (RNNLM)

Paper: Recurrent Neural Network Based Language Model
Goal: Infinite context via recurrence.

Key Concepts Implemented

simple RNN (Elman-type)

hidden state carries past information

Backpropagation Through Time (BPTT)

online learning / adaptive updating

Observed Behavior

learns local patterns well

quickly overfits if hidden size too small

vanishing gradients visible after ~30–50 steps

generation produces repetitive or drifting sequences

RNN “hallucination” is natural — once the hidden state drifts, model doubles down on its own errors.

Graves (2013) — Generating Sequences With RNNs (LSTM)

Paper: Generating Sequences with Recurrent Neural Networks
Goal: Fix vanishing gradient using LSTM memory cells.

Key Concepts Implemented

input, forget, output gates

cell state with additive gradients

stable long-term memory

character-level text generation

Observed Behavior

significantly lower loss than RNN

maintains coherence for 40–100+ characters

stops drifting mid-sentence

learns parentheses, quotes, nested structure (just like the paper)

LSTMs finally make character modeling practical.

Cho et al. (2014) — GRU Encoder–Decoder

Paper: On the Properties of Neural Machine Translation: Encoder–Decoder Approaches
Goal: Simplify LSTM and use it for translation.

Key Concepts Implemented

update gate, reset gate

encoder compresses entire sequence to a vector

decoder expands vector into translated sequence

Observations

trains faster than LSTM

struggles with very long sequences (as expected)

bottleneck vector → loses information beyond 20–30 tokens

This matches the paper’s conclusion.

Bahdanau et al. (2015) — Attention

implemented additive attention

decoder attends over all encoder states

resolves the bottleneck problem

Vaswani et al. (2017) — Transformer

Paper: Attention Is All You Need
Goal: Remove recurrence. Model long-range dependencies with self-attention.

Key Components Implemented

scaled dot-product attention

multi-head attention

position encodings (sinusoidal)

feedforward layers

decoder masking

Observed Behavior

most sample-efficient model

learns long-range structure immediately

training is highly parallel

lowest perplexity among all models tested

Transformers are simply dominant.

4. Training Details
Weight Initialization

uniform or Xavier

recurrent matrices orthogonal (for stability)

Batching

contiguous sequential batches for RNN/LSTM/GRU

random block batches for MLP/Transformer

Optimizers Tried

SGD

Adam (best)

RMSProp (for LSTM stability)

Gradient Issues

RNN suffers → needed gradient clipping

Transformer needed warmup learning rate schedule

5. Limitations & Future Work
MLP

cannot model long sequences

unstable without BatchNorm

RNN

vanishing gradients

drifts during generation

LSTM/GRU

better but slow

limited parallelism

Transformer

best performance

expensive but scalable

Future work:

add multi-layer transformers

add byte-pair encoding tokenizer

train on larger corpora

experiment with rotary embeddings

6. Why This Repo Exists

Reproducing these papers step-by-step is the only way to truly internalize:

how neural sequence models actually evolved

why each architectural change mattered

what breaks in practice

what “long-range dependency” problems actually look like

how modern LLMs emerged

This repo acts as a personal deep dive into foundational NLP.
