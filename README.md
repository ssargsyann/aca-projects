# DIY Transformer: An Encoder-Decoder Seq2Seq Architecture from Scratch

A from-scratch PyTorch implementation of the Transformer architecture ("Attention Is All You Need", Vaswani et al., 2017), built as a complete Encoder-Decoder Seq2Seq model and trained on a character-level date-translation task.

This repository is both a **learning resource** and a **reference implementation**: it walks through *why* the Transformer works — the recurrent bottleneck it solves, sub-word tokenization, positional encoding, and multi-head attention — before assembling every architectural component (self-attention, cross-attention, masking, residual connections) into a working, trainable model.

## Overview

The Transformer replaced recurrence with **parallelizable self-attention**, enabling the large-scale language models that followed it. This project reconstructs that architecture piece by piece:

- **Self-Attention & Multi-Head Attention** — computing pairwise token relationships via scaled dot-product attention, split across multiple heads.
- **Positional Encoding** — injecting sequence order into a model that has no inherent notion of position.
- **Masking** — causal masks (to preserve autoregressive generation) and padding masks (to ignore `<PAD>` tokens).
- **Encoder-Decoder Cross-Attention** — the "information bridge" that lets the decoder condition its output on the full source sequence.

The final model is trained end-to-end on a synthetic task: translating human-readable dates (`August 3, 2026`) into ISO-8601 format (`2026-08-03`), which is small enough to train quickly on a CPU or single GPU while still exercising the full architecture.

## Project Structure

The project is contained in a single, self-documented Jupyter notebook: [`transformer_from_scratch.ipynb`](./transformer_from_scratch.ipynb).

### Part 1: Instructor Demos

Four short, standalone demonstrations that build intuition for *why* each architectural choice was made:

| Demo | Concept |
|---|---|
| **1. The Recurrent Bottleneck** | Benchmarks a sequential RNN loop against a single parallelized matrix multiplication, illustrating why recurrence doesn't scale. |
| **2. Sub-Word Tokenization (BPE)** | Uses a pretrained BPE tokenizer to show how sub-word units balance vocabulary size against out-of-vocabulary robustness. |
| **3. Positional Encoding** | Derives and visualizes sinusoidal positional encodings, and shows how their dot product decays with relative distance. |
| **4. Multi-Head Splitting & Causal Masking** | Demonstrates that "multi-head" attention is a tensor reshape (not multiple layers), and walks through causal mask construction. |

### Part 2: The DIY Transformer

A complete Encoder-Decoder Seq2Seq Transformer, implemented from scratch in PyTorch and trained on the character-level date-translation task described above:

- **`TokenAndPositionEmbedding`** — combines token embeddings with learned positional embeddings.
- **`EncoderBlock`** — self-attention followed by a feed-forward network, each with a residual connection and `LayerNorm`.
- **`DecoderBlock`** — masked self-attention, cross-attention over the encoder's output ("memory"), and a feed-forward network.
- **`Seq2SeqTransformer`** — assembles stacked encoder and decoder layers into the full model, including causal-mask and padding-mask generation.
- **Training loop** — trains the model end-to-end with `CrossEntropyLoss`, tracked via a `tqdm` progress bar.
- **Autoregressive inference** — greedy decoding, generating one character at a time until `<EOS>`.
- **Bonus: Cross-Attention Heatmap** — visualizes the decoder's cross-attention weights, showing which source characters the model attends to when generating each output character.

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/ssargsyann/seq2seq-transformer.git
cd seq2seq-transformer
```

### 2. Install dependencies

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook transformer_from_scratch.ipynb
```

Run all cells sequentially. Part 2 trains in a few minutes on CPU (or seconds on GPU), reaching a low loss within 5–10 epochs.

## Author

**Sargis Sargsyan**
