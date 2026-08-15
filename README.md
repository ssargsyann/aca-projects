# Unboxing the Transformer: Decoding Strategies from Scratch

A from-scratch PyTorch implementation of the core LLM text-generation
mechanics — temperature scaling, Top-K sampling, and Top-P (nucleus)
sampling — built directly on top of GPT-2's raw logits, bypassing
`transformers`' high-level `.generate()` API.

## Overview

Most tutorials treat `model.generate()` as a black box. This project opens
it up: it walks through exactly how a decoder-only transformer's raw output
(a vector of ~50,000 unnormalized scores) gets converted into the next word
of generated text, and reimplements each stage of that pipeline by hand:

1. **Tokenization** — mapping text to integer IDs.
2. **Logits** — inspecting the model's raw, unnormalized next-token scores.
3. **Greedy decoding** — the simplest (deterministic) decoding rule.
4. **Temperature scaling** — reshaping the output distribution's randomness.
5. **Top-K masking** — restricting sampling to a fixed-size candidate pool.
6. **Top-P (nucleus) masking** — restricting sampling to a confidence-adaptive candidate pool.
7. **Autoregressive generation** — composing all of the above into a full sampling loop.

## Project Structure

```
.
├── decoding_strategies.ipynb   # Main notebook (this project)
├── README.md
└── requirements.txt
```

## Key Features

- **No black boxes** — every sampling primitive (temperature, Top-K, Top-P)
  is implemented from raw tensor operations, not library calls.
- **Unit-testable design** — each decoding function is validated against a
  small synthetic logits tensor before being applied to live GPT-2 output.
- **Reproducibility** — all relevant RNGs (Python, NumPy, PyTorch, CUDA) are
  seeded up front.
- **Device-agnostic** — automatically runs on GPU when available, CPU otherwise.
- **Numerically careful** — edge cases like near-zero temperature are
  handled explicitly to avoid overflow.

## Installation

```bash
git clone <your-repo-url>
cd <your-repo-directory>
pip install -r requirements.txt
```

Requires Python 3.9+. A CUDA-capable GPU is optional but recommended for
faster generation.

## Usage

Open `decoding_strategies.ipynb` in Jupyter or VS Code and run all cells top to
bottom, or run the core generation function directly in a Python session:

```python
from decoding_strategies import generate_text  # if extracted to a .py module

output = generate_text(
    prompt="Deep learning is fundamentally about",
    max_tokens=30,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
)
print(output)
```

Try varying `temperature`, `top_k`, and `top_p` to see how each parameter
independently affects output diversity and coherence.

## License

MIT
