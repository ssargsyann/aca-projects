# Model and use card

## Task and architecture

Canonical English month-name dates to ISO text, e.g. `August 3, 2026` → `2026-08-03`. Fixed 42-token character/special vocabulary; PAD/BOS/EOS indices 0/1/2. Source length 20, teacher-forced target length 12, generation budget 12 including any EOS step. Unsupported/oversized text is rejected, never silently truncated.

An encoder-decoder assembled from PyTorch modules: learned token/absolute-position embeddings, two post-norm encoder layers and two decoder layers, width 64, four heads, ReLU feed-forward width 128, dropout 0.1 within attention and the feed-forward network. There is no high-level `nn.Transformer`, but `nn.MultiheadAttention` supplies learned projections/kernels. This educational implementation is not every detail of the paper's architecture/training recipe. The measured model has 179,626 trainable parameters. No pretrained weights are used.

## Data and evaluation

All valid dates from 1950 through 2050 are enumerated, shuffled with a local seeded RNG, then sliced into unique disjoint 4,000/400/400 subsets. English month labels are explicitly defined, so generation does not depend on the operating-system locale. The vocabulary is part of the task specification rather than learned from held-out strings. Exact CSV hashes identify the measured split membership.

Teacher-forced NLL/token accuracy include the ten ISO characters plus EOS, excluding padding. Greedy exact-match requires all characters to equal the target and EOS to be emitted. ISO validity checks calendar validity of output text; it is not equality to the requested source date. Validation selects the highest greedy exact match, breaking ties by lower teacher-forced NLL. Test is evaluated only after checkpoint selection and reloading. Re-evaluation reproduces the saved split rather than sampling fresh data.

The measured test result is 400/400 on this one seed, same date range and same format as training. It does not establish performance on every possible calendar date, out-of-range years, unseen formats, other languages, real translation, or downstream date extraction. No throughput benchmark or CUDA validation was performed. Published weights are not bundled; commands generate/retrain everything locally.

## Baseline and intended use

A deterministic calendar parser scores 400/400 on the same task and is the practical choice for this specified format. The neural model is for learning masked self-attention, cross-attention, teacher forcing, autoregressive decoding and evaluation discipline. It provides no demonstrated product advantage over parsing. Input parsing in the neural prediction command validates the supported domain only; its ISO answer is never substituted for generated text, including when the model is wrong.

The last decoder layer's cross-attention is averaged over heads and captured at each generated step. Figure rows include EOS and columns omit source padding; the resulting weights are descriptive diagnostics, not causal explanations. Greedy decoding has an output budget and can fail/emit malformed text; results expose termination and validity without repair. Checkpoints use restricted tensor-state loading and must still come from trusted sources.
