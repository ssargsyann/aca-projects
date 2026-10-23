# Model and use card

## Source model and implementation

The project does not train a language model. It uses the pinned pretrained English GPT-2 checkpoint, 124,439,808 parameters, float32, 50,257 vocabulary entries and 1,024 context positions. The tokenizer uses byte-level BPE. Config/tokenizer/card files and safetensors are checked against fixed upstream identities before local-only loading; remote code is disabled. Upstream model provenance/license metadata is separate from this project's source-code licensing.

The custom portion is the raw-logit policy and forward-pass loop. PyTorch/Transformers provide the pretrained architecture and tokenizer. There is no high-level generate call in the implementation; one external real-model validation used that API as a greedy reference. Policies are greedy, temperature-only, Top-K, Top-P and combined filtering, with explicit ordering/tie/numerical contracts. Decoding is single-prompt and optionally caches past key/value tensors. It is not a serving system or inference optimization benchmark.

## Evidence

Two predeclared prompts, seeds 7/42, five policies and 24 new tokens per continuation produced twenty records and 480 tokens. All measured records ended at their token budgets, not EOS; EOS stopping is covered separately by deterministic fixtures. Same-seed reproduction and cached/uncached/reference greedy equality were checked on particular real GPT-2 examples, not exhaustively over all prompts/backends. Tests also use a tiny randomly initialized GPT-2, which is an API/cache integration test rather than language-quality evidence.

The four records per policy include two prompts and two seeds. Greedy ignores the seeds, so its two per-prompt outputs are duplicates, not independent samples. Different sampling policies can also produce identical continuations. No human rating, held-out perplexity, factuality assessment or statistical significance analysis was conducted. Token entropy/distinctness/base likelihood are descriptive diagnostics, not quality scores.

## Limitations and use

This is an educational decoding experiment. GPT-2 may produce unsupported claims, repetition, biased or offensive text. No factual or instruction-following reliability is implied; the observed France samples are continuations, not verified factual answers. It is not adapted or validated for consequential advice, production assistants or moderation-sensitive use.

Sampling entropy is measured after policy transformations. Greedy entropy zero is a property of argmax, not model confidence. `distinct_1/2` are length-dependent token ratios and can be high for poor text. Repetition can remain under every policy. Context overflow is rejected without truncation; outputs can end mid-sentence at the token budget. CUDA behavior and cross-version/hardware bitwise reproducibility are unverified. Use the model and sample files only within the terms in the upstream provenance record.
