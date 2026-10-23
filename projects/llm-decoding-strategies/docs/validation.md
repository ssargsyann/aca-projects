# Validation record

Checked on 2026-10-23 with Windows/Python 3.12 and CPU PyTorch:

- Eleven tests passed, including two optional framework integrations. Tests cover finite/masked logits, invalid inputs, tiny/zero temperatures, entropy, nonmutation, Top-K ties/limits, exact-CDF nucleus boundaries and batch behavior, scaling order, isolated seeded RNGs, EOS/context/zero-budget/cache shapes, missing local models and output preservation.
- Framework integration matched the installed temperature/Top-K/Top-P warpers on nonboundary random logits and compared cached/uncached generation using a tiny randomly initialized GPT-2 without downloads. Exact boundary semantics follow the explicit project contract, not an exhaustive equivalence assertion.
- The explicit pinned model fetch completed and verified all seven files. Real GPT-2 ran the predeclared twenty continuations and generated 480 tokens. The separate single-generation command and synthetic demo command also completed.
- Real seeded repetition matched the stored 24-token sample. An eight-token greedy case agreed between cached/uncached custom loops and the high-level Transformers reference, used only for validation. The real runs did not emit EOS; that branch is covered by fixtures.
- Every cell in both maintained notebooks executed successfully in fresh kernels with Hub/Transformers offline mode enabled. The real notebook repeated the complete twenty-run comparison and checked a seeded repeat plus cached/uncached greedy output. Published notebook sources have cleared outputs.
- The synthetic, first-token and continuation figures were visually inspected. Documentation links, complete trace lengths, experiment counts, recorded summaries and single-command output consistency were checked.

The runtime implementation uses direct forward passes, with no high-level generate call. CI enables Hub/Transformers offline mode and uses only synthetic/tiny-random-model tests plus the synthetic demo; it does not download the pretrained weights. Model binaries and full local artifacts stay ignored. CUDA and cross-backend exact reproducibility remain unverified.
