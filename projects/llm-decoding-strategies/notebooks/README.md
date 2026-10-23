# Maintained notebooks

- `sampling-primitives.ipynb`: offline synthetic distributions, temperature/Top-K/Top-P contracts and boundary examples. No GPT-2 dependency or download is needed.
- `local-gpt2-comparison.ipynb`: full twenty-run real comparison, plots and seeded/cache checks using local pinned assets. Install the GPT-2 extra and fetch assets explicitly first. `LLM_DECODING_MODEL` can point at an existing verified local directory.
- `original_experiment.ipynb`: unchanged imported notebook. Its implicit downloads, filter ordering, near-zero shortcut and missing EOS stop remain historical behavior; the maintained implementation documents its changes.

Launch Jupyter from the project or notebooks directory and select the installed environment. Each maintained execution chooses a unique output directory. Published sources have cleared outputs; measured records/figures are stored separately. This is an educational continuation experiment, not an instruction-following or factual-answer system.
