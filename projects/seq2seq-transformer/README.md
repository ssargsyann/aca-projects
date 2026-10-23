# Character-Level Date Transformer

An educational encoder-decoder Transformer assembled from explicit PyTorch attention, feed-forward, residual and normalization blocks. It learns English-date-to-ISO conversion, with reproducible disjoint datasets, teacher-forced and greedy evaluation, checkpoint restoration, attention exports, maintained notebooks and offline regression checks. No pretrained model, external dataset or tokenizer download is required.

[Verified results](docs/results.md) · [Model/use card](MODEL_CARD.md) · [Concept corrections](docs/concepts.md) · [Dataset provenance](docs/data.md) · [Automated checks](https://github.com/ssargsyann/aca-projects/actions/workflows/date-transformer-checks.yml)

## Verified result and scope

A fresh CPU run used 4,000 distinct training dates, 400 validation dates and 400 disjoint test dates. The 179,626-parameter model, trained for ten epochs, reached **400/400 exact greedy test predictions with EOS**. The deterministic rule parser also scored **400/400**; this task demonstrates the architecture, not an advantage over parsing or general machine-translation ability. All splits use the same canonical English format and 1950–2050 year range.

![Measured training and validation behavior](docs/figures/training.png)

The first epoch's validation teacher-forced token accuracy was 82.43%, but full greedy date accuracy was only 6%. Correct next-token predictions with ground-truth prefixes are not the same as correct self-generated dates. See the [full result record](docs/results.json) and [all 400 test outputs](docs/test-predictions.jsonl).

## Structure

```text
configs/                         Quick smoke and measured CPU training settings
src/date_transformer/
  data.py                        Unique calendar splits, fixed vocabulary, encoding
  model.py                       Explicit encoder/decoder blocks and boolean masks
  generation.py                  Batched greedy decoding, EOS and attention rows
  workflow.py                    Train/select/reload/evaluate/predict
  report.py                      Training curves and attention visualization
  mechanics.py                   Offline sinusoidal and causal-mask teaching demo
  cli.py                         Command-line entry points
tests/                           Data, masking, decoding and checkpoint contracts
notebooks/                       Two maintained notebooks and unchanged original
docs/                            Results, hashes, test predictions and figures
data/                            Generation instructions; generated CSVs ignored
artifacts/                       Local checkpoints/data/outputs; ignored
```

## Install

Python 3.10+; tested with Python 3.12 on Windows CPU. From the collection root:

```bash
cd projects/seq2seq-transformer
python -m venv .venv
```

Activate `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS/Linux.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[notebook]"
```

For commands without Jupyter, `python -m pip install -e .` suffices. No Hugging Face/Transformers, seaborn, or pretrained tokenizer is needed. For CUDA, select a compatible build from [PyTorch](https://pytorch.org/get-started/locally/); CUDA performance/reproducibility is unverified. `constraints-py312-cpu-tested.txt` lists observed direct versions, not a complete cross-platform lock.

## Dataset and training

```bash
date-transformer dataset --config configs/dates-cpu.json --output artifacts/exported-dates
date-transformer train --config configs/quickstart.json --output artifacts/quickstart
date-transformer train --config configs/dates-cpu.json --output artifacts/date-run
```

The quickstart is an intentionally tiny 128/24/24, two-epoch execution smoke and is not expected to reach the measured run's accuracy. `dates-cpu.json` reproduces the recorded 4,000/400/400, ten-epoch experiment. Choose new/empty output directories; commands preserve existing runs.

Every valid Gregorian day from 1950-01-01 through 2050-12-31 is eligible, including leap days and days 29–31. A single locally seeded shuffle and disjoint slices prevent duplicate dates across splits. The fixed vocabulary is derived from the declared format (English month names, digits and punctuation), not fitted to test observations. The exporter writes train/validation/test CSVs plus SHA-256/count metadata. Encoding rejects unsupported characters and overflow rather than silently truncating.

The model uses learned absolute positions, two encoder/two decoder blocks, width 64, four heads and feed-forward width 128. It follows the notebook's post-norm pattern with PyTorch `nn.MultiheadAttention`; Q/K/V projections and attention kernels are provided by PyTorch. It is not a custom low-level attention kernel or an exact reproduction of the original paper's full training recipe.

Train with teacher forcing, ignore padding in cross entropy, clip gradient norm to 1, and select the highest validation greedy exact-match score (lower validation NLL breaks ties). Only after selection is the saved checkpoint reloaded and test evaluated. Test is not used for model/threshold selection. The checkpoint contains tensor weights, configuration, vocabulary, selected epoch and dataset manifest; loading uses `weights_only=True` and strict state keys.

## Reloaded prediction and evaluation

```bash
date-transformer predict --checkpoint artifacts/date-run/best.pt --text "August 3, 2026" --attention --output artifacts/example
date-transformer evaluate --checkpoint artifacts/date-run/best.pt --split test --output artifacts/reloaded-test
date-transformer mechanics --output artifacts/mechanics
```

Prediction validates a canonical in-range date such as `August 3, 2026`, then runs the neural decoder; the parser result never replaces the prediction. The decoder caches encoder memory, generates greedily, and reports emitted tokens, EOS termination, and whether its text is a valid ISO date. It has a 12-token budget; malformed/special-token outputs are exposed instead of silently repaired. There is no decoder KV cache, beam search or constrained calendar decoder.

Training saves `best.pt`, exact split CSVs/manifests, `results.json`, per-test predictions, training/attention figures and machine-readable attention. Prediction saves provenance JSON plus the optional figure. Attention is captured from the final decoder layer, averaged across heads; each row labels the token generated at that step, including `<EOS>` if emitted. Source padding columns are omitted. Attention weights illustrate behavior but do not establish causal explanations.

![Cross-attention for the first held-out test date](docs/figures/attention.png)

## Notebooks and checks

```bash
jupyter lab
python -m unittest discover -s tests -v
```

- [quickstart.ipynb](notebooks/quickstart.ipynb) runs a tiny train/reload/predict/evaluate workflow without downloads and makes no accuracy claim.
- [verified-date-workflow.ipynb](notebooks/verified-date-workflow.ipynb) replays the measured ten-epoch configuration, inspects its greedy metrics, and plots correctly labeled attention.
- [original_experiment.ipynb](notebooks/original_experiment.ipynb) preserves the imported code/outputs unchanged. Use the maintained notebooks for reproducible execution.

Nine focused tests cover unique calendar splits, strict encoding, target shifts, causal leakage prevention, padding invariance, attention masking, EOS/budget/mixed-batch behavior, odd sinusoidal widths and a train/checkpoint/evaluate/predict round trip. CI runs offline. [Validation details](docs/validation.md).

The historical tokenizer demo used BERT WordPiece rather than BPE; sinusoidal similarity is oscillatory rather than monotonically decreasing; multi-head attention includes learned projections as well as a reshape; the RNN-loop versus QK-only timing is not a comparable Transformer benchmark. These corrections are documented in [concept notes](docs/concepts.md), with primary sources. The model's actual positional embeddings are learned, separately from the sinusoidal teaching demo.

**Author:** Sargis Sargsyan.
