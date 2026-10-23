# Image and weight provenance

The original notebook uses [Darknet's `data/dog.jpg`](https://github.com/pjreddie/darknet/blob/master/data/dog.jpg). This restructuring reused the same-named local image from the owner's working files, with dimensions 768x576. The explicit retrieval helper then downloaded the public source to a fresh local destination and verified its SHA-256 exactly matched `5a9522051c3cec2bbd2f6323fccba32e8fbf3ddcc2b3e2fd46b04c720bc6f866`. The three boxes in `configs/sample_geometry.json` are hand-entered teaching annotations from the notebook, not a benchmark-quality evaluation dataset.

Place the intended RGB image at `assets/dog.jpg`, or pass another local path. For the sample geometry configuration the dimensions must match exactly; changing/resizing an image requires corresponding annotations. Images are excluded from Git by convention. The annotated figures in the report credit this upstream sample source; no separate photo license is invented or inferred from a software license.

## Model weights

Four explicit official Torchvision enum identities are used instead of drifting `DEFAULT` aliases. See `docs/results.json` for URLs, filenames, and full verified SHA-256 values. Their existing local files total about 603 MiB; the previous README's 100–200 MB estimate for all models was too low. No weights are committed.

```bash
cv-zoo weights --cache-dir artifacts/weights
```

This explicitly downloads missing files from `download.pytorch.org` and checks the expected official SHA-256 filename prefix. Inference never downloads automatically; you can point `--weights-dir` at an existing Torch checkpoint directory. All selected files are verified before inference begins. Each model loads a state dictionary with `weights_only=True` and strict key matching; only use official, trusted sources. Provider model/data terms remain applicable.
