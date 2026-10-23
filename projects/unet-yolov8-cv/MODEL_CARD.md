# Model and use card

## U-Net

Binary semantic segmentation: RGB inputs, one foreground-logit channel, four pooling/skip stages, transposed convolutions, batch normalization, and crop/pad alignment. The notebook's convolutional architecture is retained with configurable base width; the maintained smoke uses width 8 rather than original width 64. Training uses Adam and equal-weight BCE-with-logits + smoothed per-image soft Dice.

The generator is a deterministic Pillow implementation of noisy/gradient backgrounds and one colored circle, rectangle or triangle. It is a revised generator, not bit-identical to the original OpenCV implementation. SeedSequence([seed, split_id, index]) gives stable indexed samples with distinct train/validation/test namespaces. Samples may be clipped at image boundaries and colors can have low contrast. Validation/test do not regenerate on repeated visits. Similarity between samples remains possible; this is a controlled synthetic distribution, not an independent real-world dataset.

Select the epoch by highest validation mean foreground IoU (first epoch wins ties), then evaluate its saved checkpoint on test once. IoU and Dice are computed per image and averaged over images rather than averaging equally weighted batches. Empty prediction plus empty truth scores 1. Threshold is configured in advance and stored in the checkpoint; the report uses 0.5. A separate dataset or threshold-tuning experiment would be needed for new domains. The test set must not be repeatedly used to choose hyperparameters.

The model is suitable for teaching architecture and checking a segmentation pipeline on synthetic shapes. It has no evidence for medicine, satellite images, general scene segmentation, or other consequential applications. The full-width preset is supplied but was not retrained. CUDA execution and exact results across platforms/versions are not verified.

## YOLOv8n

The checksum-pinned official COCO80 pretrained base is fine-tuned for one epoch on four COCO8 images and validated on four others. COCO8 comes from COCO train2017; it can overlap the base model's pretraining corpus, so the split is not an independent generalization evaluation. The model is selected/validated within this tiny set, with no separate YOLO test set. A fresh reload of best.pt is used for image inference. Final stripped-checkpoint metrics can differ from epoch CSV metrics; both are recorded explicitly.

Traffic inference is qualitative and unlabeled, using the official base checkpoint, not the fine-tuned checkpoint. The first-48-frame smoke returned no boxes; full-video inference returned 165 frame-wise boxes over 377 frames, including repeated/cross-class conflicting hypotheses. Returned labels include `cell phone` as well as vehicle categories. No accuracy, reliable traffic monitoring, real-time throughput or production-readiness claim follows from execution. The pipeline does not track identities, count distinct objects, or combine U-Net masks with YOLO detections.

Ultralytics provides upstream AGPL/Enterprise licensing choices; dataset/image/video terms are separate. No blanket license for all assets is asserted. Use trusted serialized YOLO weights only. See the provenance record and upstream terms.
