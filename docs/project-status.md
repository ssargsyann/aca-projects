# Project verification scope

All seven projects have been restructured with packages/commands, configuration, maintained notebooks, focused regression checks and documented results. This table summarizes work verified during the restructuring on 2026-10-23; it is not a shared benchmark or an assertion that every possible configuration was tested.

| Project | Verified evidence | Practical limit | Detail |
|---|---|---|---|
| ICU mortality | Recovered local-data training, saved-pipeline inference, explainability, ROC-AUC 0.8926 on validation | Exact original data provenance remains unresolved; validation is not clinical deployment evidence | [Run report](../projects/icu-mortality-prediction/docs/real-data-results.md) |
| Customer segmentation | Verified source cohort, preprocessing/PCA, K-Means/GMM, K=2 in-sample silhouette 0.3605, membership diagnostics | Descriptive clustering on one cohort; not a measured marketing intervention | [Results](../projects/customer-segmentation-marketing/docs/results.md) |
| CIFAR-10 CNN | Maintained train/checkpoint/predict/test workflow and a short real CPU execution smoke | Historical 81.56% notebook log is preserved; full training was not rerun and smoke accuracy is not that benchmark | [Results](../projects/cifar10-cnn-pytorch/docs/results.md) |
| CV Zoo | Four pinned pretrained models on one real image, tested IoU/grid/anchor routines and visual outputs | Qualitative single-image results; no YOLO training or dataset accuracy/mAP benchmark | [Results](../projects/cv-zoo-yolo-pytorch/docs/results.md) |
| U-Net / YOLOv8 | Fixed synthetic segmentation splits, test IoU 0.9022; actual COCO8 fine-tune and 377-frame video I/O/prediction | Narrow U-Net synthetic domain; COCO8 has only 4/4 images and possible pretraining overlap; video includes misclassifications | [Results](../projects/unet-yolov8-cv/docs/results.md) |
| Seq2Seq Transformer | Disjoint 4,000/400/400 dates, selected/reloaded model, 400/400 exact greedy test outputs with EOS | One synthetic English format/range; a rule parser also scores 400/400 | [Results](../projects/seq2seq-transformer/docs/results.md) |
| LLM decoding | Pinned real GPT-2, 20 continuations / 480 tokens, seeded reproduction, checked cache/reference behavior and full token traces | Two prompts/two seeds; descriptive sampling diagnostics, not quality or factuality evaluation | [Results](../projects/llm-decoding-strategies/docs/results.md) |

Individual GitHub workflows cover regression and small offline checks. Real pretrained/data runs are separately recorded in the linked reports and project validation notes; model weights and full datasets remain local. The original imported notebook outputs are identified as historical evidence. CUDA behavior, comprehensive hyperparameter sweeps and production deployment are not implied by this status.
