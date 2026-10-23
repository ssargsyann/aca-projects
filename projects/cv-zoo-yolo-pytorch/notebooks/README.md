# Notebooks

- `quickstart.ipynb`: maintained synthetic rectangle/grid/anchor demo, no pretrained downloads.
- `real-inference.ipynb`: maintained four-model inference, montage, and separate sample geometry.
- `original_experiment.ipynb`: unchanged historical notebook.

Start Jupyter from the project or its `notebooks/` directory and choose the installed package environment. The real notebook expects the documented local image and cached weights; `CV_ZOO_IMAGE` and `CV_ZOO_WEIGHTS_DIR` can supply existing locations. Generated run directories are unique. Maintained notebook outputs are cleared before publishing.

The original notebook's shared-class YOLOv1 tensor formula, anchor terminology, center-matching description, and broad two-stage-model grouping are corrected in `docs/geometry-notes.md`; the historical notebook itself is preserved for traceability. Neither notebook is a YOLO training implementation or a benchmark suite.
