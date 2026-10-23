# Data provenance and access

## Verified for this repository

The working input was recovered from the owner's local Jupyter files as `icu_train.csv`. Its dimensions, checksum, target counts, and run configuration are recorded in [the run report](real-data-results.md). Neither the original repository nor the recovered notebook identifies the exact upstream download, licensing terms, or preparation steps used to produce this file.

Raw CSVs, model artifacts, and row-level predictions are intentionally excluded from Git. For a fully downloadable example, use the synthetic demo. To reproduce the real-data run, obtain the same authorized input and compare its checksum with the report.

## Relevant upstream reference, not verified local provenance

The official [WiDS Datathon 2020 competition](https://www.kaggle.com/c/widsdatathon2020) describes predicting `hospital_death` from the first 24 hours of ICU data supplied through MIT's GOSSIS initiative. [PhysioNet's WiDS 2020 resource](https://physionet.org/content/widsdatathon2020/1.0.0/) provides an official access route and lists restricted health-data licensing and a data use agreement.

The local file's field names and task are consistent with this dataset family. That is a provenance hypothesis, not proof of the exact local dataset or its transformations. The local filename alone does not establish a license or redistribution permission. Follow the official provider's applicable access conditions; this repository does not mirror their records or automatically accept agreements.

To establish exact provenance, recover the original course handout, download link, data dictionary, and preparation/splitting script. Do not relabel a newly downloaded source file as identical unless its schema, rows, transformations, and checksum have been verified. The local hidden-test labels also need a verified correspondence to the test rows before evaluation.
