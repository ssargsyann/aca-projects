# Local assets

Run `unet-yolo fetch-assets --output assets/downloaded` from the project directory to explicitly retrieve the pinned official base weights, COCO8 archive/data and traffic sample. It writes a source/hash manifest. No inference/training command downloads missing files implicitly. Alternatively use existing local files and supply their paths. See [provenance](../docs/provenance.md) for source attribution and terms. Large files remain ignored.
