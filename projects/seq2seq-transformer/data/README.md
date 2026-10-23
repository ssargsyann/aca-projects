# Generated data

This project requires no downloaded dataset. Run `date-transformer dataset --config configs/dates-cpu.json --output artifacts/exported-dates` to generate train/validation/test CSVs and their manifest, or train directly and use the run's `data/` folder. Each valid in-range Gregorian date can occur in at most one split. All splits share the same English format/year range; distinct membership is not an out-of-domain evaluation. See [dataset provenance](../docs/data.md) and the committed hash manifest.
