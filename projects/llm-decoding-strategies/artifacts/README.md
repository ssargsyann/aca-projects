# Local experiment outputs

Choose a new/empty directory per command. The synthetic demo writes probability JSON and a chart. A single GPT-2 run writes generation/trace JSON and the source/runtime manifest. A comparison writes exact settings, twenty continuation records with per-step diagnostics as JSONL, first-step distributions, figures and summary JSON. Failed/partial outputs remain available for inspection and are never silently overwritten.

Model files and local outputs remain ignored. Committed `docs/` copies contain small measured records, complete token traces and figures; no weight file is committed. Samples are unverified generated text, not factual reference material.
