# Model and experiment provenance

Source: [openai-community/gpt2](https://huggingface.co/openai-community/gpt2), revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. The upstream README/model card declares MIT for the model repository. The downloaded pinned card is retained locally with the assets. This is not a blanket license assertion for all source material or an added license for the collection.

`fetch-model` explicitly downloads only config.json, merges.txt, tokenizer.json, tokenizer_config.json, vocab.json, README.md and model.safetensors. The 548,105,171-byte weight file is checked against upstream SHA-256 `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`; other files are checked against their Git blob identities from the pinned Hub revision. Every local file also receives a full SHA-256 in [the committed manifest](model-manifest.json).

Inference verifies those identities again and loads only existing local files with safetensors, `local_files_only=True` and `trust_remote_code=False`. No API key is needed for the public model, and the explicit downloader uses anonymous access. Model binaries and Hub download metadata remain ignored; no model is silently retrieved during generation or comparison.

The predeclared two prompts, five policies, seeds 7/42 and token budgets are in [the exact experiment configuration](experiment.json). There is no external evaluation dataset or fine-tuning. [Full continuations and traces](continuations.jsonl), [first-token distributions](first-token.json) and [runtime/version summary](summary.json) record what was measured. The generated claims are not fact-checked ground truth. The synthetic ten-logit fixture is a teaching input, clearly separated from real model runs.
