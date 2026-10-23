# Validation record

Checked on 2026-10-23 in Windows/Python 3.12 with CPU PyTorch:

- Nine network-free regression tests passed. They cover deterministic/disjoint calendar splits and exact export identity; leap days and strict domain/overflow checks; teacher-forcing shifts; causal leakage prevention; source/target padding invariance; zero attention to padded keys; dimension/all-padding guards; mixed-length greedy completion, output budgets and character/EOS row alignment; odd sinusoidal widths; and a tiny train/checkpoint/evaluate/predict round trip, including no rule-parser fallback.
- The full maintained 4,000/400/400 configuration completed ten CPU epochs. Validation selected epoch 10, then the reloaded checkpoint scored all 400 held-out test dates correctly with EOS. The deterministic rule baseline also scored 400/400.
- The saved-model prediction command, separate reloaded evaluation command and offline mechanics command were exercised. Exported hashes and per-test records make split identity/results auditable. Model binaries and generated CSVs stay local.
- Every cell in both maintained notebooks executed successfully in fresh kernels. This includes another full ten-epoch run, attention export, and reloaded evaluation checked against that run's test metrics. Published notebook sources have cleared outputs.
- Training, attention and mechanics figures were visually inspected. All local documentation links, exact CSV hashes, 400 per-test outputs, reloaded metrics and character/EOS attention labels were checked.

CI runs the regression suite and offline mechanics command without external data or pretrained downloads. CUDA speed/reproducibility and out-of-domain accuracy are unverified. See the model/use card for evaluation boundaries.
