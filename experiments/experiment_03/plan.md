# Experiment 3: Learning When to Think

Freeze the existing backbone, three independently fitted no-memory refiners, prompt prefix, and original datasets. Verify historical prediction metrics and recover trajectories from checkpoints. Keep a SHA256 inventory of all historical tracked files.

Before fitting controllers, quantify immediate next-loop loss reductions and decision transitions on the historical benchmark and freshly generated controller train/validation data. Save this opportunity analysis with a timestamp and require it before fitting. Final fresh evaluations remain unused for model and policy selection.

Generate independent controller training/validation examples and three fresh evaluation splits: original templates, rewritten templates with identical semantics/distributions, and original templates at the existing OOD difficulty. Validate independent reference answers and cross-split instance/prompt uniqueness, including all historical examples. This pilot does not claim alpha-renamed structural disjointness or unseen algorithm families.

Fit width-64 current correctness, immediate gain classification, signed gain regression, and joint three-output heads for each parent seed. Adapt the historical candidate-confidence architecture as an additional baseline. Use only current state, conditional option probabilities, entropy, margin, depth, and already-completed state change. Terminal depth has no gain target. Train on fresh trajectories outside the refiner training set. Fit scalar logistic/ridge baselines on controller training data; select checkpoints and thresholds on controller validation only.

Compare fixed depths, confidence stopping, immediate classification stopping, expected-gain stopping, and a joint rule. Select thresholds by validation accuracy, then mean loops, then NLL; select the VOC cost by validation NLL plus cost-weighted loops using a fixed reference utility cost of 0.01 nat/loop. Report all cost sweep points. Include the old continuation policy and oracle ceilings as labelled diagnostics. An oracle stops at its chosen depth and pays every intervening loop.

Report paired seed/item bootstrap intervals, matched-compute mixtures, per-depth/task predictors, scalar controls with and without gain information, behavior categories, and representative transitions. Measure full synchronized batch-one latency including tokenization, backbone, decoding and live controller decisions, plus auxiliary-only timing, on a prespecified balanced validation subset. Generate figures from recorded predictions/timing samples. Preserve ignored caches/checkpoints, public metrics, reproducible commands, and local commits.

Limit the new numerical run to four hours and 48 GiB; no new backbone/refiner training or architecture sweep. Optional earlier-layer probes and eight-loop stress tests are deferred because they are outside the primary question.
