# Notebooks

Notebooks are for exploration and presentation. Final preprocessing, training, and evaluation logic must be reproducible outside notebooks.

Stage 0 correctness visualizations may read generated manifests and transformed
samples, but notebook code must not define split, sampling, transform, or cache
semantics. Those live in tested modules and versioned configurations.
