# Third-party provenance

Original code and generated synthetic data/results: MIT, see `LICENSE`.

The frozen model is [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca), pinned revision `c1899de289a04d12100db370d81485cdf75e47ca`. Its official [LICENSE](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/LICENSE) is Apache-2.0. Model and tokenizer files are neither vendored nor committed. A reproduction downloads them from the original publisher and must retain their own license notices. The tokenizer hash and architecture config are recorded in `results/metrics/backbone.json`.

[MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-lm) are used as installed dependencies under MIT terms. Transformers and Hugging Face Hub use Apache-2.0; NumPy and SciPy use BSD terms; Matplotlib uses its project license; PyYAML uses MIT; Python-Markdown uses BSD terms; pytest uses MIT. This repository does not vendor these packages. Exact installed dependency versions are recorded in `requirements-lock.txt`.

The published papers are linked and paraphrased, not redistributed. No private training project source, proprietary data, or custom local checkpoint was imported. Only a shallow checkpoint-file discovery was performed; no readily accessible compatible custom checkpoint was found before selecting the existing public cached Qwen model. No remote workstation was accessed.
