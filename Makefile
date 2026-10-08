PYTHON ?= .venv/bin/python
CONFIG ?= configs/pilot.yaml
.PHONY: test mlx-test data run smoke middle report
test:
	$(PYTHON) -m pytest -q
mlx-test:
	LATENT_LOOP_MLX_TESTS=1 $(PYTHON) -m pytest -q tests/test_mlx.py
data:
	$(PYTHON) -m latent_loop.cli data --config $(CONFIG)
run:
	$(PYTHON) -m latent_loop.cli run --config $(CONFIG)
smoke:
	$(PYTHON) -m latent_loop.cli run --config configs/smoke.yaml --run-dir artifacts/smoke
middle:
	$(PYTHON) -m latent_loop.cli middle --config $(CONFIG)
report:
	$(PYTHON) -m latent_loop.cli report --config $(CONFIG)
