# Data pipeline for claim simulation.
# Run `make` to build the full pipeline end-to-end.
# Intermediate outputs are cached — only stale steps are re-run.

PYTHON = python

# ── Final target ──────────────────────────────────────────────────────────────

.PHONY: all
all: data/portfolio.csv data/portfolio_v2.csv

# ── Pipeline steps ────────────────────────────────────────────────────────────

data/freMTPL2freq.csv:
	$(PYTHON) python/data/download.py

models/frequency_model.lgb models/feature_metadata.json: data/freMTPL2freq.csv
	$(PYTHON) python/train.py v1

models/frequency_model.onnx: models/frequency_model.lgb models/feature_metadata.json
	$(PYTHON) python/export_onnx.py v1

data/freMTPL2freq_with_history.csv: models/frequency_model.onnx
	$(PYTHON) python/generate_history.py

models/frequency_model_v2.lgb models/feature_metadata_v2.json: data/freMTPL2freq_with_history.csv
	$(PYTHON) python/train.py v2

models/frequency_model_v2.onnx: models/frequency_model_v2.lgb models/feature_metadata_v2.json
	$(PYTHON) python/export_onnx.py v2

data/portfolio.csv data/portfolio_v2.csv: models/frequency_model.onnx models/frequency_model_v2.onnx
	$(PYTHON) python/export_portfolio.py

# ── Housekeeping ──────────────────────────────────────────────────────────────

.PHONY: clean
clean:
	rm -f models/frequency_model.lgb models/feature_metadata.json \
	      models/frequency_model.onnx \
	      data/freMTPL2freq_with_history.csv \
	      models/frequency_model_v2.lgb models/feature_metadata_v2.json \
	      models/frequency_model_v2.onnx \
	      data/portfolio.csv data/portfolio_v2.csv
