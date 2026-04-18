# make sure this is executed with bash
SHELL := /bin/bash

PYTHON ?= python3
RUN_PY := PYTHONPATH=. .venv/bin/python3


YELLOW := "\e[1;33m"
NC := "\e[0m"

# Logger function
INFO := @bash -c '\
  printf $(YELLOW); \
  echo "=> $$1"; \
  printf $(NC)' SOME_VALUE

.venv:  # creates .venv folder if does not exist
	$(PYTHON) -m venv .venv


.venv/bin/uv: .venv # installs latest pip
	.venv/bin/pip install -U uv

install: .venv/bin/uv
	# before running install cmake
	.venv/bin/python3 -m uv pip install -r requirements.txt
	# after installing source .venv/bin/activate in your shell

download_data_from_s3:
	.venv/bin/python3 -m download_data

prepare_model_inputs_v1:
	$(RUN_PY) cini/scripts/prepare_model_inputs_v1.py --data-root data/makeathon-challenge --split-dir cini/splits/split_v1 --label-root artifacts/labels_v1 --output-root artifacts/model_inputs_v1 --force --max-workers 32

train_cini:
	$(RUN_PY) models/cini/train.py

predict_cini:
	$(RUN_PY) models/cini/predict.py

eval_cini:
	$(RUN_PY) models/cini/evaluate.py

train_kaite:
	$(RUN_PY) models/kaite/train.py

predict_kaite:
	$(RUN_PY) models/kaite/predict.py

eval_kaite:
	$(RUN_PY) models/kaite/evaluate.py

train_kangi:
	$(RUN_PY) models/kangi/train.py

predict_kangi:
	$(RUN_PY) models/kangi/predict.py

eval_kangi:
	$(RUN_PY) models/kangi/evaluate.py

train_tomy:
	$(RUN_PY) models/tomy/train.py

predict_tomy:
	$(RUN_PY) models/tomy/predict.py

eval_tomy:
	$(RUN_PY) models/tomy/evaluate.py
