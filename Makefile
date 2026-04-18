# make sure this is executed with bash
SHELL := /bin/bash

PYTHON ?= python3


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
	PYTHONPATH=. .venv/bin/python3 cini/scripts/prepare_model_inputs_v1.py --data-root data/makeathon-challenge --split-dir cini/splits/split_v1 --label-root artifacts/labels_v1 --output-root artifacts/model_inputs_v1 --force --max-workers 32
