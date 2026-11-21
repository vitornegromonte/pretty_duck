PY ?= .venv/bin/python
export PYTHONPATH := src
REMOTE ?= maidai
REMOTE_DIR ?= ~/pretty_duck_gate
SYNC = rsync -a --exclude __pycache__ src scripts $(REMOTE):$(REMOTE_DIR)/

.PHONY: test lint paper sync-remote-dry sync-remote

test:  ## run the offline CPU tests
	$(PY) -m pytest -q

lint:  ## check style (needs ruff in the environment)
	ruff check src scripts tests

paper:  ## build docs/paper/method.pdf (two passes for references)
	cd docs/paper && pdflatex -interaction=nonstopmode method.tex >/dev/null && pdflatex -interaction=nonstopmode method.tex

sync-remote-dry:  ## show what would be copied to the GPU machine. Never touches outputs/
	rsync -an --itemize-changes --exclude __pycache__ src scripts $(REMOTE):$(REMOTE_DIR)/

sync-remote:  ## copy code and data to the GPU machine. Check that no run uses the files first
	$(SYNC)
