# Common tasks, listed by `make help`. Only an answer (`run`, `docker-run`) needs an API key,
# and `ingest` needs the network.

# The interpreter used to build .venv: the newest supported Python on PATH. To use a
# specific one: make setup PYTHON=/path/to/python3.12
PYTHON ?= $(shell for v in 3.14 3.13 3.12; do command -v python$$v >/dev/null 2>&1 && { echo python$$v; exit 0; }; done; echo python3)
PY := .venv/bin/python
BR := .venv/bin/bankresearch

.PHONY: help setup lock hooks key-set key-run run ingest check secrets lint types contracts test eval-dev eval-test demo fixture clean docker-build docker-ingest docker-run docker-eval docker-smoke

help:
	@echo "setup            create .venv and install the pinned dependencies"
	@echo "check            secret scan, lint, types, import contracts and tests"
	@echo "demo             read the synthetic fixture and show the figures it checked, no key"
	@echo "ingest           download and ingest the 21 real filings (network, about two minutes)"
	@echo "run              one request, asked first: ARGS='--bank NAB --period 1H26 --metric nim'"
	@echo "eval-dev         run the dev split"
	@echo "eval-test        run the locked test split once (--final)"
	@echo "docker-build     build the container image"
	@echo "docker-ingest    download and ingest the filings into the container's volume"
	@echo "docker-run       one request in the container: ARGS='--bank ...'"
	@echo "docker-eval      run the dev split in the container"
	@echo "docker-smoke     build the image and check that it answers"
	@echo "key-set          store the API key: the macOS keychain, or .env elsewhere (never echoed)"
	@echo "key-run          run one command with the key from the keychain: CMD='...'"
	@echo "lock             re-pin dependencies to the newest versions pyproject allows"
	@echo "hooks            install the pre-commit hooks"

# Checks the Python version first, because the common failure is a Mac where python3
# is the system 3.9 and pip then fails with an error that doesn't mention why.
# --clear rebuilds .venv from scratch, which also fixes one left broken by a Python
# upgrade.
setup:
	@$(PYTHON) -c 'import sys; sys.exit(not (3, 12) <= sys.version_info[:2] <= (3, 14))' 2>/dev/null \
		|| { echo "Need Python 3.12, 3.13 or 3.14, but '$(PYTHON)' is "$$($(PYTHON) --version 2>/dev/null || echo 'not installed')"."; \
		     echo "Install one (on a Mac: brew install python@3.13), or point at it: make setup PYTHON=/path/to/python3.13"; \
		     exit 1; }
	$(PYTHON) -m venv --clear .venv
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -c constraints.txt -e ".[dev,pdf,llm]"
	@echo "Ready. Try: make check, then make demo"

# Re-resolves every dependency to the newest version pyproject.toml allows, in a
# throwaway venv, and rewrites constraints.txt. Run make setup and make check after.
lock:
	rm -rf .lock-venv
	$(PYTHON) -m venv .lock-venv
	.lock-venv/bin/python -m pip install --quiet --upgrade pip
	.lock-venv/bin/python -m pip install --quiet ".[dev,pdf,llm]"
	{ sed -n '/^#/p' constraints.txt; .lock-venv/bin/python -m pip freeze --exclude-editable | grep -v '^bankresearch'; } > constraints.new
	mv constraints.new constraints.txt
	rm -rf .lock-venv
	@echo "constraints.txt updated. Now: make setup && make check"

# Anything that needs .venv depends on this, so running it before setup says what to
# do rather than failing on a missing file.
$(PY):
	@echo "No .venv here yet. Run: make setup"; exit 1

ingest: | $(PY)
	$(BR) ingest

hooks: | $(PY)
	$(PY) -m pip install --quiet pre-commit
	$(PY) -m pre_commit install

# On a Mac the key goes in the login keychain, which a backup, a screen share or a stray grep
# won't find. Elsewhere it goes in .env, which is gitignored and readable only by you.
KEYCHAIN_SERVICE := bankresearch-anthropic

key-set:
	@if command -v security >/dev/null 2>&1; then \
		echo "Paste the key, then press Return. Nothing is echoed or kept in history."; \
		security add-generic-password -U -a "$$USER" -s $(KEYCHAIN_SERVICE) -w \
			&& echo "Stored in the macOS keychain. Every run reads it from there."; \
	else \
		printf 'Paste the key, then press Return. Nothing is shown: '; \
		stty -echo 2>/dev/null; read -r key; stty echo 2>/dev/null; echo; \
		test -n "$$key" || { echo "No key entered, so nothing was saved."; exit 1; }; \
		umask 077; { grep -v '^ANTHROPIC_API_KEY=' .env 2>/dev/null; echo "ANTHROPIC_API_KEY=$$key"; } > .env.new \
			&& mv .env.new .env && echo "Stored in .env, which is gitignored and readable only by you."; \
	fi

# make can't pass a quoted command through CMD cleanly, so prefer calling
# scripts/with-key directly. This is here for convenience.
key-run:
	@test -n "$(CMD)" || (echo "Usage: make key-run CMD='bankresearch run --bank CBA ...'" \
		&& echo "   or: scripts/with-key bankresearch run --bank CBA ..." && exit 1)
	@scripts/with-key $(CMD)

check: secrets lint types contracts test

# Runs first: a leaked key is the one mistake you can't fix by changing code.
secrets: | $(PY)
	$(PY) scripts/scan_secrets.py

lint: | $(PY)
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

types: | $(PY)
	$(PY) -m mypy

contracts: | $(PY)
	.venv/bin/lint-imports

test: | $(PY)
	$(PY) -m pytest tests/ -q

# --- container ------------------------------------------------------------------
# One named volume holds the filings, the database and every run's output.
IMAGE  := bankresearch
VOLUME := bankresearch-data
DOCKER_RUN := docker run --rm -v $(VOLUME):/app/data
# Attach a terminal only when there is one, so the targets also work from scripts and CI.
TTY := $(shell [ -t 0 ] && echo -it)

docker-build:
	docker build -t $(IMAGE) .

# Downloads the filings into the volume the first time; later runs reuse them.
docker-ingest:
	$(DOCKER_RUN) $(IMAGE) ingest

# The key is passed into the container for this run only, from ANTHROPIC_API_KEY, .env or
# the macOS keychain, through the environment rather than the command line. A terminal is
# attached so the estimate can be agreed before anything is spent.
docker-run:
	@test -n "$(ARGS)" || (echo "Usage: make docker-run ARGS='--bank ANZ --period 1H26 --metric nim --as-of 2026-05-31'" && exit 1)
	@key="$${ANTHROPIC_API_KEY:-$$(sed -n 's/^ANTHROPIC_API_KEY=//p' .env 2>/dev/null | tail -n 1)}"; \
	if [ -z "$$key" ] && command -v security >/dev/null 2>&1; then \
		key="$$(security find-generic-password -a "$$USER" -s $(KEYCHAIN_SERVICE) -w 2>/dev/null)"; \
	fi; \
	ANTHROPIC_API_KEY="$$key" $(DOCKER_RUN) $(TTY) -e ANTHROPIC_API_KEY -e BANKRESEARCH_LLM_CONSENT=ask $(IMAGE) run $(ARGS) || [ $$? -eq 2 ]

docker-eval:
	$(DOCKER_RUN) $(IMAGE) eval --split dev

# Builds the image and checks it answers, without touching the data volume.
docker-smoke: docker-build
	docker run --rm $(IMAGE) config-check
	docker run --rm $(IMAGE) resolve --bank ANZ --period 1H26 --metric nim --as-of 2026-05-31

# One request. It shows an estimate of what it needs and costs, and asks before anything
# is spent. The key comes from ANTHROPIC_API_KEY or the macOS keychain (make key-set).
# A refusal to answer exits 2. That's a valid answer, so make doesn't treat it as a
# failure; real errors still exit 1. Scripts that need the exit code should call
# bankresearch directly.
run: | $(PY)
	@test -n "$(ARGS)" || (echo "Usage: make run ARGS='--bank ANZ --period 1H26 --metric nim --as-of 2026-05-31'" && exit 1)
	@$(BR) run $(ARGS) || [ $$? -eq 2 ]

# The dev split with no model: free, and the same numbers every time. Add --replay to answer
# the model steps from responses a live run recorded, or --grant to run them live.
eval-dev: | $(PY)
	$(BR) eval --split dev

eval-test: | $(PY)
	$(BR) eval --split test --final

# Reads the synthetic fixture as the made-up issuer XMPL (DESIGN.md ADR-012) and shows the
# figures it stored, each checked against its accounting identities: a check that the
# install works, with no key and nothing spent. An answer is written by a model, so it
# needs a key.
demo: | $(PY)
	$(BR) ingest --path tests/fixtures/synthetic_results_document.pdf \
		--bank XMPL --period 1H26 --publish-date 2026-05-01
	$(BR) facts --bank XMPL --period 1H26
	@echo "The install works. Answers are written by a model, so add your API key next: make key-set,"
	@echo "or a line ANTHROPIC_API_KEY=... in a file named .env here. Then try:"
	@echo "make run ARGS='--bank NAB --period 1H26 --metric nim'"

fixture: | $(PY)
	$(PY) tests/fixtures/make_fixture_pdf.py

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache runs/* data/processed data/index
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
