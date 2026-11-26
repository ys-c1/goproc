GO ?= go
PY ?= python3
BUDGET_DIR := artifacts/terrain_budget
.PHONY: all test budget

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

budget:
	mkdir -p $(BUDGET_DIR)
	$(GO) run ./cmd/goproc gen --size 129 --seed 7 --octaves 5 --out $(BUDGET_DIR)/terrain.pgm
	$(PY) tools/terrain_budget.py --pgm $(BUDGET_DIR)/terrain.pgm --outdir $(BUDGET_DIR) --eps 0.04,0.02
