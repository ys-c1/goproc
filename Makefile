GO ?= go
.PHONY: all test axis

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

axis:
	$(GO) run ./cmd/axisdemo --out artifacts/symmetry --cases 12
	python3 analysis/make_axis_charts.py
