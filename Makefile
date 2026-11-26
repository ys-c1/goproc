GO ?= go
.PHONY: all test zero

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

zero:
	$(GO) run ./cmd/zeroreport --out artifacts
	python3 analysis/zero_charts.py artifacts/zero_profile_data.json artifacts
