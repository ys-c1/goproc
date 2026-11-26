GO ?= go
.PHONY: all test analysis

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

analysis:
	python3 analysis/remap_entropy.py
