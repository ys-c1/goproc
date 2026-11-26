GO ?= go
.PHONY: all test blockaudit

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

blockaudit:
	$(GO) run ./cmd/blockaudit
	python3 -I analysis/block_audit.py
