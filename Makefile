GO ?= go
.PHONY: all test normalreport

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

normalreport:
	$(GO) run ./cmd/goproc normalreport --out artifacts/normal_quant
