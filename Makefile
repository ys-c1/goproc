GO ?= go
.PHONY: all test flowdemo

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

flowdemo:
	$(GO) run ./cmd/flowdemo --out artifacts/flow
