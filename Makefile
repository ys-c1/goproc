GO ?= go
.PHONY: all test gully

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

gully:
	$(GO) run ./cmd/gully
