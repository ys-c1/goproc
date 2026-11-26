GO ?= go
.PHONY: all test

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...
