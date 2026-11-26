GO ?= go
.PHONY: all test tile

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

tile: test
	bash artifacts/verify.sh
