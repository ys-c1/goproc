GO ?= go
.PHONY: all test verify-erosion

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

verify-erosion:
	$(GO) run ./cmd/verifyerosion --out artifacts/erosion
