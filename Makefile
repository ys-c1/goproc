GO ?= go
.PHONY: all test hydro

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

hydro:
	$(GO) run ./cmd/hydro --size 128 --seed 7 --octaves 4 --bands 21 --out artifacts/hydro_run
