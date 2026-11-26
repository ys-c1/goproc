GO ?= go
.PHONY: all test fog

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

# 两档雾浓度（beta=5、10 km^-1）实验，产物在 artifacts/fog/
fog:
	mkdir -p .tmp
	GOTMPDIR=$(CURDIR)/.tmp $(GO) run ./cmd/goproc fog
