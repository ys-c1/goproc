GO ?= go
.PHONY: all test tilt

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

# 生成两组参数的平均倾角比对与分布图存档
tilt:
	$(GO) run ./cmd/goproc tilt --size 512 --report artifacts/tilt/report.md
