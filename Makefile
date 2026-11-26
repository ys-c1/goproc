GO ?= go
.PHONY: all test slope

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

# 生成边界延拓补偿前后的坡度图与两栏比对表，输出到 artifacts/slope。
slope:
	$(GO) run ./cmd/slopeprobe
