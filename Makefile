GO ?= go
.PHONY: all test lod

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

# 生成 LOD 二/三/四级缝宽、最大高程误差与分别存档的拼接图。
lod:
	$(GO) run ./cmd/goproc lodcheck --out artifacts/lod
