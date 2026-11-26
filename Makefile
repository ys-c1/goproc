GO ?= go
PY ?= python3
.PHONY: all test dep

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

# 洼地阈值预判实验：生成规范地形 -> Python 分析/标注图 -> Go 独立复核
dep:
	$(GO) run ./cmd/goproc gen --size 128 --seed 7 --octaves 4 --out artifacts/terrain_128_s7.pgm
	$(PY) -I analysis/depressions.py --pgm artifacts/terrain_128_s7.pgm --outdir artifacts --t1 20 --t2 40
	$(GO) run ./cmd/depverify --pgm artifacts/terrain_128_s7.pgm --t1 20 --t2 40 --out artifacts/depression_go_verify.json
