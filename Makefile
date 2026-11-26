GO ?= go
PY ?= python3
.PHONY: all test slope
.DEFAULT_GOAL := all

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

# 谱斜率 -> 坡度分布/尾部 -> 可通行面积幂律 的完整复核。
# 生成三组 2048^2 原始 float64 地形，再跑 numpy/matplotlib 分析并产出 artifacts/slope_powerlaw_report.md。
slope:
	$(GO) build -o build/goproc ./cmd/goproc
	mkdir -p work artifacts/slope
	./build/goproc gen --size 2048 --seed 7 --octaves 6 --gain 0.50 --frequency 0.00390625 \
		--raw --format f64 --out work/G1_k6_g050.f64
	./build/goproc gen --size 2048 --seed 7 --octaves 6 --gain 0.65 --frequency 0.00390625 \
		--raw --format f64 --out work/G2_k6_g065.f64
	./build/goproc gen --size 2048 --seed 7 --octaves 8 --gain 0.50 --frequency 0.00390625 \
		--raw --format f64 --out work/G3_k8_g050.f64
	$(PY) analysis/slope_powerlaw.py
