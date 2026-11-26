GO ?= go
PY ?= python3

.PHONY: all test slope-demo

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

# 坡度分级：生成 float64 高度场并做谱斜率预判/核对/出图出表
slope-demo:
	$(GO) run ./cmd/goproc gen --size 2048 --seed 7 --octaves 6 --frequency 0.01 \
		--format f64 --out work/terrain_n2048_s7_o6_f01.f64
	$(PY) analysis/slope_forecast.py \
		--input work/terrain_n2048_s7_o6_f01.f64 --n 2048 \
		--outdir artifacts --seed 7 --octaves 6 --frequency 0.01
