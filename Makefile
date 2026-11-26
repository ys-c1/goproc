GO ?= go
PYTHON ?= python3
.PHONY: all test seedexp terrainpairs figures

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

# 种子扩散蒙特卡洛实验, 输出 artifacts/seedexp-report.json。
seedexp:
	$(GO) run ./cmd/seedexp --out artifacts/seedexp-report.json

# 导出相邻种子的代表性 PGM 高度图对。
terrainpairs:
	$(GO) run ./cmd/terrainpairs

# 重绘存档图(需要含 matplotlib 的 Python 环境, 见 experiments/README.md)。
figures:
	$(PYTHON) experiments/plot_correlation.py \
		artifacts/seedexp-report.json \
		artifacts/figures/correlated_area_weak.png \
		artifacts/figures/correlated_area_hash.png
	$(PYTHON) experiments/montage_terrain.py \
		artifacts/figures/terrain artifacts/figures/terrain_pairs.png
