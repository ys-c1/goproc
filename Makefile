GO ?= go
PY ?= python3
.PHONY: all test lod charts

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

# 生成 LOD 有效顶点统计、两组误差阈值比值表（CSV/JSON）
lod:
	$(GO) run ./cmd/lodreport --out artifacts
	$(PY) analysis/lod_charts.py artifacts/lod_data.json artifacts

# 仅由已有 lod_data.json 重渲染图与 Markdown 表（含独立复核）
charts:
	$(PY) analysis/lod_charts.py artifacts/lod_data.json artifacts
