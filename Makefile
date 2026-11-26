GO ?= go
PY ?= python3
.PHONY: all test audit figures clean-audit

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

# 复现“共用置换表重复山脊”审计：Go 权威指标/坐标/原始层网格 -> Python 标记图
audit: test
	$(GO) run ./cmd/repeataudit
	$(PY) analysis/fidelity.py
	$(PY) analysis/make_figures.py
	$(PY) analysis/make_proof_figures.py
	$(PY) analysis/coordinates_doc.py
	$(PY) analysis/ridge_null.py

figures:
	$(PY) analysis/make_figures.py
	$(PY) analysis/make_proof_figures.py
	$(PY) analysis/coordinates_doc.py

clean-audit:
	rm -f artifacts/repetition/raw/*.f64
