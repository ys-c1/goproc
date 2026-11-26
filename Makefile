GO ?= go
PY ?= python3
.PHONY: all test caliper verify

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

# 两画幅最远点对预判/实测比对，产物写入 artifacts/caliper
caliper:
	$(GO) run ./cmd/caliper --out artifacts/caliper
	$(PY) -I analysis/caliper_verify.py artifacts/caliper

# 只做独立复算，不重新生成
verify:
	$(PY) -I analysis/caliper_verify.py artifacts/caliper
