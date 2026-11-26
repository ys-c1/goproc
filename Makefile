GO ?= go
.PHONY: all test ridge verify-ridge

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

# 生成脊线掩码、连通段归类图与对照报告。
ridge:
	$(GO) run ./cmd/ridge run --sizes 128,256 --out artifacts/ridge

# 独立复核：不引用 Go 代码，从浮点高度重算并与掩码逐像素比对。
verify-ridge:
	python3 -I tools/verify_ridge.py artifacts/ridge 128
	python3 -I tools/verify_ridge.py artifacts/ridge 256
