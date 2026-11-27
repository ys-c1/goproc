GO ?= go
.PHONY: all test norm

all: test

test:
	$(GO) vet ./...
	$(GO) test ./...

# 打印差分核的推导与两种格距下的角误差对照。
norm:
	$(GO) test ./internal/nrm/ -v -run 'Test'
