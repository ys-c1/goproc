package api

import (
	"bytes"
	"testing"
)

func TestDeterministic(t *testing.T) {
	opt := DefaultOptions()
	a, err := Generate(opt)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Generate(opt)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a.ToPGM(), b.ToPGM()) {
		t.Fatal("同种子两次结果不同")
	}
}

func TestSeedChanges(t *testing.T) {
	a, _ := Generate(DefaultOptions())
	opt := DefaultOptions()
	opt.Seed = 2
	b, _ := Generate(opt)
	if bytes.Equal(a.ToPGM(), b.ToPGM()) {
		t.Fatal("不同种子结果相同")
	}
}

func TestRangeNormalized(t *testing.T) {
	h, err := Generate(DefaultOptions())
	if err != nil {
		t.Fatal(err)
	}
	lo, hi := h.Data[0], h.Data[0]
	for _, v := range h.Data {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	if lo < 0 || hi > 1 {
		t.Fatalf("值域越界: %v %v", lo, hi)
	}
}

func TestRejects(t *testing.T) {
	if _, err := Generate(Options{Size: 0, Octaves: 4, Lacunarity: 2, Gain: 0.5, Frequency: 0.02}); err == nil {
		t.Fatal("尺寸为零应报错")
	}
	bad := DefaultOptions()
	bad.Gain = 1
	if _, err := Generate(bad); err == nil {
		t.Fatal("增益越界应报错")
	}
	badKernel := DefaultOptions()
	badKernel.Kernel = "spline"
	if _, err := Generate(badKernel); err == nil {
		t.Fatal("非法核名应报错")
	}
}

func TestKernelSelection(t *testing.T) {
	cubic := DefaultOptions()
	cubic.Kernel = "cubic"
	quintic := DefaultOptions() // 默认五次核
	a, err := Generate(cubic)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Generate(quintic)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Equal(a.ToPGM(), b.ToPGM()) {
		t.Fatal("三次核与五次核输出不应相同")
	}
	// 空 Kernel 与显式 quintic 等价（默认路径）
	empty := DefaultOptions()
	empty.Kernel = ""
	c, _ := Generate(empty)
	if !bytes.Equal(b.ToPGM(), c.ToPGM()) {
		t.Fatal("空核名应回退到默认五次核")
	}
}
