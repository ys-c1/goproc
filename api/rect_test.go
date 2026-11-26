package api

import (
	"math"
	"testing"
)

func TestGenerateRectShapeAndRange(t *testing.T) {
	h, err := GenerateRect(RectOptions{NX: 16, NY: 32, L: 8, RX: 1, RY: 4,
		Octaves: 2, Lacunarity: 2, Gain: 0.5, Seed: 7})
	if err != nil {
		t.Fatal(err)
	}
	if h.NX != 16 || h.NY != 32 || len(h.Data) != 16*32 {
		t.Fatalf("形状错误: %dx%d len=%d", h.NX, h.NY, len(h.Data))
	}
	norm := h.Normalized()
	lo, hi := norm[0], norm[0]
	for _, v := range norm {
		lo = math.Min(lo, v)
		hi = math.Max(hi, v)
	}
	if lo < 0 || hi > 1 {
		t.Fatalf("归一化越界: %v %v", lo, hi)
	}
	pgm := h.ToPGM()
	if len(pgm) == 0 || string(pgm[:2]) != "P5" {
		t.Fatal("PGM 头非法")
	}
}

func TestGenerateRectRejects(t *testing.T) {
	if _, err := GenerateRect(RectOptions{NX: 0, NY: 8, L: 8, RX: 1, RY: 1,
		Octaves: 1, Lacunarity: 2, Gain: 0.5, Seed: 1}); err == nil {
		t.Fatal("NX=0 应报错")
	}
	if _, err := GenerateRect(RectOptions{NX: 8, NY: 8, L: 8, RX: 1, RY: -1,
		Octaves: 1, Lacunarity: 2, Gain: 0.5, Seed: 1}); err == nil {
		t.Fatal("负缩放比应报错")
	}
}
