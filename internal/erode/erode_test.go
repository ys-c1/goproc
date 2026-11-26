package erode

import (
	"math"
	"testing"
)

func testField(size int) []float64 {
	h := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			h[y*size+x] = 0.5 + 0.3*math.Sin(float64(x)*0.3)*math.Cos(float64(y)*0.2)
		}
	}
	return h
}

func TestRoundConservesMass(t *testing.T) {
	h := testField(32)
	sum := func() float64 {
		var s float64
		for _, v := range h {
			s += v
		}
		return s
	}
	before := sum()
	for i := 0; i < 5; i++ {
		Round(h, 32, 0, 0.1)
	}
	if math.Abs(sum()-before) > 1e-9 {
		t.Fatalf("质量不守恒: before=%v after=%v", before, sum())
	}
}

func TestRoundDeterministic(t *testing.T) {
	a, b := testField(32), testField(32)
	for i := 0; i < 3; i++ {
		Round(a, 32, 0, 0.1)
		Round(b, 32, 0, 0.1)
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("第 %d 格不一致: %v != %v", i, a[i], b[i])
		}
	}
}

func TestErosionFlattens(t *testing.T) {
	h := testField(32)
	m0 := Median(Slopes(h, 32, 8))
	v0 := Variance(Slopes(h, 32, 8))
	for i := 0; i < 20; i++ {
		Round(h, 32, 0, 0.1)
	}
	m1 := Median(Slopes(h, 32, 8))
	v1 := Variance(Slopes(h, 32, 8))
	if m1 >= m0 {
		t.Fatalf("坡度中位数未下降: %v -> %v", m0, m1)
	}
	if v1 >= v0 {
		t.Fatalf("坡度方差未下降: %v -> %v", v0, v1)
	}
}

func TestTalusBlocksShallow(t *testing.T) {
	// 高差全部低于坡止角时一轮后不得有任何变化。
	h := []float64{0.5, 0.5001, 0.5002, 0.5003}
	before := make([]float64, len(h))
	copy(before, h)
	Round(h, 2, 0.01, 0.1)
	for i := range h {
		if h[i] != before[i] {
			t.Fatalf("低于坡止角仍发生搬运: 第 %d 格 %v -> %v", i, before[i], h[i])
		}
	}
}
