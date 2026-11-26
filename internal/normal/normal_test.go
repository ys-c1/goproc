package normal

import (
	"math"
	"testing"
)

// 平面 h=x 应有恒定梯度 (1,0)，法线指向 (-1,0,·)，方位角恒为 π。
func TestPlaneGradient(t *testing.T) {
	size := 8
	h := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			h[y*size+x] = float64(x) * 0.25
		}
	}
	f := FromHeightmap(size, h)
	i := 4*size + 4
	if math.Abs(f.P[i]-0.25) > 1e-12 || math.Abs(f.Q[i]) > 1e-12 {
		t.Fatalf("梯度错误: p=%v q=%v", f.P[i], f.Q[i])
	}
	if math.Abs(f.Nx[i]+0.25/math.Sqrt(1+0.25*0.25)) > 1e-12 {
		t.Fatalf("法线 x 分量错误: %v", f.Nx[i])
	}
	if phi := f.Azimuth(i); math.Abs(phi-math.Pi) > 1e-9 {
		t.Fatalf("平面法线方位角应为 π，实际 %v", phi)
	}
}

// 方位角与梯度角必须恰好相差 π（mod 2π）。
func TestAzimuthGradientOffset(t *testing.T) {
	size := 16
	h := make([]float64, size*size)
	rng := deterministic()
	for i := range h {
		h[i] = rng()
	}
	f := FromHeightmap(size, h)
	for y := 1; y < size-1; y++ {
		for x := 1; x < size-1; x++ {
			i := y*size + x
			d := math.Mod(f.Azimuth(i)-f.GradientAngle(i)+2*math.Pi, 2*math.Pi)
			if math.Abs(d-math.Pi) > 1e-9 {
				t.Fatalf("像素 (%d,%d) 方位角差不是 π: %v", x, y, d)
			}
		}
	}
}

func TestHistogramSumsToOne(t *testing.T) {
	size := 32
	h := make([]float64, size*size)
	rng := deterministic()
	for i := range h {
		h[i] = rng()
	}
	f := FromHeightmap(size, h)
	for _, nb := range []int{8, 16} {
		counts, frac := f.Histogram(nb)
		var sumC int
		var sumF float64
		for k := range frac {
			sumC += counts[k]
			sumF += frac[k]
		}
		if sumC != f.Interior {
			t.Fatalf("nb=%d 计数和 %d ≠ 内部像素 %d", nb, sumC, f.Interior)
		}
		if math.Abs(sumF-1) > 1e-12 {
			t.Fatalf("nb=%d 占比和 %v ≠ 1", nb, sumF)
		}
	}
}

// 简单 LCG，避免测试依赖全局随机源。
func deterministic() func() float64 {
	seed := uint64(12345)
	return func() float64 {
		seed = seed*6364136223846793005 + 1442695040888963407
		return float64(seed%10000) / 10000
	}
}
