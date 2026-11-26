package main

import (
	"math"
	"testing"

	"goproc/internal/noise"
)

// 坐标必须严格满足 c_k = c0 / rho^k, side_k = anchorW / rho^k。
func TestLevelBoxRatio(t *testing.T) {
	cases := []struct {
		cx0, cy0, k, cx, cy, side int
	}{
		{256, 256, 0, 256, 256, 256},
		{256, 256, 1, 128, 128, 128},
		{768, 512, 2, 192, 128, 64},
		{768, 768, 5, 24, 24, 8},
	}
	for _, c := range cases {
		cx, cy, side := levelBox(c.cx0, c.cy0, c.k, 2)
		if cx != c.cx || cy != c.cy || side != c.side {
			t.Fatalf("levelBox(%d,%d,k=%d)=(%d,%d,%d), want (%d,%d,%d)",
				c.cx0, c.cy0, c.k, cx, cy, side, c.cx, c.cy, c.side)
		}
	}
}

// 周期关系 Pi = P/gcd(P,rho-1)：rho=2 时 rho-1=1，故 Pi=P=256。
func TestCoincidencePeriod(t *testing.T) {
	if got := PermPeriod / gcd(PermPeriod, 1); got != 256 {
		t.Fatalf("rho=2 coincidence period=%d, want 256", got)
	}
	// rho=3 -> rho-1=2, gcd(256,2)=2 -> 128。
	if got := PermPeriod / gcd(PermPeriod, 2); got != 128 {
		t.Fatalf("rho=3 coincidence period=%d, want 128", got)
	}
}

// 核心定理：共用置换表时 L_{i+1}(p)=L_i(rho p) 逐点成立，因此所有比例对齐
// 重复片的 Pearson 相关必须恰为 1；按层派生子表后应跌到零附近。
func TestSharedIdentityAndLayeredDecorrelation(t *testing.T) {
	const size = 1024
	const freq = 0.02
	const rho = 2
	const seed = 7
	const maxK = 6

	sharedSrc := noise.New(seed)
	shared := make([][]float64, maxK)
	layered := make([][]float64, maxK)
	for i := 0; i < maxK; i++ {
		shared[i] = layerGrid(sharedSrc, size, freq, rho, i)
		layered[i] = layerGrid(noise.NewLayer(seed, i), size, freq, rho, i)
	}

	// 先直接验证尺度恒等式在完整网格上的最大误差（容限 1e-12）。
	for i := 0; i < maxK-1; i++ {
		fi := freq * math.Pow(rho, float64(i))
		var maxDiff float64
		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				want := sharedSrc.Value(float64(x)*rho*fi, float64(y)*rho*fi)
				if d := math.Abs(shared[i+1][y*size+x] - want); d > maxDiff {
					maxDiff = d
				}
			}
		}
		if maxDiff > 1e-12 {
			t.Fatalf("scale identity layer %d maxdiff=%g", i, maxDiff)
		}
	}

	for _, K := range []int{2, 4, 6} {
		var cs, cl []float64
		for _, cx := range centers {
			for _, cy := range centers {
				for k := 0; k < K-1; k++ {
					A, B := pairVectors(shared, size, rho, cx, cy, k)
					cs = append(cs, pearson(A, B))
					A2, B2 := pairVectors(layered, size, rho, cx, cy, k)
					cl = append(cl, pearson(A2, B2))
				}
			}
		}
		for i, v := range cs {
			if math.Abs(v-1) > 1e-12 {
				t.Fatalf("K=%d shared pair %d corr=%.15f, want 1", K, i, v)
			}
		}
		m, _ := meanSD(cl)
		if math.Abs(m) > 0.25 {
			t.Fatalf("K=%d layered mean corr=%.4f, want near 0", K, m)
		}
	}
}

// 不同层的子种子必须不同且确定。
func TestLayerSeedDistinctDeterministic(t *testing.T) {
	if noise.LayerSeed(7, 0) != noise.LayerSeed(7, 0) {
		t.Fatal("LayerSeed 不确定")
	}
	seen := map[uint64]bool{}
	for i := 0; i < 12; i++ {
		s := noise.LayerSeed(7, i)
		if seen[s] {
			t.Fatalf("layer %d seed collision", i)
		}
		seen[s] = true
	}
}
