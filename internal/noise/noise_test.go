package noise

import (
	"math"
	"testing"
)

func TestSeedDerivation(t *testing.T) {
	a := New(7)
	b := New(7)
	if a.Value(1.5, 2.5) != b.Value(1.5, 2.5) {
		t.Fatal("同种子不同值")
	}
	c := New(8)
	if a.Value(1.5, 2.5) == c.Value(1.5, 2.5) {
		t.Fatal("不同种子同值")
	}
}

// TestSeedIndependenceCovariance 是种子独立性命题的回归守卫：
// 三档种子间隔下，对应位置格点值的样本协方差必须落在 H0 的 4σ 波动门限内。
// 完整分档对照与报告由 cmd/seedcov 输出，这里只用小样本保证 CI 不被绕过。
func TestSeedIndependenceCovariance(t *testing.T) {
	const n = 1024
	// Var(G)=E[G²]≈0.33595，H0 下 Var(Ĉ)=Var(G)²/n。
	theoryVar := 0.0
	for h := 0; h < 256; h++ {
		g := float64(h)/127.5 - 1
		theoryVar += g * g
	}
	theoryVar /= 256
	gate := 4 * theoryVar / math.Sqrt(n)

	for _, delta := range []uint64{1, 1 << 16, 1 << 32} {
		var sumX, sumY, sumXY, sumX2, sumY2 float64
		for k := 0; k < n; k++ {
			s1 := 5_000_000 + uint64(k)*7919
			s2 := s1 + delta
			x := New(s1).Lattice(k&255, (k>>8)&255)
			y := New(s2).Lattice(k&255, (k>>8)&255)
			sumX += x
			sumY += y
			sumXY += x * y
			sumX2 += x * x
			sumY2 += y * y
		}
		fn := float64(n)
		cov := sumXY/fn - (sumX/fn)*(sumY/fn)
		if math.Abs(cov) > gate {
			t.Fatalf("delta=%d 协方差 %+.5f 超出 4σ 门限 ±%.5f，种子派生有偏", delta, cov, gate)
		}
	}
}

func TestBounded(t *testing.T) {
	s := New(3)
	for i := 0; i < 1000; i++ {
		v := s.Value(float64(i)*0.37, float64(i)*0.11)
		if v < -1.5 || v > 1.5 {
			t.Fatalf("越界: %v", v)
		}
	}
}
