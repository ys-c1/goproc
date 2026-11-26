package fog

import (
	"math"
	"testing"
)

func TestContrastThresholdEquivalence(t *testing.T) {
	eps := 0.05
	tau := TauThreshold(eps)
	if math.Abs(tau+math.Log(eps)) > 1e-12 {
		t.Fatalf("tau* 应为 -ln eps, got %v", tau)
	}
	// C>eps 与 tau<tau* 必须在截断点两侧一致。
	chi := 7.0
	hc := HeightCutoff(chi, eps)
	for _, h := range []float64{0, hc - 1e-6, hc + 1e-6, 1} {
		c := ContrastAt(h, chi)
		tauH := chi * (1 - h)
		if (c > eps) != (tauH < tau) {
			t.Fatalf("判据不等价 h=%v C=%v tau=%v", h, c, tauH)
		}
	}
}

func TestPredictCDFMatchesLoop(t *testing.T) {
	h := []float64{0.01, 0.2, 0.35, 0.5, 0.62, 0.8, 0.99}
	for _, chi := range []float64{2, 4, 6, 9, 12} {
		a := Predict(h, chi, DefaultEpsilon)
		b := PredictViaCDF(h, chi, DefaultEpsilon)
		if a.Visible != b.Visible || math.Abs(a.Fraction-b.Fraction) > 1e-15 {
			t.Fatalf("chi=%v loop=%d/%v cdf=%d/%v", chi, a.Visible, a.Fraction, b.Visible, b.Fraction)
		}
	}
}

func TestProductInvariance(t *testing.T) {
	// 同一归一化地形，任意 (beta,R) 组合只要 beta*R 相同，预判比例必须相同。
	h := make([]float64, 0, 101)
	for i := 0; i <= 100; i++ {
		h = append(h, float64(i)/100)
	}
	target := 8.0
	base := Predict(h, target, DefaultEpsilon).Fraction
	for _, R := range []float64{0.25, 0.5, 1, 2, 4} {
		beta := target / R
		got := Predict(h, Chi(beta, R), DefaultEpsilon).Fraction
		if math.Abs(got-base) > 1e-15 {
			t.Fatalf("R=%v beta=%v fraction=%v base=%v", R, beta, got, base)
		}
	}
}

func TestAllVisibleWhenChiBelowTau(t *testing.T) {
	h := []float64{0, 0.25, 0.5, 0.75, 1}
	// chi 严格小于 tau* 时，最低点 C=e^-chi > eps，全部可见。
	c := Predict(h, 2.5, DefaultEpsilon)
	if c.Visible != len(h) {
		t.Fatalf("应全部可见, got %d/%d", c.Visible, len(h))
	}
}

func TestRenderCountAndMask(t *testing.T) {
	n := 16
	h := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			h[y*n+x] = float64(x) / float64(n-1)
		}
	}
	res := Render(h, n, DefaultRenderOptions(9))
	if res.FloatCount.Total != n*n {
		t.Fatalf("total=%d", res.FloatCount.Total)
	}
	maskWhite := 0
	for _, px := range res.Mask.Pix {
		if px == 255 {
			maskWhite++
		}
	}
	if maskWhite != res.FloatCount.Visible {
		t.Fatalf("mask 白点数 %d != 可见数 %d", maskWhite, res.FloatCount.Visible)
	}
	if got := CountContrastGray8(res.Contrast8, 9, DefaultEpsilon); got.Total != n*n {
		t.Fatalf("readback total=%d", got.Total)
	}
}

func TestRelativeDiff(t *testing.T) {
	if d := RelativeDiff(0.9, 1.0); math.Abs(d-0.1) > 1e-12 {
		t.Fatalf("reldiff=%v", d)
	}
	if !math.IsNaN(RelativeDiff(1, 0)) {
		t.Fatal("基准为零应返回 NaN")
	}
}
