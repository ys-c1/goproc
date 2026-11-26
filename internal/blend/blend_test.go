package blend

import (
	"math"
	"testing"
)

// syntheticRow 构造 Δ(a)=0 的两层：L 为二次曲线，H=L+Δ，Δ 为正弦形。
func syntheticRow(n, a, b int) (l, h []float64) {
	l = make([]float64, n)
	h = make([]float64, n)
	for i := 0; i < n; i++ {
		x := float64(i)
		l[i] = 0.01 * x * x
		d := 0.0
		if i >= a {
			d = math.Sin(0.7 * float64(i-a))
		}
		h[i] = l[i] + d
	}
	return l, h
}

func TestBoundaryValues(t *testing.T) {
	for _, wf := range []Weight{Linear, Smoothstep} {
		if wf(0) != 0 || wf(1) != 1 {
			t.Fatalf("weight endpoints wrong: w(0)=%v w(1)=%v", wf(0), wf(1))
		}
	}
	dm := []float64{0.1, -0.2, 0.3, 0.05}
	w := OptimalWeights(dm, Eps2FromD(dm, 1e-3))
	if w[0] != 0 || w[len(w)-1] != 1 {
		t.Fatalf("optimal endpoints wrong: %v %v", w[0], w[len(w)-1])
	}
}

// TestELStationarity 检查离散 EL 方程 c_i·(w_{i+1}-w_i) 在带内为常数。
func TestELStationarity(t *testing.T) {
	l, h := syntheticRow(40, 10, 30)
	dm := EdgeMidDeltas(l, h, 10, 30)
	eps2 := Eps2FromD(dm, 1e-3)
	w := OptimalWeights(dm, eps2)
	flux0 := (dm[0]*dm[0] + eps2) * (w[1] - w[0])
	for i := range dm {
		flux := (dm[i]*dm[i] + eps2) * (w[i+1] - w[i])
		if math.Abs(flux-flux0) > 1e-12*math.Max(1, math.Abs(flux0)) {
			t.Fatalf("flux not constant at edge %d: %v vs %v", i, flux, flux0)
		}
	}
}

// TestCauchyEquality 无正则、Δmid 全非零时，最优能量应等于 Cauchy 下界。
func TestCauchyEquality(t *testing.T) {
	d := []float64{0.4, -0.9, 0.7, 1.2, -0.5}
	w := OptimalWeights(d, 0)
	got := RidgeEnergy(w, d, 0, 1)
	want := CauchyBound(d, 1)
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("optimal energy %v != Cauchy bound %v", got, want)
	}
	// 线性/平滑权重的能量严格更大。
	wl := SampleWeight(Linear, 0, len(d))
	ws := SampleWeight(Smoothstep, 0, len(d))
	if RidgeEnergy(wl, d, 0, 1) <= got {
		t.Fatal("linear weight unexpectedly optimal")
	}
	if RidgeEnergy(ws, d, 0, 1) <= got {
		t.Fatal("smoothstep weight unexpectedly optimal")
	}
}

// TestBlendIdentity 验证 B-B* = ∫ w'Δ dx（Δ(a)=0 时的积分恒等式）。
func TestBlendIdentity(t *testing.T) {
	l, h := syntheticRow(60, 15, 45)
	a, b := 15, 45
	for _, wf := range []Weight{Linear, Smoothstep} {
		B := BlendRow(l, h, a, b, wf)
		w := SampleWeight(wf, a, b)
		dm := EdgeMidDeltas(l, h, a, b)
		star := make([]float64, len(l))
		copy(star, l)
		for i := a; i < b; i++ {
			wmid := 0.5 * (w[i-a] + w[i+1-a])
			dL := l[i+1] - l[i]
			dH := h[i+1] - h[i]
			star[i+1] = star[i] + (1-wmid)*dL + wmid*dH
		}
		integral := 0.0
		for k := a + 1; k <= b; k++ {
			integral += (w[k-a] - w[k-1-a]) * dm[k-1-a]
			if got := B[k] - star[k]; math.Abs(got-integral) > 1e-10 {
				t.Fatalf("%s identity failed at %d: %v vs %v",
					weightName(wf), k, got, integral)
			}
		}
	}
}

// TestRegularizedEquality 正则化最优能量应等于正则化下界，即使 Δmid 含零。
func TestRegularizedEquality(t *testing.T) {
	d := []float64{0.4, -0.9, 0.0, 1.2, -0.5}
	eps2 := 0.01
	for _, dx := range []float64{1, 0.5, 2.5} {
		w := OptimalWeights(d, eps2)
		got := RidgeEnergy(w, d, eps2, dx)
		want := RegularizedBound(d, eps2, dx)
		if math.Abs(got-want) > 1e-12*math.Max(1, math.Abs(want)) {
			t.Fatalf("dx=%v energy %v != bound %v", dx, got, want)
		}
	}
	if !math.IsNaN(CauchyBound(d, 1)) {
		t.Fatal("unregularized Cauchy bound should be NaN when Δmid contains zero")
	}
}

// TestOptimalBeatsNaive 在带入口对齐的真实行上，最优权重能量应低于两个朴素权重。
func TestOptimalBeatsNaive(t *testing.T) {
	l, h := syntheticRow(120, 40, 80)
	a, b := 40, 80
	dm := EdgeMidDeltas(l, h, a, b)
	eps2 := Eps2FromD(dm, 0.15)
	wOpt := OptimalWeights(dm, eps2)
	eOpt := RidgeEnergy(wOpt, dm, eps2, 1)
	for _, wf := range []Weight{Linear, Smoothstep} {
		e := RidgeEnergy(SampleWeight(wf, a, b), dm, eps2, 1)
		if e <= eOpt {
			t.Fatalf("%s energy %v <= optimal %v", weightName(wf), e, eOpt)
		}
	}
}

func weightName(wf Weight) string {
	if wf(0.5) == 0.5 {
		return "linear"
	}
	return "smoothstep"
}
