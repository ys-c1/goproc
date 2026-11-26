package slope

import (
	"math"
	"testing"
)

// quad 构造二次场 h(x,y)=a X² + b Y² + c X Y + d X + e Y + f，X=x·dx，Y=y·dx。
func quad(n int, dx, a, b, c, d, e, f float64) *ScalarField {
	data := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			X, Y := float64(x)*dx, float64(y)*dx
			data[y*n+x] = a*X*X + b*Y*Y + c*X*Y + d*X + e*Y + f
		}
	}
	return NewScalarField(n, data)
}

const (
	qA = 0.7
	qB = -0.4
	qC = 0.3
	qD = 0.11
	qE = -0.07
)

// trueGrad 是二次场的解析梯度。
func trueGrad(dx float64, x, y int) (gx, gy float64) {
	X, Y := float64(x)*dx, float64(y)*dx
	return 2*qA*X + qC*Y + qD, 2*qB*Y + qC*X + qE
}

// TestBiasIsDxTimesCurvature 验证：单侧差减双侧差 = (dx/2)·κ = 格距×曲率/2。
// 二次场高阶项为零，等式应精确成立。
func TestBiasIsDxTimesCurvature(t *testing.T) {
	for _, dx := range []float64{1, 0.5, 2, 0.25} {
		n := 9
		h := quad(n, dx, qA, qB, qC, qD, qE, 0)
		y := n / 2
		// 左边界 (0,y)：单侧前差减解析双侧差。
		forward := (h.At(1, y) - h.At(0, y)) / dx
		centered, _ := trueGrad(dx, 0, y)
		bias := forward - centered
		// κ = h'' = 2a；理论偏量 (dx/2)·2a = a·dx。
		want := qA * dx
		if math.Abs(bias-want) > 1e-12 {
			t.Fatalf("dx=%v: 偏量=%v 理论(dx·κ/2)=%v", dx, bias, want)
		}
		// 下边界对称核对（n-1 处向后差为 (h(n-1)-h(n-2))/dx）。
		// 偏量符号：F(x0)=(h(x0)-h(x0-dx))/dx = h'(x0) - dx·κ/2，
		// 故 Δ = F - C = -dx·κ/2。
		x := n / 2
		backward := (h.At(x, n-1) - h.At(x, n-2)) / dx
		_, cy := trueGrad(dx, x, n-1)
		if got := backward - cy; math.Abs(got-(-qB*dx)) > 1e-12 {
			t.Fatalf("dx=%v: 下边界偏量=%v 理论=%v", dx, got, -qB*dx)
		}
	}
}

// TestBiasEstimateAtUnitSpacing 验证 dx=1 时 BiasEstimate 给出 κ/2，即真实偏量。
func TestBiasEstimateAtUnitSpacing(t *testing.T) {
	n := 9
	h := quad(n, 1, qA, qB, qC, qD, qE, 0)
	y := n / 2
	forward := h.At(1, y) - h.At(0, y)
	centered, _ := trueGrad(1, 0, y)
	bx, _ := BiasEstimate(h, 0, y)
	if math.Abs(bx-(forward-centered)) > 1e-12 {
		t.Fatalf("BiasEstimate=%v 实际偏量=%v", bx, forward-centered)
	}
}

// TestCompensationExactOnQuadratic 验证：dx=1（本实现的格距单位）时，
// 二次场上补偿后的边缘坡度与解析双侧差一致，即边缘与内侧同一定义、同一精度。
func TestCompensationExactOnQuadratic(t *testing.T) {
	const dx = 1.0
	n := 12
	h := quad(n, dx, qA, qB, qC, qD, qE, 0)
	naive := Gradient(h)
	comp := GradientCompensated(h)
	var maxNaive, maxComp float64
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			if !IsRing(n, x, y) {
				continue
			}
			tgx, tgy := trueGrad(dx, x, y)
			ngx, ngy := naive.At(x, y)
			cgx, cgy := comp.At(x, y)
			maxNaive = math.Max(maxNaive, math.Hypot(ngx-tgx, ngy-tgy))
			maxComp = math.Max(maxComp, math.Hypot(cgx-tgx, cgy-tgy))
		}
	}
	t.Logf("边缘未补偿最大误差=%g 补偿后最大误差=%g", maxNaive, maxComp)
	if maxNaive == 0 {
		t.Fatal("未补偿边缘不应为零误差")
	}
	if maxComp > 1e-9 {
		t.Fatalf("补偿后边缘仍有偏差 %g", maxComp)
	}
}

// TestOnlyRingChanges 验证：补偿只改边缘一圈，内部逐位不变，改变数不超过一圈大小。
func TestOnlyRingChanges(t *testing.T) {
	n := 37
	data := make([]float64, n*n)
	for i := range data {
		x, y := float64(i%n), float64(i/n)
		data[i] = math.Sin(x*0.31) * math.Cos(y*0.17) * math.Exp(-0.001*(x*x+y*y))
	}
	h := NewScalarField(n, data)
	before := Gradient(h)
	after := GradientCompensated(h)
	if bad := RingMismatch(before, after); bad != 0 {
		t.Fatalf("有 %d 个内部像素被改动", bad)
	}
	changed := CountChanged(before, after)
	if changed > RingSize(n) {
		t.Fatalf("改变像素 %d 超过一圈大小 %d", changed, RingSize(n))
	}
	t.Logf("改变像素=%d，边缘一圈=%d", changed, RingSize(n))
}

// TestCompensationEqualsBias 是核心恒等式：边缘一圈上，
//
//	未补偿坡度 - 补偿后坡度  =  BiasEstimate
//
// 逐分量精确成立。这直接证明「按该偏量做边界延拓补偿」，
// 而不是只统计上接近。用非二次的起伏场，排除二次场巧合。
func TestCompensationEqualsBias(t *testing.T) {
	n := 41
	data := make([]float64, n*n)
	for i := range data {
		x, y := float64(i%n), float64(i/n)
		data[i] = math.Sin(x*0.23+0.4) * math.Cos(y*0.19-0.7) * (1 + 0.3*math.Sin(x*0.05*y*0.03))
	}
	h := NewScalarField(n, data)
	raw := Gradient(h)
	comp := GradientCompensated(h)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			if !IsRing(n, x, y) {
				continue
			}
			bx, by := BiasEstimate(h, x, y)
			rgx, rgy := raw.At(x, y)
			cgx, cgy := comp.At(x, y)
			if math.Abs((rgx-cgx)-bx) > 1e-12 {
				t.Fatalf("(%d,%d) x 分量: 差=%v 偏量=%v", x, y, rgx-cgx, bx)
			}
			if math.Abs((rgy-cgy)-by) > 1e-12 {
				t.Fatalf("(%d,%d) y 分量: 差=%v 偏量=%v", x, y, rgy-cgy, by)
			}
		}
	}
}

// TestConvergenceOrder 是「边缘与内侧连续」的严格判据：固定物理域，
// 缩小格距 dx，看边缘误差的收敛阶。
//
//	内部（中心差）          应为 2 阶
//	边缘未补偿（单侧差）     应为 1 阶  <- 偏量造成的一阶降级
//	边缘补偿后             应回到约 2 阶，与内部同量级
//
// 三者合并即证明补偿消掉了边界偏置、边缘恢复与内侧相同的精度。
func TestConvergenceOrder(t *testing.T) {
	f := func(x, y float64) float64 {
		return math.Sin(1.3*x+0.4) * math.Cos(0.9*y-0.2) * math.Exp(0.02*(x+y))
	}
	const e = 1e-6
	dfx := func(x, y float64) float64 { return (f(x+e, y) - f(x-e, y)) / (2 * e) }
	dfy := func(x, y float64) float64 { return (f(x, y+e) - f(x, y-e)) / (2 * e) }

	type sample struct{ inner, raw, comp float64 }
	var errs []sample
	for _, n := range []int{32, 64, 128, 256} {
		dx := 6.4 / float64(n)
		data := make([]float64, n*n)
		for y := 0; y < n; y++ {
			for x := 0; x < n; x++ {
				data[y*n+x] = f(float64(x)*dx, float64(y)*dx)
			}
		}
		h := NewScalarField(n, data)
		raw := Gradient(h)
		comp := GradientCompensated(h)
		var inSq, rawSq, compSq float64
		var inN, ringN int
		for y := 0; y < n; y++ {
			for x := 0; x < n; x++ {
				tx, ty := dfx(float64(x)*dx, float64(y)*dx), dfy(float64(x)*dx, float64(y)*dx)
				rx, ry := raw.At(x, y)
				cx, cy := comp.At(x, y)
				// 像素单位斜率 / dx 即物理导数。
				re := math.Hypot(rx/dx-tx, ry/dx-ty)
				ce := math.Hypot(cx/dx-tx, cy/dx-ty)
				if IsRing(n, x, y) {
					rawSq += re * re
					compSq += ce * ce
					ringN++
				} else {
					inSq += re * re
					inN++
				}
			}
		}
		errs = append(errs, sample{
			inner: math.Sqrt(inSq / float64(inN)),
			raw:   math.Sqrt(rawSq / float64(ringN)),
			comp:  math.Sqrt(compSq / float64(ringN)),
		})
	}
	// 从相邻两档估计收敛阶（每次都减半 dx）。
	order := func(get func(sample) float64) float64 {
		var sum float64
		for i := 1; i < len(errs); i++ {
			sum += math.Log2(get(errs[i-1]) / get(errs[i]))
		}
		return sum / float64(len(errs)-1)
	}
	oInner := order(func(s sample) float64 { return s.inner })
	oRaw := order(func(s sample) float64 { return s.raw })
	oComp := order(func(s sample) float64 { return s.comp })
	t.Logf("收敛阶: 内部=%.2f 边缘前=%.2f 边缘后=%.2f", oInner, oRaw, oComp)
	if math.Abs(oInner-2) > 0.25 {
		t.Fatalf("内部收敛阶应约 2，实为 %.2f", oInner)
	}
	if math.Abs(oRaw-1) > 0.25 {
		t.Fatalf("未补偿边缘收敛阶应约 1（一阶偏置），实为 %.2f", oRaw)
	}
	if math.Abs(oComp-2) > 0.25 {
		t.Fatalf("补偿后边缘收敛阶应回到约 2，实为 %.2f", oComp)
	}
	// 补偿后边缘误差须回落至与内部同量级。
	last := errs[len(errs)-1]
	if last.comp > 3*last.inner {
		t.Fatalf("补偿后边缘误差 %.3e 仍是内部 %.3e 的 %.1f 倍", last.comp, last.inner, last.comp/last.inner)
	}
}

// TestRingSize 校验一圈按一格计的计数。
func TestRingSize(t *testing.T) {
	if got := RingSize(8); got != 28 {
		t.Fatalf("RingSize(8)=%d", got)
	}
	if got := RingSize(2); got != 4 {
		t.Fatalf("RingSize(2)=%d", got)
	}
}
