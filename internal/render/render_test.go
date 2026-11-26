package render

import (
	"math"
	"testing"
)

// 化简的核心恒等式：N·L ≥ 0  ⟺  u ≤ tan(elev)
// 其中 u = ∇h·w 是沿光照方位的方向导数。
// 证明：N ∝ (-p,-q,1)，L = cos(e)w + sin(e)ẑ，
//
//	N·L = (sin e - u cos e)/√(1+p²+q²)，
//	分母为正，故 N·L ≥ 0 ⟺ sin e ≥ u cos e ⟺ u ≤ tan e。□
func TestLitEquivalentToSlopeInequality(t *testing.T) {
	hm := &Heightmap{Size: 16, Data: make([]float64, 256)}
	for i := range hm.Data {
		x, y := i%16, i/16
		hm.Data[i] = 0.4*math.Sin(float64(x)*0.7) + 0.3*math.Cos(float64(y)*0.5)
	}
	for _, elevDeg := range []float64{2, 15, 45, 70} {
		for _, azDeg := range []float64{45, 135, 210} {
			p := Params{Azimuth: azDeg * math.Pi / 180, Elev: elevDeg * math.Pi / 180, Ambient: 0.25}
			inc := hm.Incidence(p)
			u := hm.DirectionalSlope(p.Azimuth)
			tanE := math.Tan(p.Elev)
			for i := range inc {
				// Incidence 已将负值截零，故用"是否为正"判 N·L > 0；
				// 截断边界两侧都可能落到 0，跳过浮点相等附近。
				litByDot := inc[i] > 0
				litByU := u[i] < tanE
				if inc[i] == 0 && math.Abs(u[i]-tanE) < 1e-9 {
					continue
				}
				if litByDot != litByU {
					t.Fatalf("elev=%v az=%v 第 %d 格: N·L>0=%v 但 u<tanE=%v (N·L=%.9f u=%.9f tanE=%.9f)",
						elevDeg, azDeg, i, litByDot, litByU, inc[i], u[i], tanE)
				}
			}
		}
	}
}

// 合取化简：A ∧ B  ⟺  ∀s>0: h(x+sw) - h(x) - s·tan e ≤ τ  并且  u ≤ tan e。
// 第二项正是第一项在 s→0 时的导数条件，因此合取等价于单条"光线包络"不等式：
//
//	max_{s≥0} [ h(x+sw) - h(x) - s·tan e ] ≤ τ   且   φ'(0) ≤ 0,
//
// 而 φ(0)=0、φ'(0)=u-tan e，故 A 就是 B 的零阶极限条件，两者在 τ→0 时合一。
func TestConjunctionEqualsRayEnvelope(t *testing.T) {
	// 构造一条已知的过零点剖面：沿 +x 方向高度先升后降，模拟山脊遮挡。
	const size = 32
	hm := &Heightmap{Size: size, Data: make([]float64, size*size)}
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			hm.Data[y*size+x] = 0.5 * math.Exp(-math.Pow(float64(x-18)/5, 2))
		}
	}
	p := Params{Azimuth: 0, Elev: 5 * math.Pi / 180, Ambient: 0.25}
	const tau = 0.02
	inc := hm.Incidence(p)
	gap := hm.DepthGap(p)
	u := hm.DirectionalSlope(p.Azimuth)
	tanE := math.Tan(p.Elev)

	for i := range hm.Data {
		a := u[i] <= tanE                // A: N·L ≥ 0
		b := gap[i] > 0 && gap[i] <= tau // B: 0 < gap ≤ τ
		direct := a && b

		// 化简后：φ(s) = h(x+sw) - h(x) - s·tan e，φ'(0) ≤ 0 且 max φ ≤ τ 且 max φ > 0
		phiMax, phiSlope := rayEnvelope(hm, p, i%size, i/size)
		reduced := phiSlope <= 1e-12 && phiMax > 0 && phiMax <= tau

		if direct != reduced {
			t.Fatalf("第 %d 格 (x=%d,y=%d): 原式=%v 化简式=%v (gap=%.6f φmax=%.6f φ'(0)=%.6f N·L=%.6f)",
				i, i%size, i/size, direct, reduced, gap[i], phiMax, phiSlope, inc[i])
		}
	}
}

func rayEnvelope(h *Heightmap, p Params, x, y int) (maxPhi, slope0 float64) {
	ca, sa := math.Cos(p.Azimuth), math.Sin(p.Azimuth)
	tanE := math.Tan(p.Elev)
	maxPhi = 0
	for s := 1.0; s <= float64(h.Size); s++ {
		ix := int(math.Round(float64(x) + ca*s))
		iy := int(math.Round(float64(y) + sa*s))
		if ix < 0 || iy < 0 || ix >= h.Size || iy >= h.Size {
			break
		}
		phi := h.At(ix, iy) - h.At(x, y) - s*tanE
		if phi > maxPhi {
			maxPhi = phi
		}
	}
	dx := (h.AtClamped(x+1, y) - h.AtClamped(x-1, y)) * 0.5
	dy := (h.AtClamped(x, y+1) - h.AtClamped(x, y-1)) * 0.5
	return maxPhi, dx*ca + dy*sa - tanE
}

// 深度差为 0（完全未被遮挡）时判据 B 必须为假，否则"冲突"会把整片平地卷进来。
func TestMarginalRequiresOcclusion(t *testing.T) {
	hm := &Heightmap{Size: 8, Data: make([]float64, 64)} // 完全平坦
	p := Params{Azimuth: 0, Elev: 30 * math.Pi / 180, Ambient: 0.25}
	cr := hm.Conflict(p, 0.02)
	if n := Count(cr.Marginal); n != 0 {
		t.Fatalf("平地不应有边缘遮挡格，得到 %d", n)
	}
	if n := Count(cr.Conflict); n != 0 {
		t.Fatalf("平地不应有冲突格，得到 %d", n)
	}
}
