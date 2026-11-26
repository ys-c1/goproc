package tilt

import (
	"math"
	"testing"

	"goproc/internal/fbm"
	"goproc/internal/noise"
)

func field(seed uint64, size int, f0 float64, oct int, lac, gain float64) []float64 {
	src := noise.New(seed)
	out := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			out[y*size+x] = fbm.Eval(src, float64(x)*f0, float64(y)*f0, oct, lac, gain)
		}
	}
	return out
}

// TestKappaClosedForm 用解析式复算 Kappa，并用极小步长数值微分核对。
func TestKappaClosedForm(t *testing.T) {
	const sigmaC2 = 21845.0 / 65025.0 // 角点 g=k/127.5-1 的方差
	ev2 := 36.0/11 - 18 + 345.0/9 - 37.5 + 100.0/7
	k2 := 2 * (10.0 / 7) * (2 * sigmaC2) * (1 - 2*0.5 + 2*ev2)
	if got := math.Sqrt(k2); math.Abs(got-Kappa) > 1e-4 {
		t.Fatalf("Kappa 闭式 %.6f 与常量 %.6f 不符", got, Kappa)
	}

	s := noise.New(4242)
	var sum float64
	const n, h = 20000, 1e-4
	state := uint64(9)
	next := func() float64 {
		state = state*6364136223846793005 + 1442695040888963407
		return float64((state>>11)%20000) * 0.05
	}
	for i := 0; i < n; i++ {
		px, py := next(), next()
		gx := (s.Value(px+h, py) - s.Value(px-h, py)) / (2 * h)
		gy := (s.Value(px, py+h) - s.Value(px, py-h)) / (2 * h)
		sum += gx*gx + gy*gy
	}
	if got := math.Sqrt(sum / n); math.Abs(got-Kappa)/Kappa > 0.03 {
		t.Fatalf("Kappa 数值 %.4f 与解析 %.4f 相差超过 3%%", got, Kappa)
	}
}

// TestOverviewScalesWithBaseFrequency 验证斜率正比于基础频率、倾角是 atan 的关系。
func TestOverviewScalesWithBaseFrequency(t *testing.T) {
	s1, _ := Overview(0.02, 5, 2, 0.5)
	s2, _ := Overview(0.04, 5, 2, 0.5)
	if math.Abs(s2/s1-2) > 1e-12 {
		t.Fatalf("斜率未按基础频率成比例: %.6f / %.6f", s2, s1)
	}
	if _, deg := Overview(0.04, 5, 2, 0.5); math.Abs(s2-math.Tan(Radians(deg))) > 1e-12 {
		t.Fatalf("倾角与斜率不满足 tan: %v vs %v", deg, s2)
	}
}

// TestOverviewMatchesMeasured 在无高频混叠的频段上，预判应与实测在 1 度内。
func TestOverviewMatchesMeasured(t *testing.T) {
	for _, c := range []struct {
		f0        float64
		oct       int
		lac, gain float64
	}{
		{0.012, 4, 2, 0.5}, // 最高档 0.096，远低于 Nyquist
		{0.01, 5, 1.7, 0.5},
	} {
		_, pred := Overview(c.f0, c.oct, c.lac, c.gain)
		_, st := FromField(384, field(7, 384, c.f0, c.oct, c.lac, c.gain))
		if d := pred - st.MeanTilt; math.Abs(d) > 1 {
			t.Fatalf("f0=%.3f pred=%.3f meas=%.3f 偏差 %.3f 超过 1 度", c.f0, pred, st.MeanTilt, d)
		}
	}
}

// TestHighFrequencyUnderpredicts 记录模型已知边界：超过 Nyquist 的高频档
// 在像素网格上被中心差分抹平，实测低于预判，偏差随最高档频率增大。
func TestHighFrequencyUnderpredicts(t *testing.T) {
	_, predLow := Overview(0.05, 6, 2, 0.62) // 最高档 1.6
	_, predHi := Overview(0.08, 6, 2, 0.62)  // 最高档 2.56
	_, low := FromField(384, field(7, 384, 0.05, 6, 2, 0.62))
	_, hi := FromField(384, field(7, 384, 0.08, 6, 2, 0.62))
	if predLow-low.MeanTilt <= 0 || predHi-hi.MeanTilt <= 0 {
		t.Fatalf("高频档应低估: dLow=%.3f dHi=%.3f", predLow-low.MeanTilt, predHi-hi.MeanTilt)
	}
	if predHi-hi.MeanTilt <= predLow-low.MeanTilt {
		t.Fatalf("偏差未随最高档频率增大: dLow=%.3f dHi=%.3f", predLow-low.MeanTilt, predHi-hi.MeanTilt)
	}
}

// TestFromFieldExcludesBorder 验证边界一圈为 NaN 且不计入统计。
func TestFromFieldExcludesBorder(t *testing.T) {
	const size = 32
	data := make([]float64, size*size)
	for i := range data {
		data[i] = float64(i%size) * 0.01
	}
	g, st := FromField(size, data)
	if st.Count != (size-2)*(size-2) {
		t.Fatalf("统计格数 %d，应为 %d", st.Count, (size-2)*(size-2))
	}
	for i, v := range g.Tilt {
		x, y := i%size, i/size
		border := x == 0 || y == 0 || x == size-1 || y == size-1
		if border != math.IsNaN(v) {
			t.Fatalf("格 (%d,%d) 边界=%v 但值=%v", x, y, border, v)
		}
	}
	// 东西向坡度 0.01/px，倾角应为 atan(0.01)。
	if d := st.MeanTilt - Degrees(math.Atan(0.01)); math.Abs(d) > 1e-9 {
		t.Fatalf("平均倾角 %.9f，应 %.9f", st.MeanTilt, Degrees(math.Atan(0.01)))
	}
}

// TestNormalizedTilt 验证归一化按极差整体缩放斜率。
func TestNormalizedTilt(t *testing.T) {
	slope, _ := Overview(0.02, 4, 2, 0.5)
	got := NormalizedTilt(slope, 2)
	want := Degrees(math.Atan(slope / 2))
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("归一化倾角 %.9f，应 %.9f", got, want)
	}
}

// TestEnsembleAgreesWithOverview 用 6 个种子集成平均，排除单图抽样摆幅后
// 检验低频参数的预判与实测一致。
func TestEnsembleAgreesWithOverview(t *testing.T) {
	_, pred := Overview(0.012, 4, 2, 0.5)
	var sum float64
	const n = 6
	for seed := uint64(1); seed <= n; seed++ {
		_, st := FromField(256, field(seed, 256, 0.012, 4, 2, 0.5))
		sum += st.MeanTilt
	}
	ens := sum / n
	if math.Abs(pred-ens) > 0.15 {
		t.Fatalf("集成均值 %.3f 与预判 %.3f 相差超过 0.15 度", ens, pred)
	}
}

// TestAtanBiasBounded 记录 atan(E) 与 E[atan] 的差随斜率增大而增大，
// 低频小斜率时应在 3% 以内。
func TestAtanBiasBounded(t *testing.T) {
	_, pred := Overview(0.012, 4, 2, 0.5)
	var sum float64
	const n = 6
	for seed := uint64(1); seed <= n; seed++ {
		_, st := FromField(256, field(seed, 256, 0.012, 4, 2, 0.5))
		sum += st.MeanTilt
	}
	ens := sum / n
	if bias := (pred - ens) / ens; bias < 0 || bias > 0.03 {
		t.Fatalf("低频工况 atan(E) 应略高于 E[atan] 且差距小于 3%%，实测 %.2f%%", bias*100)
	}
}
