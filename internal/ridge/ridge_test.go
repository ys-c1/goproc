package ridge

import (
	"math"
	"testing"

	"goproc/api"
)

// 曲率在常数场上为零，线性场上也为零；二阶中心差分只对二次项有响应。
func TestCurvatureOnPolynomials(t *testing.T) {
	const n = 9
	cases := []struct {
		name string
		f    func(x, y int) float64
		want float64
	}{
		{"constant", func(x, y int) float64 { return 3 }, 0},
		{"linear", func(x, y int) float64 { return float64(2*x - 5*y + 1) }, 0},
		{"quadratic", func(x, y int) float64 { return float64((x-4)*(x-4) + (y-4)*(y-4)) }, 2},
	}
	for _, tc := range cases {
		data := make([]float64, n*n)
		for y := 0; y < n; y++ {
			for x := 0; x < n; x++ {
				data[y*n+x] = tc.f(x, y)
			}
		}
		c := CurvatureOf(n, data)
		for y := 0; y < n; y++ {
			for x := 1; x <= n-2; x++ {
				if got := c.At(x, y); math.Abs(got-tc.want) > 1e-12 {
					t.Fatalf("%s: κ(%d,%d)=%v, want %v", tc.name, x, y, got, tc.want)
				}
			}
		}
		if !math.IsNaN(c.At(0, 0)) || !math.IsNaN(c.At(n-1, 0)) {
			t.Fatal("最左最右列应为 NaN，不参与判定")
		}
	}
}

// 曲率极小当且仅当左右差分严格异号；判定列恰好是 [2,size-3]。
func TestDetectIsSignFlip(t *testing.T) {
	for _, n := range []int{32, 64} {
		h := mustGen(t, n)
		c := CurvatureOf(n, h.Data)
		m, _ := Detect(c)
		for y := 0; y < n; y++ {
			for x := 0; x < n; x++ {
				got := m.At(x, y)
				if x < 2 || x > n-3 {
					if got {
						t.Fatalf("n=%d 判定列越界: (%d,%d)", n, x, y)
					}
					continue
				}
				dL := c.At(x, y) - c.At(x-1, y)
				dR := c.At(x+1, y) - c.At(x, y)
				want := dL < 0 && dR > 0
				if got != want {
					t.Fatalf("n=%d (%d,%d): got %v want %v (dL=%v dR=%v)", n, x, y, got, want, dL, dR)
				}
			}
		}
	}
}

// 四次剖面 h=(x-5)^4：曲率 12(x-5)^2+2 在 x=5 取严格极小；
// 二次剖面曲率为常数，整行并列，会被记入 ties 而不计入脊线。
func TestDetectSyntheticProfile(t *testing.T) {
	const n = 11
	data := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			d := float64(x - 5)
			data[y*n+x] = d * d * d * d
		}
	}
	m, ties := Detect(CurvatureOf(n, data))
	if ties != 0 {
		t.Fatalf("四次剖面不应有并列差分，ties=%d", ties)
	}
	for y := 0; y < n; y++ {
		if !m.At(5, y) {
			t.Fatalf("第 %d 行未在 x=5 检出曲率极小", y)
		}
	}
	if got := m.Count(); got != n {
		t.Fatalf("应恰有每行一个极小，共 %d，实际 %d", n, got)
	}

	// 常数曲率：整行都是并列，没有严格极小。
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			d := float64(x - 5)
			data[y*n+x] = d * d
		}
	}
	m2, ties2 := Detect(CurvatureOf(n, data))
	if m2.Count() != 0 || ties2 == 0 {
		t.Fatalf("二次剖面应无严格极小且存在并列，got count=%d ties=%d", m2.Count(), ties2)
	}
}

// 相邻差分相关高斯模型下，异号概率的闭式解。
func TestModelDensityClosedForm(t *testing.T) {
	for _, tc := range []struct{ rho, want float64 }{
		{-1, 0.5}, {0, 0.25}, {1, 0}, {0.5, 0.25 - math.Asin(0.5)/(2*math.Pi)},
	} {
		if got := modelDensity(tc.rho); math.Abs(got-tc.want) > 1e-12 {
			t.Fatalf("ρ=%v: got %v want %v", tc.rho, got, tc.want)
		}
	}
	// 超界输入应被夹紧而非返回 NaN。
	if got := modelDensity(3); got != 0 {
		t.Fatalf("ρ=3 应夹紧为 1，got p=%v", got)
	}
}

// 独立同分布假设给出 1/3 密度，约为实测曲率脊的两倍：相邻曲率强正相关。
// 该测试把「朴素均匀假设偏差约一倍」钉成回归项。
func TestIIDModelOverestimates(t *testing.T) {
	const n = 128
	f := AnalyzeFrame(n, mustGen(t, n).Data)
	ratio := f.IIDCount / float64(f.KMin.Measured)
	if ratio < 1.5 || ratio > 2.5 {
		t.Fatalf("独立同分布预判/实测 = %.2f，预期约 1.9", ratio)
	}
}

// 实测与预判按画幅比对；默认参数下两个画幅都应在一成以内。
func TestPredictionWithinTenPercent(t *testing.T) {
	for _, n := range []int{128, 256} {
		f := AnalyzeFrame(n, mustGen(t, n).Data)
		if f.Ties != 0 {
			t.Fatalf("n=%d 出现并列差分 ties=%d", n, f.Ties)
		}
		if dev := f.KMin.Deviation(); math.Abs(dev) > 0.10 {
			t.Fatalf("曲率脊 n=%d 预判 %.1f 实测 %d 偏差 %.1f%%", n, f.KMin.Count, f.KMin.Measured, 100*dev)
		}
		if dev := f.Crest.Deviation(); math.Abs(dev) > 0.10 {
			t.Fatalf("分水脊 n=%d 预判 %.1f 实测 %d 偏差 %.1f%%", n, f.Crest.Count, f.Crest.Measured, 100*dev)
		}
	}
}

// 回归基准：默认参数、种子 1 的两个画幅。
func TestGoldenCounts(t *testing.T) {
	for _, tc := range []struct{ n, kmin, crest int }{{128, 2818, 785}, {256, 11348, 3080}} {
		f := AnalyzeFrame(tc.n, mustGen(t, tc.n).Data)
		if f.KMin.Measured != tc.kmin {
			t.Fatalf("n=%d 曲率脊实测 %d，基准 %d", tc.n, f.KMin.Measured, tc.kmin)
		}
		if f.Crest.Measured != tc.crest {
			t.Fatalf("n=%d 分水脊实测 %d，基准 %d", tc.n, f.Crest.Measured, tc.crest)
		}
	}
}

// 连通段划分覆盖全部脊线像素，标签与掩码一一对应。
func TestLabelPartitions(t *testing.T) {
	const n = 96
	f := AnalyzeFrame(n, mustGen(t, n).Data)
	sum := 0
	for _, c := range f.Comps {
		sum += c.Pixels
	}
	if sum != f.Mask.Count() {
		t.Fatalf("连通段像素合计 %d ≠ 掩码像素 %d", sum, f.Mask.Count())
	}
	for i := 0; i+1 < len(f.Comps); i++ {
		if f.Comps[i].Pixels < f.Comps[i+1].Pixels {
			t.Fatalf("连通段未按像素数降序：%d < %d", f.Comps[i].Pixels, f.Comps[i+1].Pixels)
		}
	}
	seen := map[int32]bool{}
	for i, v := range f.Labels {
		if (v != 0) != f.Mask.Data[i] {
			t.Fatalf("像素 %d 标签 %d 与掩码 %v 不符", i, v, f.Mask.Data[i])
		}
		seen[v] = true
	}
	for _, c := range f.Comps {
		if !seen[int32(c.ID)] {
			t.Fatalf("段 %d 无对应像素", c.ID)
		}
	}
}

// 水平平移一列后，内部判定应整体平移一列。
func TestDetectTranslationCovariant(t *testing.T) {
	const n = 64
	h := mustGen(t, n)
	m, _ := Detect(CurvatureOf(n, h.Data))
	shifted := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			shifted[y*n+x] = h.Data[y*n+(x-1+n)%n]
		}
	}
	m2, _ := Detect(CurvatureOf(n, shifted))
	for y := 0; y < n; y++ {
		for x := 3; x <= n-4; x++ {
			if m.At(x, y) != m2.At(x+1, y) {
				t.Fatalf("平移不一致: (%d,%d)", x, y)
			}
		}
	}
}

// 曲率脊与分水脊是两个不同的集合：重合率很低，不能互相替代。
func TestCurvatureRidgeIsNotCrest(t *testing.T) {
	const n = 128
	f := AnalyzeFrame(n, mustGen(t, n).Data)
	share := float64(f.Both) / float64(f.KMin.Measured)
	if share > 0.25 {
		t.Fatalf("曲率极小中有 %.0f%% 同时是高度极大，超出预期", 100*share)
	}
}

func mustGen(t *testing.T, n int) *api.Heightmap {
	t.Helper()
	opt := api.DefaultOptions()
	opt.Size = n
	opt.Seed = 1
	h, err := api.Generate(opt)
	if err != nil {
		t.Fatal(err)
	}
	return h
}

// 直线段：n 个像素的直链，长度应为 n-1。
func TestTotalLengthStraight(t *testing.T) {
	const n = 10
	m := Mask{Size: n, Data: make([]bool, n*n)}
	for x := 0; x < 7; x++ {
		m.Data[3*n+x] = true
	}
	got, edges := TotalLength(m)
	if got != 6 || edges != 6 {
		t.Fatalf("直链长度 %v（%d 边），应为 6", got, edges)
	}
}

// 对角直链：每步 √2。
func TestTotalLengthDiagonal(t *testing.T) {
	const n = 12
	m := Mask{Size: n, Data: make([]bool, n*n)}
	for i := 0; i < 5; i++ {
		m.Data[i*n+i] = true
	}
	got, _ := TotalLength(m)
	if math.Abs(got-4*math.Sqrt2) > 1e-12 {
		t.Fatalf("对角链长度 %v，应为 4√2", got)
	}
}

// 单像素段长度为 0；不相连的两段按各自长度相加，不跨越空隙。
func TestTotalLengthSeparate(t *testing.T) {
	const n = 20
	m := Mask{Size: n, Data: make([]bool, n*n)}
	for x := 0; x < 3; x++ {
		m.Data[2*n+x] = true
	}
	for x := 10; x < 14; x++ {
		m.Data[15*n+x] = true
	}
	got, _ := TotalLength(m)
	if got != 2+3 {
		t.Fatalf("两段长度 %v，应为 5", got)
	}
}

// 总长与像素数之比随画幅基本稳定，且明显小于像素数（斜向或稀疏段会拉低比值）。
func TestTotalLengthStable(t *testing.T) {
	var ratios []float64
	for _, n := range []int{128, 256} {
		f := AnalyzeFrame(n, mustGen(t, n).Data)
		l, _ := TotalLength(f.Mask)
		ratios = append(ratios, l/float64(f.KMin.Measured))
	}
	if ratios[0] <= 0 || math.Abs(ratios[0]-ratios[1]) > 0.1 {
		t.Fatalf("两个画幅的总长/像素比不稳定: %v", ratios)
	}
}

// 参与预判的差分对数必须与候选像素数严格一致，否则预判的分母会缺一行。
func TestPairCountMatchesCandidates(t *testing.T) {
	for _, n := range []int{128, 256} {
		f := AnalyzeFrame(n, mustGen(t, n).Data)
		if f.KMin.Pairs.Count != f.Candidates {
			t.Fatalf("n=%d 曲率差分对 %d ≠ 候选像素 %d", n, f.KMin.Pairs.Count, f.Candidates)
		}
		if f.Crest.Pairs.Count != f.Candidates {
			t.Fatalf("n=%d 高度差分对 %d ≠ 候选像素 %d", n, f.Crest.Pairs.Count, f.Candidates)
		}
	}
}

// 曲线脊的实测与预判在多个种子、多个画幅下都要稳定在一成以内。
func TestPredictionRobustAcrossSeeds(t *testing.T) {
	for _, n := range []int{128, 256} {
		for _, seed := range []uint64{1, 7, 13, 42} {
			opt := api.DefaultOptions()
			opt.Size, opt.Seed = n, seed
			h, err := api.Generate(opt)
			if err != nil {
				t.Fatal(err)
			}
			f := AnalyzeFrame(n, h.Data)
			if dev := f.KMin.Deviation(); math.Abs(dev) > 0.10 {
				t.Fatalf("n=%d seed=%d 曲率脊偏差 %.1f%%", n, seed, 100*dev)
			}
			if dev := f.Crest.Deviation(); math.Abs(dev) > 0.10 {
				t.Fatalf("n=%d seed=%d 分水脊偏差 %.1f%%", n, seed, 100*dev)
			}
			if f.Ties != 0 {
				t.Fatalf("n=%d seed=%d 出现并列 %d", n, seed, f.Ties)
			}
		}
	}
}
