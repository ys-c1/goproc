package slope

import (
	"math"
	"sort"
	"testing"

	"goproc/api"
)

// D(f) 在 f→0 时必须趋于 f²·∫φ'² = f²·20/7（命题 1 的解析梯度方差）。
func TestFDKernelEnergySmallF(t *testing.T) {
	f := 1e-3
	got := FDKernelEnergy(f)
	want := f * f * 20.0 / 7.0
	if math.Abs(got-want)/want > 0.02 {
		t.Fatalf("D(%v)=%v, 解析值 %v", f, got, want)
	}
}

// 数值积分 ∫φ² 必须等于解析值 181/231。
func TestPhi2(t *testing.T) {
	n := 100000
	h := 2.0 / float64(n)
	sum := 0.0
	for i := 0; i <= n; i++ {
		u := -1.0 + float64(i)*h
		w := 1.0
		if i == 0 || i == n {
			w = 0.5
		}
		v := phi(u)
		sum += w * v * v
	}
	got := sum * h
	if math.Abs(got-Phi2)/Phi2 > 1e-6 {
		t.Fatalf("∫φ²=%v, 解析值 %v", got, Phi2)
	}
}

// Generate 的归一化输出必须与 api.Generate 完全一致。
func TestGenerateMatchesAPI(t *testing.T) {
	for _, tc := range []struct {
		seed uint64
		k    int
	}{{1, 1}, {7, 4}, {13, 8}} {
		opt := api.Options{Size: 64, Octaves: tc.k, Lacunarity: 2, Gain: 0.5, Seed: tc.seed, Frequency: 0.02}
		h, err := api.Generate(opt)
		if err != nil {
			t.Fatal(err)
		}
		data, rng := slope_GenerateForTest(opt)
		if rng <= 0 {
			t.Fatal("极差非正")
		}
		for i := range data {
			if data[i] != h.Data[i] {
				t.Fatalf("seed=%d K=%d 像素 %d 不一致: %v != %v", tc.seed, tc.k, i, data[i], h.Data[i])
			}
		}
	}
}

func slope_GenerateForTest(opt api.Options) ([]float64, float64) { return Generate(opt) }

// 单层梯度标准差的实测值必须落在核积分预判的 3% 以内（命题 1）。
func TestSingleLayerSigma(t *testing.T) {
	opt := api.Options{Size: 512, Octaves: 1, Lacunarity: 2, Gain: 0.5, Seed: 1, Frequency: 0.02}
	data, rng := Generate(opt)
	dx, dy := SlopesXY(data, opt.Size)
	var v float64
	for i := range dx {
		v += dx[i]*dx[i] + dy[i]*dy[i]
	}
	meas := math.Sqrt(v / float64(2*len(dx)))
	pred := SigmaFD(1, DefaultParams()) / rng // 用实测极差剔除归一化随机性
	if d := math.Abs(pred-meas) / meas; d > 0.03 {
		t.Fatalf("单层梯度标准差: 预判 %v, 实测 %v, 偏差 %.2f%%", pred, meas, d*100)
	}
}

// 分位与超越概率互为逆运算。
func TestRayleighQuantileExceed(t *testing.T) {
	s, r := 0.012, 262.144
	g := RayleighQuantile(s, r)
	if p := ExceedProb(s, g); math.Abs(p-1/r)*r > 1e-9 {
		t.Fatalf("ExceedProb(s, Q(r))=%v, 期望 1/r=%v", p, 1/r)
	}
}

func TestQuantile(t *testing.T) {
	v := []float64{1, 2, 3, 4}
	if q := Quantile(v, 0.5); q != 2.5 {
		t.Fatalf("中位数 %v", q)
	}
	if q := Quantile(v, 1); q != 4 {
		t.Fatalf("最大值 %v", q)
	}
}

// 端到端：K=4 时 R=N/1000 分位的预判与实测偏差必须在两成以内（命题 2、3）。
func TestPredictionVsMeasuredK4(t *testing.T) {
	size := 256
	p := DefaultParams()
	nEff := NEff(size, p.F0)
	s := PredS(4, p, nEff)
	pred := RayleighQuantile(s, float64(size*size)/1000)
	var qs []float64
	for seed := 1; seed <= 4; seed++ {
		opt := api.Options{Size: size, Octaves: 4, Lacunarity: 2, Gain: 0.5, Seed: uint64(seed), Frequency: 0.02}
		data, _ := Generate(opt)
		sl := Slopes(data, size)
		sort.Float64s(sl)
		qs = append(qs, Quantile(sl, 1-1000.0/float64(size*size)))
	}
	var meas float64
	for _, q := range qs {
		meas += q
	}
	meas /= float64(len(qs))
	if d := math.Abs(pred-meas) / meas; d > 0.20 {
		t.Fatalf("K=4 分位: 预判 %v, 实测 %v, 偏差 %.1f%%", pred, meas, d*100)
	}
}
