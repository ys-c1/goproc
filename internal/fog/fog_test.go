package fog

import (
	"math"
	"testing"

	"goproc/internal/resample"
)

func TestMeasureIdentical(t *testing.T) {
	n := 24
	orig := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			orig[y*n+x] = math.Sin(float64(x)*0.5) * math.Cos(float64(y)*0.4)
		}
	}
	r, f := Measure(orig, orig, n, 7, 1)
	if math.Abs(r-1) > 1e-12 || math.Abs(f) > 1e-12 {
		t.Fatalf("同图对比应为 R=1,F=0，得到 R=%v F=%v", r, f)
	}
}

func TestMeasureFlatThumbnail(t *testing.T) {
	// 完全抹平的缩略图局部方差为零：R=0, F=1。
	n := 20
	orig := make([]float64, n*n)
	for i := range orig {
		orig[i] = float64(i%7) * 0.1
	}
	flat := make([]float64, n*n)
	r, f := Measure(orig, flat, n, 5, 0.5)
	if r != 0 || f != 1 {
		t.Fatalf("全雾应为 R=0,F=1，得到 R=%v F=%v", r, f)
	}
}

func TestCover1(t *testing.T) {
	cases := []struct {
		c, r, p, want float64
	}{
		{3, 2, 3, 1},      // 中心像素完整覆盖
		{3, 2, 5, 0.5},    // 右边界像素只覆盖一半
		{3, 2, 1, 0.5},    // 左边界像素只覆盖一半
		{3, 2, 6, 0},      // 窗外
		{2.5, 1, 3.5, 0.5},  // 窗口边界切过像素中心，覆盖半像素
		{2.5, 1, 3.6, 0.4},  // 像素左沿 3.1 在窗内、右沿在窗外
		{2.5, 1, 4.5, 0},    // 窗外
	}
	for _, c := range cases {
		if got := cover1(c.c, c.r, c.p); math.Abs(got-c.want) > 1e-12 {
			t.Fatalf("cover1(%v,%v,%v)=%v want %v", c.c, c.r, c.p, got, c.want)
		}
	}
}

func TestWeightedVarianceConstant(t *testing.T) {
	n := 10
	img := make([]float64, n*n)
	for i := range img {
		img[i] = 0.7
	}
	if got := WeightedLocalVariance(img, n, 5, 5, 2.5); math.Abs(got) > 1e-15 {
		t.Fatalf("常数场方差应为 0，得到 %v", got)
	}
}

// TestMeasureNoStaircase 验证连续窗设计消除了整数匹配窗的台阶伪迹。
// 台阶设计的特征：窗宽不变的平台内相邻档 R 几乎不动（差分趋零），
// 变化全部集中在窗宽跳变点。连续窗设计下每个细网格档都应有可测变化。
func TestMeasureNoStaircase(t *testing.T) {
	n := 128
	orig := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			orig[y*n+x] = math.Sin(float64(x)*0.35)*math.Cos(float64(y)*0.31) +
				0.5*math.Sin(float64(x+y)*0.8) +
				0.25*math.Sin(float64(x)*1.7)*math.Cos(float64(y)*1.3)
		}
	}
	const W = 11
	var rs []float64
	// 步长 0.02 的细网格，必然多次穿过旧设计的窗宽跳变点。
	for k := 0; k <= 37; k++ {
		s := 1 - float64(k)*0.02
		m := int(math.Round(float64(n) * s))
		actual := float64(m) / float64(n)
		thumb := resample.Downsample(orig, n, actual, resample.Box)
		up := resample.UpscaleNearest(thumb, m, n)
		r, _ := Measure(orig, up, n, W, actual)
		rs = append(rs, r)
	}
	if math.Abs(rs[0]-1) > 1e-12 {
		t.Fatalf("s=1 应 R=1，得到 %v", rs[0])
	}
	if rs[len(rs)-1] > 0.6 {
		t.Fatalf("s≈0.26 应有明显雾化，R=%v 过大", rs[len(rs)-1])
	}
	// 每一对相邻档都必须有可测的相对变化（平台会给出趋零差分）。
	zeroDiffs := 0
	for i := 1; i < len(rs); i++ {
		if math.Abs(rs[i]-rs[i-1])/rs[i-1] < 1e-4 {
			zeroDiffs++
		}
	}
	if zeroDiffs > 0 {
		t.Fatalf("发现 %d 个无变化平台档，连续窗设计不应有台阶", zeroDiffs)
	}
}
