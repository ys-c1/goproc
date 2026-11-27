package calibration

import (
	"math"
	"testing"

	"goproc/internal/resample"
)

func TestCalScales(t *testing.T) {
	got := CalScales()
	want := []float64{1, 1 / math.Sqrt2, 0.5, 1 / (2 * math.Sqrt2), 0.25}
	if len(got) != 5 {
		t.Fatalf("标定点数 %d", len(got))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Fatalf("标定点 %d: got %v want %v", i, got[i], want[i])
		}
	}
}

func TestPowerLawFit(t *testing.T) {
	// 用已知幂律 R=0.8*s^1.3 造点，紧凑模型应还原参数。
	pts := make([]Point, 5)
	for i, s := range CalScales() {
		r := 0.8 * math.Pow(s, 1.3)
		pts[i] = Point{Scale: s, R: r, F: 1 - r}
	}
	a, g, r2 := Fit(pts, false)
	if math.Abs(a-0.8) > 1e-10 || math.Abs(g-1.3) > 1e-10 {
		t.Fatalf("自由拟合失败: A=%v gamma=%v", a, g)
	}
	if r2 < 1-1e-20 {
		t.Fatalf("完美幂律 R² 应为 1，得到 %v", r2)
	}
	c := Curve{A: a, Gamma: g, Points: pts}
	if got := c.PowerLawR(0.37); math.Abs(got-0.8*math.Pow(0.37, 1.3)) > 1e-12 {
		t.Fatal("PowerLawR 计算错误")
	}
}

func TestPiecewisePassesThroughPointsAndInverts(t *testing.T) {
	// 非幂律数据：主模型必须精确过实测点，且反解还原。
	pts := []Point{
		{Scale: 1, R: 1},
		{Scale: 0.7, R: 0.6},
		{Scale: 0.5, R: 0.35},
		{Scale: 0.35, R: 0.15},
		{Scale: 0.25, R: 0.08},
	}
	for i := range pts {
		pts[i].F = 1 - pts[i].R
	}
	c := Curve{Points: pts}
	for _, p := range pts {
		if got := c.PredictR(p.Scale); math.Abs(got-p.R) > 1e-12 {
			t.Fatalf("s=%v: 主模型未过实测点 got %v want %v", p.Scale, got, p.R)
		}
		if back := c.InvertScale(p.R); math.Abs(back-p.Scale) > 1e-10 {
			t.Fatalf("R=%v 反解失败: got %v want %v", p.R, back, p.Scale)
		}
	}
	// 区间内中间点：单调且落在两端点之间。
	mid := c.PredictR(0.6)
	if !(mid < 0.6 && mid > 0.35) {
		t.Fatalf("插值点越界: %v", mid)
	}
	// 外推不崩溃且保持正、单调。
	e := c.PredictR(0.15)
	if e <= 0 || e >= 0.08 {
		t.Fatalf("外推异常: %v", e)
	}
	// F 反解与 R 反解一致。
	if got := c.InvertScaleF(0.4); math.Abs(got-c.InvertScale(0.6)) > 1e-12 {
		t.Fatal("按 F 反解与按 R 反解不一致")
	}
}

func TestCalibrationOnRamp(t *testing.T) {
	// 确定性斜坡图上做端到端标定：R 应随 s 单调，gamma 为正。
	n := 128
	orig := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			orig[y*n+x] = math.Mod(float64(x+y)*0.1, 1)
		}
	}
	for _, k := range []resample.Kind{resample.Box, resample.Triangle} {
		c := Calibrate(orig, n, k, 9)
		if c.Gamma <= 0 {
			t.Fatalf("%s: gamma 应为正，得到 %v", k, c.Gamma)
		}
		if math.Abs(c.Points[0].F) > 1e-12 {
			t.Fatalf("%s: s=1 出现雾化 F=%v", k, c.Points[0].F)
		}
		for i := 1; i < len(c.Points); i++ {
			if c.Points[i].R > c.Points[i-1].R+1e-12 {
				t.Fatalf("%s: R 非单调", k)
			}
		}
		// 主预判曲线在实测点上零误差。
		for _, p := range c.Points {
			if math.Abs(c.PredictR(p.Scale)-p.R) > 1e-12 {
				t.Fatalf("%s: 主模型未过实测点", k)
			}
		}
	}
}

func TestValidateScalesDistinct(t *testing.T) {
	cal := map[float64]bool{}
	for _, s := range CalScales() {
		cal[math.Round(s*1000)] = true
	}
	for _, s := range ValScales() {
		if cal[math.Round(s*1000)] {
			t.Fatalf("验证档 %v 与标定档重合", s)
		}
	}
}
