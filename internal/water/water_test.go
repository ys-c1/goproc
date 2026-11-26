package water

import (
	"math"
	"testing"
)

func TestNormalFieldUnitLength(t *testing.T) {
	size := 16
	h := make([]float64, size*size)
	for i := range h {
		h[i] = math.Sin(float64(i) * 0.3)
	}
	nf := NewNormalField(size, h, 0.15)
	for i, n := range nf.N {
		l := math.Sqrt(n.Dot(n))
		if math.Abs(l-1) > 1e-12 {
			t.Fatalf("像素 %d 法线非单位长度: %v", i, l)
		}
	}
}

func TestFlatNormalFieldPointsUp(t *testing.T) {
	size := 4
	h := make([]float64, size*size)
	nf := NewNormalField(size, h, 1)
	for i, n := range nf.N {
		if math.Abs(n.Z-1) > 1e-12 || math.Abs(n.X) > 1e-12 || math.Abs(n.Y) > 1e-12 {
			t.Fatalf("平面法线应朝天: 像素 %d = %+v", i, n)
		}
	}
}

func TestWhiteIsTopGrayOnly(t *testing.T) {
	if !IsWhite(1.0) {
		t.Fatal("强度 1.0 必须是白像素")
	}
	if !IsWhite(WhiteLevel) {
		t.Fatal("恰在顶格阈值上应判白")
	}
	if IsWhite(WhiteLevel - 1e-6) {
		t.Fatal("低于顶格阈值不应判白")
	}
	if Gray(1.0) != 255 || Gray(WhiteLevel) != 255 {
		t.Fatal("顶格强度灰度应为 255")
	}
	if Gray(WhiteLevel-1e-6) != 254 {
		t.Fatalf("阈值下一档灰度应为 254，实际 %d", Gray(WhiteLevel-1e-6))
	}
}

func TestRenderDeterministicAndMonotonic(t *testing.T) {
	size := 32
	h := make([]float64, size*size)
	for i := range h {
		h[i] = 0.5 + 0.4*math.Sin(float64(i%size)*0.4)*math.Cos(float64(i/size)*0.3)
	}
	sh := DefaultShader()
	nf := NewNormalField(size, h, sh.GridStep)
	a := sh.Render(nf, 0.5)
	b := sh.Render(nf, 0.5)
	if a.WhiteCount != b.WhiteCount {
		t.Fatal("同角度两次渲染结果不一致")
	}
	for i := range a.Intensity {
		if a.Intensity[i] != b.Intensity[i] {
			t.Fatal("同角度两次渲染强度不一致")
		}
	}
	// 反射系数随夹角单调递减。
	prev := math.Inf(1)
	for g := 0.0; g <= math.Pi/2; g += 0.05 {
		r := sh.ReflectanceAt(math.Cos(g))
		if r > prev+1e-12 {
			t.Fatalf("反射系数非单调: gamma=%v rho=%v prev=%v", g, r, prev)
		}
		prev = r
	}
}

func TestMeasureThreshold(t *testing.T) {
	// 构造一个法线朝天的平面，60 度光照时半向量也朝天，必然顶格。
	size := 4
	h := make([]float64, size*size)
	sh := DefaultShader()
	nf := NewNormalField(size, h, sh.GridStep)
	p, err := sh.MeasureThreshold(nf, math.Pi/3)
	if err != nil {
		t.Fatal(err)
	}
	if p.WhiteCount != size*size {
		t.Fatalf("平面在对齐角度应全白，实际 %d", p.WhiteCount)
	}
	if p.Threshold < WhiteLevel {
		t.Fatalf("阈值不应低于白像素判定线: %v", p.Threshold)
	}
	// 低角度无白像素时应返回 ErrEmptyWhite。
	if _, err := sh.MeasureThreshold(nf, 0.05); err != ErrEmptyWhite {
		t.Fatalf("无白像素应返回 ErrEmptyWhite，实际 %v", err)
	}
}
