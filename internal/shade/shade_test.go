package shade

import (
	"math"
	"testing"
)

// TestRenderFlatPlane 平面法线为 (0,0,1)，双灯结果只取决于仰角。
func TestRenderFlatPlane(t *testing.T) {
	size := 3
	h := make([]float64, size*size) // 全零平面
	elev := 20.0 * math.Pi / 180
	l1 := Light{Azimuth: 37.5 * math.Pi / 180, Elev: elev}
	l2 := Light{Azimuth: 52.5 * math.Pi / 180, Elev: elev}
	out := Render(size, h, l1, l2)
	want := math.Sin(elev)
	for i, v := range out {
		if math.Abs(v-want) > 1e-12 {
			t.Fatalf("平面像素 %d 照度 %v, 期望 %v", i, v, want)
		}
	}
}

// TestRenderSlopedPlane 用线性斜面 h=a*x+b*y 核对解析值，
// 并要求交换两灯结果不变（回归 nz*bz 漏乘法线的缺陷）。
func TestRenderSlopedPlane(t *testing.T) {
	size := 3
	a, b := 0.08, 0.05
	h := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			h[y*size+x] = a*float64(x) + b*float64(y)
		}
	}
	elev := 20.0 * math.Pi / 180
	l1 := Light{Azimuth: 37.5 * math.Pi / 180, Elev: elev}
	l2 := Light{Azimuth: 52.5 * math.Pi / 180, Elev: elev}

	out := Render(size, h, l1, l2)
	outSwap := Render(size, h, l2, l1)
	for i := range out {
		if math.Abs(out[i]-outSwap[i]) > 1e-12 {
			t.Fatalf("像素 %d 交换两灯结果改变: %v vs %v", i, out[i], outSwap[i])
		}
	}

	// 中心像素 (1,1) 用中心差分：gx=2a, gy=2b。
	gx, gy := 2*a, 2*b
	m := math.Sqrt(1 + gx*gx + gy*gy)
	nx, ny, nz := -gx/m, -gy/m, 1/m
	dot := func(l Light) float64 {
		lx, ly, lz := math.Cos(l.Elev)*math.Sin(l.Azimuth),
			math.Cos(l.Elev)*math.Cos(l.Azimuth), math.Sin(l.Elev)
		return math.Max(0, nx*lx+ny*ly+nz*lz)
	}
	want := (dot(l1) + dot(l2)) / 2
	if got := out[1*size+1]; math.Abs(got-want) > 1e-12 {
		t.Fatalf("斜面中心照度 %v, 解析期望 %v", got, want)
	}
}
