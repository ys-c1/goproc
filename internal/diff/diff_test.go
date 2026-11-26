package diff

import (
	"math"
	"testing"
)

// 二阶中心差分对二次函数是精确的：f=x² 的 fx=2x, fxx=2。
func TestQuadraticExactInterior(t *testing.T) {
	n := 16
	v := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			v[y*n+x] = float64(x*x) + 0.5*float64(y*y)
		}
	}
	g := Grid{N: n, V: v}
	fx := D1X(g)
	fxx := D2X(g)
	fyy := D2Y(g)
	for y := 1; y < n-1; y++ {
		for x := 1; x < n-1; x++ {
			if got, want := fx[y*n+x], 2*float64(x); math.Abs(got-want) > 1e-12 {
				t.Fatalf("fx(%d,%d)=%v want %v", x, y, got, want)
			}
			if math.Abs(fxx[y*n+x]-2) > 1e-12 {
				t.Fatalf("fxx(%d,%d)=%v want 2", x, y, fxx[y*n+x])
			}
			if math.Abs(fyy[y*n+x]-1) > 1e-12 {
				t.Fatalf("fyy(%d,%d)=%v want 1", x, y, fyy[y*n+x])
			}
		}
	}
}

// 单侧模板对二次函数也应精确，这是边界“另算”的正确性来源。
func TestQuadraticExactBoundary(t *testing.T) {
	n := 12
	v := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			v[y*n+x] = 3*float64(x*x) + 7*float64(y*y) + 2*float64(x*y)
		}
	}
	g := Grid{N: n, V: v}
	fx := D1X(g)
	fy := D1Y(g)
	fxx := D2X(g)
	fyy := D2Y(g)
	fxy := DXY(g)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			i := y*n + x
			if want := 6*float64(x) + 2*float64(y); math.Abs(fx[i]-want) > 1e-10 {
				t.Fatalf("fx(%d,%d)=%v want %v", x, y, fx[i], want)
			}
			if want := 14*float64(y) + 2*float64(x); math.Abs(fy[i]-want) > 1e-10 {
				t.Fatalf("fy(%d,%d)=%v want %v", x, y, fy[i], want)
			}
			if math.Abs(fxx[i]-6) > 1e-10 {
				t.Fatalf("fxx(%d,%d)=%v want 6", x, y, fxx[i])
			}
			if math.Abs(fyy[i]-14) > 1e-10 {
				t.Fatalf("fyy(%d,%d)=%v want 14", x, y, fyy[i])
			}
			if math.Abs(fxy[i]-2) > 1e-10 {
				t.Fatalf("fxy(%d,%d)=%v want 2", x, y, fxy[i])
			}
		}
	}
}

func TestRejectsBadGrid(t *testing.T) {
	if err := Validate(Grid{N: 3, V: make([]float64, 9)}); err == nil {
		t.Fatal("N=3 应报错")
	}
	if err := Validate(Grid{N: 4, V: make([]float64, 15)}); err == nil {
		t.Fatal("长度不符应报错")
	}
}
