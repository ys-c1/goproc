package hmap

import (
	"math"
	"testing"

	"goproc/internal/fbm"
	"goproc/internal/noise"
)

func approx(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// TestGradientForwardDiff 用解析平面 h=ax+by 验证前向差分：内部像素精确成立。
func TestGradientForwardDiff(t *testing.T) {
	const size = 8
	const a, b = 0.35, -0.21
	h := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			h[y*size+x] = a*float64(x) + b*float64(y)
		}
	}
	gx, gy := Gradient(size, h, 1)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			i := y*size + x
			// 边缘退回后向差分，平面上结果仍等于 a、b。
			if !approx(gx[i], a, 1e-15) || !approx(gy[i], b, 1e-15) {
				t.Fatalf("(%d,%d) 梯度错: gx=%v gy=%v 期望 %v %v", x, y, gx[i], gy[i], a, b)
			}
		}
	}
	// 显式格距：D = Δh/dx，平面上应为 a/dx。
	gx2, _ := Gradient(size, h, 2)
	if !approx(gx2[0], a/2, 1e-15) {
		t.Fatalf("格距缩放错: %v", gx2[0])
	}
}

// TestShadeIsDotProduct 直接验证亮度等于法向量与光向单位向量的点积。
func TestShadeIsDotProduct(t *testing.T) {
	const size = 6
	const a, b = 0.4, -0.15
	h := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			h[y*size+x] = a*float64(x) + b*float64(y)
		}
	}
	gx, gy := Gradient(size, h, 1)
	for _, tc := range []struct {
		name    string
		az, alt float64
	}{
		{"左上前", math.Atan2(1, 1), math.Pi / 4}, // 方位 45°（左上），仰角 45°
		{"正上", 0, math.Pi / 2},
		{"右上前", math.Atan2(1, -1), math.Pi / 4}, // 方位 135°（右上），仰角 45°
	} {
		l := Light{Azimuth: tc.az, Altitude: tc.alt}
		got := ShadeLinear(gx, gy, l)
		lx, ly, lz := l.Vector()
		for i := range got {
			denom := math.Sqrt(1 + gx[i]*gx[i] + gy[i]*gy[i])
			want := lx*(-gx[i]/denom) + ly*(-gy[i]/denom) + lz*(1/denom)
			if !approx(got[i], want, 1e-14) {
				t.Fatalf("%s 点积不成立: %v != %v", tc.name, got[i], want)
			}
		}
		if n := math.Sqrt(lx*lx + ly*ly + lz*lz); !approx(n, 1, 1e-15) {
			t.Fatalf("光向未归一化: %v", n)
		}
	}
}

// TestSkewAffineInvariance 偏度在仿射变换 aX+b（a>0）下不变，
// 这是“预测偏度只依赖梯度投影分布形状”的代数依据。
func TestSkewAffineInvariance(t *testing.T) {
	v := []float64{-1.2, -0.4, 0.1, 0.7, 1.1, 2.3, -0.9}
	base := Stats(v).Skew
	shifted := make([]float64, len(v))
	for i, x := range v {
		shifted[i] = 2.5*x + 3.7
	}
	if !approx(base, Stats(shifted).Skew, 1e-12) {
		t.Fatalf("偏度不满足仿射不变: %v vs %v", base, Stats(shifted).Skew)
	}
}

// TestResolutionEquivalence 核心等价性证明。
// 配置 A (N=64, f=0.02, dx=1) 与配置 B (N=128, f'=0.01, dx'=2)
// 满足 f·dx = f'·dx' = 0.02（q 不变）。
//
// 无量纲坐标 u=fx、v=fy 只依赖 q，故在匹配索引 (i,j)↔(2i,2j) 上高度场逐点相同。
// A 的单胞元前向差分，与 B 跨两个像素的粗胞元差分作用在同一段物理距离 dx'=2dx 上，
// 因此无量纲差分 Δ（也就是单位像素光照所用的梯度）逐点相同：
//
//	ΔxA(i,j) = h(i+1,j)-h(i,j) = hB(2i+2,2j)-hB(2i,2j)
//
// 显式格距商 D=Δ/dx 则满足 D/f 不变；同一光向下点积光照逐像素相同。
func TestResolutionEquivalence(t *testing.T) {
	src := noise.New(7)
	const (
		na        = 64
		nb        = 128
		fa, dxA   = 0.02, 1.0
		fb, dxB   = 0.01, 2.0
		oct       = 4
		lac, gain = 2.0, 0.5
	)
	if fa*dxA != fb*dxB {
		t.Fatalf("前置条件不成立: %v != %v", fa*dxA, fb*dxB)
	}
	ha := make([]float64, na*na)
	hb := make([]float64, nb*nb)
	for y := 0; y < na; y++ {
		for x := 0; x < na; x++ {
			ha[y*na+x] = fbm.Eval(src, float64(x)*fa, float64(y)*fa, oct, lac, gain)
		}
	}
	for y := 0; y < nb; y++ {
		for x := 0; x < nb; x++ {
			hb[y*nb+x] = fbm.Eval(src, float64(x)*fb, float64(y)*fb, oct, lac, gain)
		}
	}
	// A 的单位像素梯度（存档光照图所用形式）与显式格距商。
	gxa, gya := Gradient(na, ha, 1)
	dxa, dya := Gradient(na, ha, dxA)
	l := Light{Azimuth: math.Pi / 4, Altitude: math.Pi / 4}
	sa := ShadeLinear(gxa, gya, l)
	for j := 0; j < na-1; j++ {
		for i := 0; i < na-1; i++ {
			ia := j*na + i
			ib := (2*j)*nb + 2*i
			// 1. 高度场在匹配点相同。
			if !approx(ha[ia], hb[ib], 1e-15) {
				t.Fatalf("高度场匹配点不等: %v vs %v", ha[ia], hb[ib])
			}
			// 2. B 的粗胞元（2 像素）前向差分 == A 的单位像素前向差分。
			cx := hb[ib+2] - hb[ib]
			cy := hb[ib+2*nb] - hb[ib]
			if !approx(gxa[ia], cx, 1e-15) || !approx(gya[ia], cy, 1e-15) {
				t.Fatalf("粗胞元差分不等 @ (%d,%d): ΔA=(%v,%v) ΔB=(%v,%v)",
					i, j, gxa[ia], gya[ia], cx, cy)
			}
			// 3. 显式格距商满足 D/f 不变：D_A/f_A == D_B/f_B，
			//    其中 B 的粗胞元商为 ΔB/dx'。
			dBx, dBy := cx/dxB, cy/dxB
			if !approx(dxa[ia]/fa, dBx/fb, 1e-15) || !approx(dya[ia]/fa, dBy/fb, 1e-15) {
				t.Fatalf("D/f 不变量不成立 @ (%d,%d)", i, j)
			}
			// 4. 用同一（单位像素）梯度算子时，点积光照逐像素相同。
			sb := ShadeLinear([]float64{cx}, []float64{cy}, l)[0]
			if !approx(sa[ia], sb, 1e-14) {
				t.Fatalf("光照不等 @ (%d,%d): %v vs %v", i, j, sa[ia], sb)
			}
		}
	}
}

// TestShadeClamp 完整朗伯模型截断负值并给出占比。
func TestShadeClamp(t *testing.T) {
	gx := []float64{5, -5}
	gy := []float64{0, 0}
	l := Light{Azimuth: 0, Altitude: math.Pi / 4} // 光从右上来，gx=5 的坡面背光
	b, zeroFrac := Shade(gx, gy, l)
	for _, v := range b {
		if v < 0 || v > 1 {
			t.Fatalf("亮度越界: %v", v)
		}
	}
	if !approx(zeroFrac, 0.5, 1e-15) || b[0] != 0 {
		t.Fatalf("截断错误: b=%v frac=%v", b, zeroFrac)
	}
}
