package stretch

import (
	"math"
	"testing"
)

// TestIdentity 轴比为零时必须恒等。
func TestIdentity(t *testing.T) {
	w := Warp{Theta: 0.7, Ratio: 0}
	for _, p := range [][2]float64{{1.5, -2.0}, {0, 0}, {3.1, 2.2}} {
		u, v := w.Apply(p[0], p[1])
		if u != p[0] || v != p[1] {
			t.Fatalf("零轴比非恒等: (%v,%v)->(%v,%v)", p[0], p[1], u, v)
		}
	}
}

// TestAgainstExplicitMatrix 用显式矩阵 M=R(θ)M_rR(-θ) 逐元素核对。
func TestAgainstExplicitMatrix(t *testing.T) {
	const tol = 1e-12
	cases := []Warp{
		{Theta: 0, Ratio: 2},
		{Theta: math.Pi / 6, Ratio: 1.5},
		{Theta: math.Pi / 2, Ratio: 0.5},
		{Theta: 2.3, Ratio: 3},
	}
	pts := [][2]float64{{1.2, -0.7}, {-3, 4}, {0.4, 0.4}}
	mmul := func(a, b [2][2]float64) [2][2]float64 {
		var o [2][2]float64
		for i := 0; i < 2; i++ {
			for j := 0; j < 2; j++ {
				o[i][j] = a[i][0]*b[0][j] + a[i][1]*b[1][j]
			}
		}
		return o
	}
	for _, w := range cases {
		c, s := math.Cos(w.Theta), math.Sin(w.Theta)
		r := w.Ratio
		rot := [2][2]float64{{c, -s}, {s, c}}
		rotInv := [2][2]float64{{c, s}, {-s, c}}
		mr := [2][2]float64{{1, r}, {0, 1}}
		m := mmul(mmul(rot, mr), rotInv) // M=R(θ)M_rR(-θ)
		for _, p := range pts {
			wantX := m[0][0]*p[0] + m[0][1]*p[1]
			wantY := m[1][0]*p[0] + m[1][1]*p[1]
			gotX, gotY := w.Apply(p[0], p[1])
			if math.Abs(gotX-wantX) > tol || math.Abs(gotY-wantY) > tol {
				t.Fatalf("theta=%v r=%v p=%v: got (%v,%v) want (%v,%v)",
					w.Theta, r, p, gotX, gotY, wantX, wantY)
			}
		}
	}
}

// TestRidgeAngleFormula 预测式 θ_ridge = θ + atan(r) 的矩阵级证据：
// 局部系中各向同性噪声梯度沿 e1=(1,0)，经 M_r^T 变换后为 (1,r)，
// 其方位角恰为 atan(r)；整体旋转 θ 后方向角为 θ+atan(r)。
func TestRidgeAngleFormula(t *testing.T) {
	const tol = 1e-12
	for r := 0.25; r <= 4; r += 0.25 {
		got := math.Atan2(r, 1) // M_r^T e1 = (1, r) 的方位角
		want := math.Atan(r)
		if math.Abs(got-want) > tol {
			t.Fatalf("r=%v 偏移角 %v != atan(r) %v", r, got, want)
		}
	}
	theta := 35 * math.Pi / 180
	r := 2.0
	pred := math.Mod(theta+math.Atan(r), math.Pi)
	if pred < 0 {
		pred += math.Pi
	}
	c, s := math.Cos(theta), math.Sin(theta)
	gx, gy := c*1-s*r, s*1+c*r // R(θ)(1,r)
	direct := math.Mod(math.Atan2(gy, gx), math.Pi)
	if direct < 0 {
		direct += math.Pi
	}
	if math.Abs(pred-direct) > 1e-12 {
		t.Fatalf("旋转后预测角 %v 与直接计算 %v 不符", pred, direct)
	}
}
