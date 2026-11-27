package orient

import (
	"math"
	"testing"
)

const tol = 1e-9

func approx(t *testing.T, name string, got, want, eps float64) {
	t.Helper()
	if math.Abs(got-want) > eps {
		t.Errorf("%s = %.6f, want %.6f", name, got, want)
	}
}

// 斜坡场 u = gx x + gy y 的结构张量在任意窗口内都等于 ggᵀ，
// 主轴方向就是梯度方向，可解析核对 ½atan2 公式。
func ramp(size int, gx, gy float64) []float64 {
	d := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			d[y*size+x] = gx*float64(x) + gy*float64(y)
		}
	}
	return d
}

func TestPrincipalAngleRamp(t *testing.T) {
	const size = 16
	cases := []struct {
		name    string
		gx, gy  float64
		wantDeg float64
	}{
		{"horizontal gradient", 1, 0, 0},
		{"vertical gradient", 0, 1, 90},
		{"45 degree", 1, 1, 45},
		{"-45 degree", 1, -1, -45},
		{"135 axis wraps to -45", -1, 1, -45},
	}
	for _, c := range cases {
		d := ramp(size, c.gx, c.gy)
		tn, n := StructureTensor(size, d, 1, 1, size-1, size-1, Central)
		if n == 0 {
			t.Fatalf("%s: no valid pixels", c.name)
		}
		theta := PrincipalAngle(tn)
		approx(t, c.name, theta, c.wantDeg, 1e-8)
	}
}

func TestWrap180(t *testing.T) {
	cases := map[float64]float64{
		0: 0, 90: 90, -90: 90, 180: 0, 270: 90,
		135: -45, -135: 45, 400: 40, -400: -40,
	}
	for in, want := range cases {
		approx(t, "wrap", Wrap180(in), want, tol)
	}
}

func TestRotationAngleAndDifference(t *testing.T) {
	approx(t, "rotation 30->70", RotationAngle(70, 30), 40, tol)
	// 跨过 ±90° 回卷：thetaA=-80, thetaB=80，最短无向差为 20。
	approx(t, "wrapped difference", AngularDifference(-80, 80), 20, tol)
	approx(t, "rotation across seam", RotationAngle(-80, 80), 20, tol)
}

// 旋转斜坡场后重新估计主方向，角度应增加 phi（模 180°），验证旋转方向。
func TestRotateFieldRamp(t *testing.T) {
	const size = 32
	const phi = 35.0
	d := ramp(size, 1, 0) // 主方向 0°
	rot := RotateField(size, d, phi)
	tn, _ := StructureTensor(size, rot, 2, 2, size-2, size-2, Central)
	theta := PrincipalAngle(tn)
	// 双线性重采样引入的离散化误差为像素步长量级，容差取 0.1°。
	approx(t, "rotated ramp angle", theta, Wrap180(phi), 0.1)
}

// 对齐恒等式：对 B 施加 RotationAngle(thetaA,thetaB) 后，主方向差解析上为零。
func TestAlgebraicAlignment(t *testing.T) {
	for _, pair := range [][2]float64{{12, 78}, {-80, 80}, {45, -60}, {0, 90}, {170, -170}} {
		phi := RotationAngle(pair[0], pair[1])
		got := Wrap180(Wrap180(pair[1]+phi) - pair[0])
		approx(t, "algebraic residual", got, 0, 1e-12)
	}
}

// RotateTensor 必须保持特征值不变，且主轴角增加 phi（模 180°）。
func TestRotateTensor(t *testing.T) {
	tn := Tensor{Jxx: 3, Jyy: 1, Jxy: 1}
	tr := tn.Jxx + tn.Jyy
	rr := math.Hypot(tn.Jxx-tn.Jyy, 2*tn.Jxy)
	lam1, lam2 := (tr+rr)/2, (tr-rr)/2
	theta0 := PrincipalAngle(tn)
	for _, phi := range []float64{10.0, 56.0, -40.0, 89.0} {
		r := RotateTensor(tn, phi)
		rtr := r.Jxx + r.Jyy
		rrr := math.Hypot(r.Jxx-r.Jyy, 2*r.Jxy)
		approx(t, "lambda1", (rtr+rrr)/2, lam1, 1e-12)
		approx(t, "lambda2", (rtr-rrr)/2, lam2, 1e-12)
		approx(t, "rotated angle", PrincipalAngle(r), Wrap180(theta0+phi), 1e-12)
	}
}
