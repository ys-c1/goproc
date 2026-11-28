package orient

import (
	"math"
	"testing"
)

// 合成脊线 g(x,y)=cos(αx+βy)，梯度法向角应等于 atan2(β,α)（卷绕后）。
func TestGradientAngleOnSyntheticRidge(t *testing.T) {
	const size = 64
	h := make([]float64, size*size)
	want := 35.0 // 度
	rad := want * math.Pi / 180
	a, b := math.Cos(rad), math.Sin(rad)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			h[y*size+x] = math.Cos(a*float64(x) + b*float64(y))
		}
	}
	ts := GradientTensor(h, size, 0, 0, size-1, size-1)
	got := ts.GradientAngle()
	// 二阶中心差分在有限网格上有 O(1/像素数) 的离散偏差，容差 2°。
	if d := math.Abs(got - want); d > 2.0 {
		t.Fatalf("梯度主方向角 got=%.2f want=%.2f", got, want)
	}
	// 脊线走向与法向正交
	if d := math.Abs(wrap90(ts.RidgeAngle()-wrap90(want-90))); d > 2.0 {
		t.Fatalf("脊线角偏差 %.2f", d)
	}
}

// 卷绕与反解：旋转角落在 (-90,90]，且 θR = wrap90(θL+rot)。
func TestRotationBacksolve(t *testing.T) {
	if r := Rotation(80, -80); r != 20 {
		t.Fatalf("跨 ±90 卷绕错误: %v", r)
	}
	if r := Rotation(-45, 45); r != 90 {
		t.Fatalf("有向转角错误: %v", r)
	}
	for _, c := range [][2]float64{{0, 30}, {60, -70}, {-85, 85}} {
		r := Rotation(c[0], c[1])
		if r <= -90 || r > 90 {
			t.Fatalf("转角越界 %v", r)
		}
		if d := math.Abs(wrap90(c[0]+r) - c[1]); d > 1e-9 {
			t.Fatalf("反解不自洽 %v %v", c, r)
		}
	}
}

// 左右分块覆盖整个方阵且不重叠。
func TestSplitVertical(t *testing.T) {
	l, r := SplitVertical(128)
	if l != [4]int{0, 0, 63, 127} || r != [4]int{64, 0, 127, 127} {
		t.Fatalf("分块错误 l=%v r=%v", l, r)
	}
}
