package fbm

import (
	"math"
	"testing"

	"goproc/internal/noise"
)

// 解析导数必须与数值差分一致，否则后续核的比较没有意义。
func TestEvalGradMatchesNumeric(t *testing.T) {
	src := noise.New(7)
	const h = 1e-6
	for i := 0; i < 200; i++ {
		x := float64(i)*0.317 + 0.11
		y := float64(i)*0.193 + 0.27
		v, gx, gy := EvalGrad(src, x, y, 4, 2.0, 0.5)
		if math.Abs(v-Eval(src, x, y, 4, 2.0, 0.5)) > 1e-12 {
			t.Fatalf("值不一致 @(%v,%v)", x, y)
		}
		nx := (Eval(src, x+h, y, 4, 2.0, 0.5) - Eval(src, x-h, y, 4, 2.0, 0.5)) / (2 * h)
		ny := (Eval(src, x, y+h, 4, 2.0, 0.5) - Eval(src, x, y-h, 4, 2.0, 0.5)) / (2 * h)
		if math.Abs(gx-nx) > 1e-5 || math.Abs(gy-ny) > 1e-5 {
			t.Fatalf("导数不一致 @(%v,%v): (%v,%v) vs (%v,%v)", x, y, gx, gy, nx, ny)
		}
	}
}
