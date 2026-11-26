package sampling

import "math"

import "testing"

func TestStepsIsotropic(t *testing.T) {
	hX, hY := Steps(1.0, 1.0, 1.0)
	if hX != 0.5 || hY != 0.5 {
		t.Fatalf("各向同性步长应为 1/(2f)=0.5，实际 %v %v", hX, hY)
	}
}

func TestStepsScaled(t *testing.T) {
	// 缩放比 r 的短轴方向：局部频率 r*fBase，步长反比于 r。
	hX, hY := Steps(2.0, 1.0, 4.0)
	if math.Abs(hX-0.25) > 1e-12 || math.Abs(hY-0.0625) > 1e-12 {
		t.Fatalf("步长反解错误: %v %v", hX, hY)
	}
}

func TestCounts(t *testing.T) {
	nX, nY := Counts(32.0, 0.5, 0.125)
	if nX != 64 || nY != 256 {
		t.Fatalf("奈奎斯特边采样数错误: %d %d", nX, nY)
	}
}
