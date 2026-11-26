package aniso

import (
	"math"
	"testing"
)

func comps() []Component {
	return []Component{
		{Seed: 1, Strike: math.Pi / 3, Ratio: 3.0, Frequency: 0.03, Weight: 1},
		{Seed: 2, Strike: 5 * math.Pi / 6, Ratio: 3.0, Frequency: 0.03, Weight: 1},
	}
}

// TestRMSDeterministic 同样输入必须逐点一致（种子派生，无全局状态）。
func TestRMSDeterministic(t *testing.T) {
	a := FieldRMS(64, 2, 2.0, 0.5, comps())
	b := FieldRMS(64, 2, 2.0, 0.5, comps())
	if len(a) != len(b) {
		t.Fatalf("长度不符 %d != %d", len(a), len(b))
	}
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("第 %d 点不确定: %v != %v", i, a[i], b[i])
		}
	}
}

// TestRMSNormalized 输出须落在 [0,1]。
func TestRMSNormalized(t *testing.T) {
	z := FieldRMS(64, 2, 2.0, 0.5, comps())
	lo, hi := z[0], z[0]
	for _, v := range z {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	if lo < 0 || hi > 1 || math.Abs(lo) > 1e-12 || math.Abs(hi-1) > 1e-12 {
		t.Fatalf("归一化越界: lo=%v hi=%v", lo, hi)
	}
}

// TestRMSContinuous RMS 合成保持连续：相邻像素有限差分有界，无 abs 式尖点跳变。
func TestRMSContinuous(t *testing.T) {
	n := 64
	z := FieldRMS(n, 2, 2.0, 0.5, comps())
	var maxJump float64
	for y := 0; y < n; y++ {
		for x := 1; x < n; x++ {
			if d := math.Abs(z[y*n+x] - z[y*n+x-1]); d > maxJump {
				maxJump = d
			}
		}
	}
	// 归一化到 [0,1] 后，单像素跳变应远小于 1（光滑场）
	if maxJump > 0.25 {
		t.Fatalf("存在尖点跳变 maxJump=%v，RMS 合成可能不光滑", maxJump)
	}
}

// TestAnisoBimodality 非相干合成的两个正交脊系应在两个期望梯度方向都形成高 HOG 占比。
func TestAnisoBimodality(t *testing.T) {
	n := 128
	z := FieldRMS(n, 2, 2.0, 0.5, comps())
	h := make([]float64, 180)
	for y := 1; y < n-1; y++ {
		for x := 1; x < n-1; x++ {
			gx := z[y*n+x+1] - z[y*n+x-1]
			gy := z[(y+1)*n+x] - z[(y-1)*n+x]
			a := math.Atan2(gy, gx) * 180 / math.Pi
			if a < 0 {
				a += 180
			}
			h[int(a)%180] += math.Hypot(gx, gy)
		}
	}
	// 走向 60->梯度 150；走向 150->梯度 60。两方向邻域均值应高于全局均值。
	var mean float64
	for _, v := range h {
		mean += v
	}
	mean /= 180
	windowMean := func(c int) float64 {
		var s float64
		for k := -6; k <= 6; k++ {
			s += h[(c+k+180)%180]
		}
		return s / 13
	}
	for _, dir := range []int{60, 150} {
		if windowMean(dir) <= mean {
			t.Fatalf("方向 %d 未形成 HOG 峰: 窗均值 %.3f <= 全局均值 %.3f", dir, windowMean(dir), mean)
		}
	}
}
