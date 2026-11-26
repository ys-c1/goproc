package shadow

import (
	"math"
	"testing"

	"goproc/internal/fbm"
	"goproc/internal/noise"
)

// 平地无遮挡，两法都应全受光。
func TestFlatNoShadow(t *testing.T) {
	const size = 16
	h := make([]float64, size*size)
	l := Light{Azimuth: 30, Elevation: 20}
	for name, got := range map[string][]bool{"march": March(h, size, l), "sweep": Sweep(h, size, l)} {
		if n := Count(got); n != 0 {
			t.Fatalf("%s: 平地阴影格数 = %d，应为 0", name, n)
		}
	}
}

// 单根柱子，光朝 +x（az=0），高度角 45°，柱高 1：阴影应恰好落在西侧一格。
func TestPillarShadowLength(t *testing.T) {
	const size = 20
	h := make([]float64, size*size)
	px, py := 10, 7
	h[py*size+px] = 1
	l := Light{Azimuth: 0, Elevation: 45} // tanα = 1，阴影长 = 1/1 = 1 格
	want := map[int]bool{(py*size + px - 1): true}
	for name, got := range map[string][]bool{"march": March(h, size, l), "sweep": Sweep(h, size, l)} {
		if Count(got) != len(want) {
			t.Fatalf("%s: 阴影格数 = %d，应为 %d", name, Count(got), len(want))
		}
		for i := range want {
			if !got[i] {
				t.Fatalf("%s: 格 (%d,%d) 应为阴影", name, i%size, i/size)
			}
		}
	}
}

// 包络递推式：光朝 +x、高度仅依赖 x 时双线性退化为精确值，
// E(x) 应等于解析式 max_{k≥1} f(x+k) − k·tanα。
func TestEnvelopeRecurrence(t *testing.T) {
	const size = 12
	src := noise.New(42)
	f := make([]float64, size)
	for x := 0; x < size; x++ {
		f[x] = fbm.Eval(src, float64(x)*0.3, 0, 3, 2, 0.5)
	}
	h := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			h[y*size+x] = f[x]
		}
	}
	l := Light{Azimuth: 0, Elevation: 25}
	env := Envelope(h, size, l)
	tan := l.tan()
	for x := 0; x < size; x++ {
		want := math.Inf(-1)
		for k := 1; x+k < size; k++ {
			if v := f[x+k] - float64(k)*tan; v > want {
				want = v
			}
		}
		if got := env[x]; got != want { // 全行相同，查 y=0 即可
			if want == math.Inf(-1) {
				if got > -1e8 { // 光线方向上一格外出界：包络应为大负数
					t.Fatalf("x=%d: E = %v，应为负无穷量级", x, got)
				}
				continue
			}
			t.Fatalf("x=%d: E = %v，解析期望 %v", x, got, want)
		}
	}
}

// 两个光照角下两法逐格比对，不一致率应很低（只可能出现在阴影边界）。
func TestSweepMatchesMarch(t *testing.T) {
	const size = 64
	src := noise.New(7)
	h := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			h[y*size+x] = fbm.Eval(src, float64(x)*0.05, float64(y)*0.05, 4, 2, 0.5)
		}
	}
	for _, l := range []Light{{Azimuth: 45, Elevation: 30}, {Azimuth: 300, Elevation: 15}} {
		m := March(h, size, l)
		s := Sweep(h, size, l)
		mis := Diff(h, Envelope(h, size, l), m, s, size)
		rate := float64(len(mis)) / float64(size*size)
		if rate > 0.02 {
			t.Fatalf("az=%v el=%v: 不一致率 %.4f 超过 2%%（%d 格）", l.Azimuth, l.Elevation, rate, len(mis))
		}
	}
}
