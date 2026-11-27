package edge

import (
	"math"
	"testing"
)

const testM = 0.01 // 二阶导上限（每像素平方）

// 量化容差：八位灰度每点舍入误差 ≤ 0.5/255，二阶差分最多放大四倍。
const quantTol = 2.0 / 255.0

// quantize 模拟 EncodePGM 的八位灰度舍入。
func quantize(v []float64) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		g := int(x*255 + 0.5)
		if g < 0 {
			g = 0
		}
		if g > 255 {
			g = 255
		}
		out[i] = float64(g) / 255
	}
	return out
}

// countJumps 逐行逐列统计灰度二阶差分越限（|d2| > limit）的点数。
func countJumps(gray []float64, size int, limit float64) int {
	n := 0
	for y := 0; y < size; y++ {
		for x := 1; x < size-1; x++ {
			d2 := gray[y*size+x+1] - 2*gray[y*size+x] + gray[y*size+x-1]
			if math.Abs(d2) > limit {
				n++
			}
		}
	}
	for x := 0; x < size; x++ {
		for y := 1; y < size-1; y++ {
			d2 := gray[(y+1)*size+x] - 2*gray[y*size+x] + gray[(y-1)*size+x]
			if math.Abs(d2) > limit {
				n++
			}
		}
	}
	return n
}

// measureWidth 从中线自边缘向内数灰度单调变化的像素数。
func measureWidth(gray []float64, size int) int {
	y := size / 2
	w := 0
	for x := 0; x+1 < size; x++ {
		if gray[y*size+x+1] > gray[y*size+x] {
			w++
		} else {
			break
		}
	}
	return w
}

func TestMinWidthDerivation(t *testing.T) {
	w := Width(1, testM)
	if want := 20; w != want {
		t.Fatalf("Width(1, %v) = %d, 期望 %d", testM, w, want)
	}
	if c := Curvature(w); c > testM {
		t.Fatalf("w=%d 时曲率 %v 越限 %v", w, c, testM)
	}
	if c := Curvature(w - 1); c <= testM {
		t.Fatalf("w=%d 时曲率 %v 应已越限 %v", w-1, c, testM)
	}
}

func TestProfileC1(t *testing.T) {
	w := Width(1, testM)
	const h = 1e-6
	for d := h; d < float64(w); d += 0.37 {
		dl := (Profile(d, w) - Profile(d-h, w)) / h
		dr := (Profile(d+h, w) - Profile(d, w)) / h
		if math.Abs(dl-dr) > 1e-3 {
			t.Fatalf("d=%v 处一阶导不连续: %v vs %v", d, dl, dr)
		}
	}
	if Profile(0, w) != 0 || Profile(float64(w), w) != 1 {
		t.Fatal("端点值错误")
	}
}

// 核心验收：按解出的宽度渲染，跳变点数为零；宽度减一则必然出现跳变。
func TestEdgeRingNoJumps(t *testing.T) {
	w := Width(1, testM)
	for _, size := range []int{128, 256} {
		gray := quantize(Mask(size, w))
		if n := countJumps(gray, size, testM+quantTol); n != 0 {
			t.Fatalf("size=%d w=%d: 渲染灰度跳变点数 %d，期望 0", size, w, n)
		}
		if n := countJumps(Mask(size, w), size, testM*(1+1e-9)); n != 0 {
			t.Fatalf("size=%d w=%d: 浮点掩膜跳变点数 %d，期望 0", size, w, n)
		}
		if got := measureWidth(gray, size); got != w {
			t.Fatalf("size=%d: 实测过渡宽度 %d，期望 %d", size, got, w)
		}
		// 宽度减一：4/(w-1)² > M，二阶差分必然越限。
		if n := countJumps(Mask(size, w-1), size, testM); n == 0 {
			t.Fatalf("size=%d w=%d: 宽度不足应出现跳变", size, w-1)
		}
	}
}

// TestEdgeReport 输出两档画幅的过渡宽度与跳变点数（go test -v 可见）。
func TestEdgeReport(t *testing.T) {
	w := Width(1, testM)
	t.Log("画幅(px)  过渡宽度(px)  跳变点数")
	for _, size := range []int{128, 256} {
		gray := quantize(Mask(size, w))
		n := countJumps(gray, size, testM+quantTol)
		t.Logf("%8d  %12d  %8d", size, measureWidth(gray, size), n)
	}
}
