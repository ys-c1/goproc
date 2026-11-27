package resample

import (
	"math"
	"testing"
)

func TestDownsampleIdentityAtScaleOne(t *testing.T) {
	n := 16
	src := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			src[y*n+x] = math.Sin(float64(x)*0.7) + math.Cos(float64(y)*0.5)
		}
	}
	for _, k := range []Kind{Box, Triangle} {
		out := Downsample(src, n, 1, k)
		if len(out) != len(src) {
			t.Fatalf("%s: 尺寸改变", k)
		}
		for i := range src {
			if math.Abs(out[i]-src[i]) > 1e-12 {
				t.Fatalf("%s: s=1 不是恒等映射，索引 %d 偏差 %g", k, i, out[i]-src[i])
			}
		}
	}
}

func TestDownsampleConstant(t *testing.T) {
	n := 32
	src := make([]float64, n*n)
	for i := range src {
		src[i] = 0.42
	}
	for _, k := range []Kind{Box, Triangle} {
		for _, s := range []float64{0.5, 0.25} {
			out := Downsample(src, n, s, k)
			for _, v := range out {
				if math.Abs(v-0.42) > 1e-12 {
					t.Fatalf("%s s=%v: 常数场未保持，值 %g", k, s, v)
				}
			}
		}
	}
}

func TestBoxHalfSizeAverages(t *testing.T) {
	// s=0.5 盒核在偶数尺寸上应等于 2x2 块平均。
	n := 8
	src := make([]float64, n*n)
	for i := range src {
		src[i] = float64(i)
	}
	out := Downsample(src, n, 0.5, Box)
	if len(out) != 16 {
		t.Fatalf("输出尺寸 %d", len(out))
	}
	for j := 0; j < 4; j++ {
		for i := 0; i < 4; i++ {
			y, x := 2*j, 2*i
			want := (src[y*n+x] + src[y*n+x+1] + src[(y+1)*n+x] + src[(y+1)*n+x+1]) / 4
			if math.Abs(out[j*4+i]-want) > 1e-12 {
				t.Fatalf("块 (%d,%d) 均值不符：got %v want %v", j, i, out[j*4+i], want)
			}
		}
	}
}

func TestUpscaleNearest(t *testing.T) {
	thumb := []float64{1, 2, 3, 4}
	up := UpscaleNearest(thumb, 2, 4)
	want := []float64{
		1, 1, 2, 2,
		1, 1, 2, 2,
		3, 3, 4, 4,
		3, 3, 4, 4,
	}
	for i := range want {
		if up[i] != want[i] {
			t.Fatalf("索引 %d: got %v want %v", i, up[i], want[i])
		}
	}
}

func TestTriangleWiderThanBox(t *testing.T) {
	if Triangle.radius(0.5) <= Box.radius(0.5) {
		t.Fatal("三角核支撑应宽于盒核")
	}
	if Triangle.radius(0.5) != 2*Box.radius(0.5) {
		t.Fatal("三角核支撑应为盒核两倍")
	}
}
