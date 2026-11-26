// Package hmap 负责归一化与导出。
package hmap

import "fmt"

// Normalize 把数据线性拉伸到零到一。
func Normalize(v []float64) {
	if len(v) == 0 {
		return
	}
	lo, hi := v[0], v[0]
	for _, x := range v {
		if x < lo {
			lo = x
		}
		if x > hi {
			hi = x
		}
	}
	if hi == lo {
		for i := range v {
			v[i] = 0
		}
		return
	}
	scale := 1 / (hi - lo)
	for i := range v {
		v[i] = (v[i] - lo) * scale
	}
}

func clamp8(g int) int {
	if g < 0 {
		return 0
	}
	if g > 255 {
		return 255
	}
	return g
}

// ToGray8 把零到一的浮点高度量化为零到二五五的灰度整数。
func ToGray8(v []float64) []int {
	g := make([]int, len(v))
	for i, x := range v {
		g[i] = clamp8(int(x*255 + 0.5))
	}
	return g
}

// EncodePGM 输出八位灰度 PGM。
func EncodePGM(size int, v []float64) []byte {
	return EncodePGM8(size, ToGray8(v))
}

// EncodePGM8 直接由灰度整数输出八位 PGM。
func EncodePGM8(size int, g []int) []byte {
	head := fmt.Sprintf("P5\n%d %d\n255\n", size, size)
	out := make([]byte, 0, len(head)+size*size)
	out = append(out, head...)
	for _, x := range g {
		out = append(out, byte(clamp8(x)))
	}
	return out
}
