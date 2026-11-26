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

// EncodePGM 输出八位灰度 PGM。
func EncodePGM(size int, v []float64) []byte {
	head := fmt.Sprintf("P5\n%d %d\n255\n", size, size)
	out := make([]byte, 0, len(head)+size*size)
	out = append(out, head...)
	for _, x := range v {
		g := int(x*255 + 0.5)
		if g < 0 {
			g = 0
		}
		if g > 255 {
			g = 255
		}
		out = append(out, byte(g))
	}
	return out
}

// EncodePGM16 输出十六位灰度 PGM，码值按 PGM 规范大端存放。
func EncodePGM16(size int, v []float64) []byte {
	head := fmt.Sprintf("P5\n%d %d\n65535\n", size, size)
	out := make([]byte, 0, len(head)+2*size*size)
	out = append(out, head...)
	for _, x := range v {
		g := int(x*65535 + 0.5)
		if g < 0 {
			g = 0
		}
		if g > 65535 {
			g = 65535
		}
		out = append(out, byte(g>>8), byte(g))
	}
	return out
}
