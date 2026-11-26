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

// EncodePGM 输出八位灰度 PGM，输入假定已归一化到零到一。
func EncodePGM(size int, v []float64) []byte {
	return EncodePGMScaled(size, v, 0, 1)
}

// EncodePGMScaled 按给定最小/最大值线性映射到八位灰度，越界截断。
func EncodePGMScaled(size int, v []float64, lo, hi float64) []byte {
	head := fmt.Sprintf("P5\\n%d %d\\n255\\n", size, size)
	out := make([]byte, 0, len(head)+size*size)
	out = append(out, head...)
	span := hi - lo
	for _, x := range v {
		var g int
		if span == 0 {
			g = 0
		} else {
			g = int((x-lo)/span*255 + 0.5)
		}
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
