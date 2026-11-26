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

// Gray8 把归一化到零到一的高度量化成八位灰度，四舍五入并夹到 [0,255]。
func Gray8(x float64) uint8 {
	g := int(x*255 + 0.5)
	if g < 0 {
		g = 0
	}
	if g > 255 {
		g = 255
	}
	return uint8(g)
}

// EncodePGM 输出八位灰度 PGM。文件头按 PGM 规范以换行分隔，
// 此前版本误写成字面量反斜杠加 n，导致多数解码器无法读取。
func EncodePGM(size int, v []float64) []byte {
	head := fmt.Sprintf("P5\n%d %d\n255\n", size, size)
	out := make([]byte, 0, len(head)+size*size)
	out = append(out, head...)
	for _, x := range v {
		out = append(out, Gray8(x))
	}
	return out
}
