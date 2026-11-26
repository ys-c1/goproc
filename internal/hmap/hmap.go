// Package hmap 负责归一化与导出。
package hmap

import "fmt"

// Normalize 把数据线性拉伸到零到一，返回拉伸前的极小值与极大值。
//
// 极差 hi-lo 即归一化把高度缩放掉的倍数；调用方需要还原原始尺度时使用。
func Normalize(v []float64) (lo, hi float64) {
	if len(v) == 0 {
		return 0, 0
	}
	lo, hi = v[0], v[0]
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
		return lo, hi
	}
	scale := 1 / (hi - lo)
	for i := range v {
		v[i] = (v[i] - lo) * scale
	}
	return lo, hi
}

// EncodePGM 输出八位灰度 PGM。
func EncodePGM(size int, v []float64) []byte {
	head := fmt.Sprintf("P5\\n%d %d\\n255\\n", size, size)
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
