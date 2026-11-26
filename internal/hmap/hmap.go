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

// EncodePGMRange 按给定值域输出八位灰度 PGM：lo 映射为黑、hi 为白，
// 越界裁剪。多幅图使用同一组 lo、hi 时灰度才可比，例如侵蚀前后的
// 高度图与差值图。
func EncodePGMRange(size int, v []float64, lo, hi float64) []byte {
	head := fmt.Sprintf("P5\n%d %d\n255\n", size, size)
	out := make([]byte, 0, len(head)+size*size)
	out = append(out, head...)
	if hi == lo {
		for range v {
			out = append(out, 128)
		}
		return out
	}
	scale := 255 / (hi - lo)
	for _, x := range v {
		g := int((x-lo)*scale + 0.5)
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

// EncodePGM 输出八位灰度 PGM。
func EncodePGM(size int, v []float64) []byte {
	lo, hi := v[0], v[0]
	for _, x := range v {
		if x < lo {
			lo = x
		}
		if x > hi {
			hi = x
		}
	}
	return EncodePGMRange(size, v, lo, hi)
}
