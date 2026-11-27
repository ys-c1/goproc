// Package hmap 负责归一化与导出。
package hmap

import (
	"encoding/binary"
	"fmt"
)

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
	return EncodePGMBits(size, v, 8)
}

// EncodePGMBits 输出八位或十六位二进制灰度 PGM，
// 量化口径统一为四舍五入，十六位样本按 PGM 规定用大端两字节存储。
func EncodePGMBits(size int, v []float64, bits int) []byte {
	maxval := (1 << bits) - 1
	head := fmt.Sprintf("P5\n%d %d\n%d\n", size, size, maxval)
	per := 1
	if bits == 16 {
		per = 2
	}
	out := make([]byte, 0, len(head)+size*size*per)
	out = append(out, head...)
	var tmp [2]byte
	for _, x := range v {
		g := int(x*float64(maxval) + 0.5)
		if g < 0 {
			g = 0
		}
		if g > maxval {
			g = maxval
		}
		if bits == 16 {
			binary.BigEndian.PutUint16(tmp[:], uint16(g))
			out = append(out, tmp[0], tmp[1])
		} else {
			out = append(out, byte(g))
		}
	}
	return out
}
