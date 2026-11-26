// Package band 统计量化高度图中的等高带：同一灰度码的四连通段即一条带。
package band

import "math"

// Quantize 把零到一的高度值量化为 bits 位灰度码，码值范围零到 2^bits-1，
// 舍入方式与 hmap.EncodePGM 一致（四舍五入并截断到码值域）。
func Quantize(v []float64, bits uint) []uint32 {
	if bits == 0 || bits > 16 {
		panic("band: bits 须在 1 到 16 之间")
	}
	maxCode := int64(1<<bits) - 1
	out := make([]uint32, len(v))
	for i, x := range v {
		c := int64(x*float64(maxCode) + 0.5)
		if c < 0 {
			c = 0
		}
		if c > maxCode {
			c = maxCode
		}
		out[i] = uint32(c)
	}
	return out
}

// Step 返回 bits 位量化的步长：码值域 [0, 2^bits-1] 铺满单位区间，
// 相邻码值间距 Δ = 1/(2^bits - 1)。
func Step(bits uint) float64 {
	return 1 / float64(int64(1<<bits)-1)
}

// Predict 按高度跨度与量化步长预判带数：floor(span/Δ) + 1，
// 加一是因为跨度两端各落在一条带上。
func Predict(span float64, bits uint) int {
	return int(math.Floor(span/Step(bits))) + 1
}

// Segment 是同一灰度码的一个四连通段。
type Segment struct {
	Code   uint32 // 灰度码
	Row0   int    // 段占据的首行行号
	Row1   int    // 段占据的末行行号
	Pixels int    // 段内像素数
}

// Scan 逐像素扫描 size×size 的量化网格，按四连通统计同灰度连通段，
// 返回总段数与逐段信息。
func Scan(grid []uint32, size int) (int, []Segment) {
	seen := make([]bool, len(grid))
	var segs []Segment
	dx := [4]int{1, -1, 0, 0}
	dy := [4]int{0, 0, 1, -1}
	for start := range grid {
		if seen[start] {
			continue
		}
		code := grid[start]
		seg := Segment{Code: code, Row0: start / size, Row1: start / size}
		stack := []int{start}
		seen[start] = true
		for len(stack) > 0 {
			p := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			seg.Pixels++
			x, y := p%size, p/size
			if y < seg.Row0 {
				seg.Row0 = y
			}
			if y > seg.Row1 {
				seg.Row1 = y
			}
			for d := 0; d < 4; d++ {
				nx, ny := x+dx[d], y+dy[d]
				if nx < 0 || ny < 0 || nx >= size || ny >= size {
					continue
				}
				q := ny*size + nx
				if !seen[q] && grid[q] == code {
					seen[q] = true
					stack = append(stack, q)
				}
			}
		}
		segs = append(segs, seg)
	}
	return len(segs), segs
}
