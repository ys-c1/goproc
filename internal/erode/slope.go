package erode

import (
	"math"
	"sort"
)

// Slopes 返回每个格子的坡度（度），按整图逐格计算，边界用复制延拓。
// 高度图视为张在边长为 span 的正方形上，格距 cell = span/size，
// 高度值域零到一，与水平同单位，故坡度角有实际几何意义。
// 梯度用 Horn 三点加权算子。
func Slopes(h []float64, size int, span float64) []float64 {
	cell := span / float64(size)
	out := make([]float64, len(h))
	at := func(x, y int) float64 {
		if x < 0 {
			x = 0
		}
		if x >= size {
			x = size - 1
		}
		if y < 0 {
			y = 0
		}
		if y >= size {
			y = size - 1
		}
		return h[y*size+x]
	}
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			dzdx := (at(x+1, y-1) + 2*at(x+1, y) + at(x+1, y+1) -
				at(x-1, y-1) - 2*at(x-1, y) - at(x-1, y+1)) / (8 * cell)
			dzdy := (at(x-1, y+1) + 2*at(x, y+1) + at(x+1, y+1) -
				at(x-1, y-1) - 2*at(x, y-1) - at(x+1, y-1)) / (8 * cell)
			out[y*size+x] = math.Atan(math.Hypot(dzdx, dzdy)) * 180 / math.Pi
		}
	}
	return out
}

// Median 返回中位数，不修改入参。
func Median(v []float64) float64 {
	s := make([]float64, len(v))
	copy(s, v)
	sort.Float64s(s)
	n := len(s)
	if n%2 == 1 {
		return s[n/2]
	}
	return (s[n/2-1] + s[n/2]) / 2
}

// Variance 返回总体方差。
func Variance(v []float64) float64 {
	if len(v) == 0 {
		return 0
	}
	var mean float64
	for _, x := range v {
		mean += x
	}
	mean /= float64(len(v))
	var acc float64
	for _, x := range v {
		d := x - mean
		acc += d * d
	}
	return acc / float64(len(v))
}
