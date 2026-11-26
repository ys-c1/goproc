// Package diff 提供方格网上的有限差分算子。
//
// 约定：网格为 N×N，行主序存放，V[y*N+x] 对应第 y 行第 x 列。
// 内部格点一律用二阶中心差分；边界格点无法取到双侧邻点，
// 因此另用单侧公式（一阶导三点，二阶导四点），保证 O(h²) 精度。
package diff

import "errors"

// ErrSize 表示网格太小或数据长度与边长不符。
var ErrSize = errors.New("diff: 需要 N>=4 且 len(V)==N*N")

// Grid 是 N×N 的方格网。
type Grid struct {
	N int
	V []float64
}

// At 返回格点 (x,y) 的值。
func (g Grid) At(x, y int) float64 { return g.V[y*g.N+x] }

// Validate 检查网格是否满足单侧模板所需的最小尺寸。
func Validate(g Grid) error {
	if g.N < 4 || len(g.V) != g.N*g.N {
		return ErrSize
	}
	return nil
}

// D1X 返回 ∂f/∂x 的差分场。
func D1X(g Grid) []float64 {
	if Validate(g) != nil {
		return nil
	}
	return d1x(g.N, g.V)
}

// D1Y 返回 ∂f/∂y 的差分场。
func D1Y(g Grid) []float64 {
	if Validate(g) != nil {
		return nil
	}
	return d1y(g.N, g.V)
}

// D2X 返回 ∂²f/∂x² 的差分场。
func D2X(g Grid) []float64 {
	if Validate(g) != nil {
		return nil
	}
	return d2x(g.N, g.V)
}

// D2Y 返回 ∂²f/∂y² 的差分场。
func D2Y(g Grid) []float64 {
	if Validate(g) != nil {
		return nil
	}
	return d2y(g.N, g.V)
}

// DXY 返回 ∂²f/∂x∂y 的差分场：先沿 y 求一阶差分，再沿 x 求一阶差分。
func DXY(g Grid) []float64 {
	if Validate(g) != nil {
		return nil
	}
	return d1x(g.N, d1y(g.N, g.V))
}

func d1x(n int, a []float64) []float64 {
	out := make([]float64, n*n)
	for y := 0; y < n; y++ {
		row := a[y*n : (y+1)*n]
		o := out[y*n : (y+1)*n]
		o[0] = (-3*row[0] + 4*row[1] - row[2]) / 2
		for x := 1; x < n-1; x++ {
			o[x] = (row[x+1] - row[x-1]) / 2
		}
		o[n-1] = (3*row[n-1] - 4*row[n-2] + row[n-3]) / 2
	}
	return out
}

func d1y(n int, a []float64) []float64 {
	out := make([]float64, n*n)
	for x := 0; x < n; x++ {
		out[x] = (-3*a[x] + 4*a[n+x] - a[2*n+x]) / 2
		for y := 1; y < n-1; y++ {
			out[y*n+x] = (a[(y+1)*n+x] - a[(y-1)*n+x]) / 2
		}
		out[(n-1)*n+x] = (3*a[(n-1)*n+x] - 4*a[(n-2)*n+x] + a[(n-3)*n+x]) / 2
	}
	return out
}

func d2x(n int, a []float64) []float64 {
	out := make([]float64, n*n)
	for y := 0; y < n; y++ {
		row := a[y*n : (y+1)*n]
		o := out[y*n : (y+1)*n]
		o[0] = 2*row[0] - 5*row[1] + 4*row[2] - row[3]
		for x := 1; x < n-1; x++ {
			o[x] = row[x+1] - 2*row[x] + row[x-1]
		}
		o[n-1] = 2*row[n-1] - 5*row[n-2] + 4*row[n-3] - row[n-4]
	}
	return out
}

func d2y(n int, a []float64) []float64 {
	out := make([]float64, n*n)
	for x := 0; x < n; x++ {
		out[x] = 2*a[x] - 5*a[n+x] + 4*a[2*n+x] - a[3*n+x]
		for y := 1; y < n-1; y++ {
			out[y*n+x] = a[(y+1)*n+x] - 2*a[y*n+x] + a[(y-1)*n+x]
		}
		out[(n-1)*n+x] = 2*a[(n-1)*n+x] - 5*a[(n-2)*n+x] + 4*a[(n-3)*n+x] - a[(n-4)*n+x]
	}
	return out
}
