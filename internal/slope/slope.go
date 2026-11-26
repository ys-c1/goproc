// Package slope 估计高度图的坡度，并给出边缘一圈的偏量修正。
//
// 记格距 dx。对内部像素用中心差分（双侧差分）：
//
//	gx = (h[x+1,y]-h[x-1,y]) / (2dx)
//
// 对边缘像素，朴素做法只能用单侧差分，例如左边界：
//
//	gx_edge = (h[1,y]-h[0,y]) / dx
//
// 把两者写成同一像素处的泰勒展开（h_k 表示 h 在 x=k 处的值）：
//
//	h(1)  = h0 + h' dx + h'' dx²/2 + h''' dx³/6 + ...
//	h(-1) = h0 - h' dx + h'' dx²/2 - h''' dx³/6 + ...
//	单侧前差  F = (h(1)-h(0))/dx      = h' + h'' dx/2 + O(dx²)
//	双侧中心差 C = (h(1)-h(-1))/(2dx) = h' + O(dx²)
//
// 于是单侧差相对双侧差的偏量为
//
//	Δ = F - C = (h(1)+h(-1)-2h(0)) / (2dx) = (dx/2)·κ + O(dx³)
//
// 其中 κ = h” 是曲率。也就是说偏量恰是「格距 × 曲率 / 2」：
// 每单位格距贡献 κ/2，量纲为斜率（高度/长度）。dx=1 时 Δ = κ/2，
// 数值上等于该格二阶差分的一半。
//
// 边界延拓补偿：在图像外补一圈鬼点，鬼点取
//
//	h(-1) = 2h(0) - h(1) + dx·κ_est
//
// 则补偿后的中心差
//
//	(h(1)-h(-1))/(2dx) = F - (dx/2)·κ_est ≈ h' + O(dx²)
//
// 偏量被扣掉，边缘坡度与内部同为二阶精度。κ_est 只能由图像内点估计：
// 用 4 点单侧二阶差分 2h0-5h1+4h2-h3（误差 O(dx²)，与内部中心差同阶），
// 因此补偿既去偏又不降阶。本实现以像素为格距单位（dx=1），
// 鬼点即 2h(0)-h(1)+κ_est，与包内 EdgeCurvature 的取值一致。
//
// 补偿只改动边缘一圈（按一格计），内部像素的坡度逐位不变。
package slope

import "math"

// ScalarField 是标量场，结果按行主序存放。
type ScalarField struct {
	Size int
	Data []float64
}

// At 取 (x,y) 处的值。
func (f *ScalarField) At(x, y int) float64 { return f.Data[y*f.Size+x] }

// NewScalarField 由按行主序的长 data 构造，尺寸由边长给定。
func NewScalarField(size int, data []float64) *ScalarField {
	if len(data) != size*size {
		panic("slope: 长度与尺寸不符")
	}
	return &ScalarField{Size: size, Data: data}
}

// Field 是坡度场，保留有符号分量便于逐像素比对与调试。
type Field struct {
	Size int
	Gx   []float64
	Gy   []float64
}

// At 返回 (x,y) 处坡度分量。
func (f *Field) At(x, y int) (gx, gy float64) {
	return f.Gx[y*f.Size+x], f.Gy[y*f.Size+x]
}

// Mag 返回 (x,y) 处坡度模长。
func (f *Field) Mag(x, y int) float64 {
	gx, gy := f.At(x, y)
	return math.Hypot(gx, gy)
}

// MaxMag 返回全场坡度模长的最大值及其位置。
func (f *Field) MaxMag() (m float64, x, y int) {
	m = -1
	for yy := 0; yy < f.Size; yy++ {
		for xx := 0; xx < f.Size; xx++ {
			if v := f.Mag(xx, yy); v > m {
				m, x, y = v, xx, yy
			}
		}
	}
	return m, x, y
}

// IsRing 判断 (x,y) 是否落在边缘一圈（一格宽）内。
func IsRing(n, x, y int) bool { return x == 0 || y == 0 || x == n-1 || y == n-1 }

// RingSize 返回边缘一圈的像素数（一格宽）。
func RingSize(n int) int {
	if n <= 0 {
		return 0
	}
	if n <= 2 {
		return n * n
	}
	return 4*n - 4
}

// CountChanged 统计两场坡度模长不同的像素数。
func CountChanged(a, b *Field) int {
	if a.Size != b.Size {
		panic("slope: 尺寸不一致")
	}
	c := 0
	for i := range a.Gx {
		if a.Mag2(i) != b.Mag2(i) {
			c++
		}
	}
	return c
}

// Mag2 按行主序下标取模长。
func (f *Field) Mag2(i int) float64 {
	return math.Hypot(f.Gx[i], f.Gy[i])
}

// RingMismatch 返回改变像素中落在边缘一圈之外的个数。
func RingMismatch(a, b *Field) int {
	if a.Size != b.Size {
		panic("slope: 尺寸不一致")
	}
	bad := 0
	n := a.Size
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			i := y*n + x
			if a.Mag2(i) != b.Mag2(i) && !IsRing(n, x, y) {
				bad++
			}
		}
	}
	return bad
}

// Gradient 用中心差分估计坡度。边缘一圈只有单侧差分可用，
// 边界处退化为前向/后向差分，这正是偏量的来源。
func Gradient(h *ScalarField) *Field {
	n := h.Size
	f := &Field{Size: n, Gx: make([]float64, n*n), Gy: make([]float64, n*n)}
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			var gx, gy float64
			switch {
			case n < 2:
				gx, gy = 0, 0
			case x == 0:
				gx = h.At(1, y) - h.At(0, y)
			case x == n-1:
				gx = h.At(n-1, y) - h.At(n-2, y)
			default:
				gx = (h.At(x+1, y) - h.At(x-1, y)) / 2
			}
			switch {
			case n < 2:
				gy = 0
			case y == 0:
				gy = h.At(x, 1) - h.At(x, 0)
			case y == n-1:
				gy = h.At(x, n-1) - h.At(x, n-2)
			default:
				gy = (h.At(x, y+1) - h.At(x, y-1)) / 2
			}
			f.Gx[y*n+x] = gx
			f.Gy[y*n+x] = gy
		}
	}
	return f
}

// EdgeCurvature 估计一维序列在左端的曲率 κ≈h”（dx=1，即二阶差分）。
// 用 4 点单侧公式 2a0-5a1+4a2-a3，误差 O(dx²)；
// 点数不足时退化为 3 点公式 a0-2a1+a2。
func EdgeCurvature(a []float64) float64 {
	switch {
	case len(a) >= 4:
		return 2*a[0] - 5*a[1] + 4*a[2] - a[3]
	case len(a) == 3:
		return a[0] - 2*a[1] + a[2]
	default:
		return 0
	}
}

// EdgeCurvatureRight 是右端的对称版本。
func EdgeCurvatureRight(a []float64) float64 {
	n := len(a)
	switch {
	case n >= 4:
		return 2*a[n-1] - 5*a[n-2] + 4*a[n-3] - a[n-4]
	case n == 3:
		return a[n-1] - 2*a[n-2] + a[n-3]
	default:
		return 0
	}
}

// Extend 做边界延拓：在图像外补一圈鬼点，
// 鬼点 = 2*边缘 - 相邻内点 + κ_est（线性外推再加曲率修正）。
// 返回 (n+2)x(n+2) 的延拓场，坐标偏移 1。
func Extend(h *ScalarField) *ScalarField {
	n := h.Size
	if n < 2 {
		m := n + 2
		out := make([]float64, m*m)
		for i := range out {
			out[i] = h.Data[0]
		}
		return &ScalarField{Size: m, Data: out}
	}
	m := n + 2
	// 第一遍：内部复制 + 左右鬼列。
	w := make([]float64, n*m) // n 行，宽 m
	for y := 0; y < n; y++ {
		row := make([]float64, n)
		for x := 0; x < n; x++ {
			row[x] = h.At(x, y)
		}
		kl := EdgeCurvature(row)
		kr := EdgeCurvatureRight(row)
		w[y*m+0] = 2*row[0] - row[1] + kl
		w[y*m+m-1] = 2*row[n-1] - row[n-2] + kr
		for x := 0; x < n; x++ {
			w[y*m+1+x] = row[x]
		}
	}
	// 第二遍：上下鬼行。对每一列（宽 m）用同一公式沿 y 方向延拓。
	out := make([]float64, m*m)
	col := make([]float64, n)
	for x := 0; x < m; x++ {
		for y := 0; y < n; y++ {
			col[y] = w[y*m+x]
		}
		kt := EdgeCurvature(col)
		kb := EdgeCurvatureRight(col)
		out[0*m+x] = 2*col[0] - col[1] + kt
		out[(m-1)*m+x] = 2*col[n-1] - col[n-2] + kb
		for y := 0; y < n; y++ {
			out[(y+1)*m+x] = col[y]
		}
	}
	return &ScalarField{Size: m, Data: out}
}

// GradientCompensated 在延拓场上做中心差分，再取回原画幅的坡度。
// 边缘一圈因此与内部使用同一个差分算子；由包注释的推导，
// 其相对未补偿值的改变量正是偏量 (dx/2)κ_est。
func GradientCompensated(h *ScalarField) *Field {
	n := h.Size
	ext := Extend(h)

	f := &Field{Size: n, Gx: make([]float64, n*n), Gy: make([]float64, n*n)}
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			ex, ey := x+1, y+1
			f.Gx[y*n+x] = (ext.At(ex+1, ey) - ext.At(ex-1, ey)) / 2
			f.Gy[y*n+x] = (ext.At(ex, ey+1) - ext.At(ex, ey-1)) / 2
		}
	}
	return f
}

// CurvatureEstimate 返回 (x,y) 处沿 x、y 方向的曲率估计 κ≈h”。
// 内部点用中心二阶差分；边缘点用 4 点单侧公式（与补偿所用一致）。
func CurvatureEstimate(h *ScalarField, x, y int) (cxx, cyy float64) {
	n := h.Size
	if n < 3 {
		return 0, 0
	}
	if x == 0 || x == n-1 {
		row := make([]float64, n)
		for xx := 0; xx < n; xx++ {
			row[xx] = h.At(xx, y)
		}
		if x == 0 {
			cxx = EdgeCurvature(row)
		} else {
			cxx = EdgeCurvatureRight(row)
		}
	} else {
		cxx = h.At(x-1, y) - 2*h.At(x, y) + h.At(x+1, y)
	}
	if y == 0 || y == n-1 {
		col := make([]float64, n)
		for yy := 0; yy < n; yy++ {
			col[yy] = h.At(x, yy)
		}
		if y == 0 {
			cyy = EdgeCurvature(col)
		} else {
			cyy = EdgeCurvatureRight(col)
		}
	} else {
		cyy = h.At(x, y-1) - 2*h.At(x, y) + h.At(x, y+1)
	}
	return cxx, cyy
}

// BiasEstimate 返回 (x,y) 处「未补偿坡度 − 补偿后坡度」的偏量，
// 分量为 (dx/2)·κ，dx=1 时即 κ/2。
//
// 只有当该轴在该侧确实退化为单侧差分时才非零、且符号随侧翻转：
//   - 左边界 x=0：前向差，偏量 +κx/2；右边界 x=n-1：后向差，偏量 −κx/2。
//   - 上边界 y=0：前向差，+κy/2；下边界 y=n-1：后向差，−κy/2。
//   - 该轴为内部（两侧都有邻居）时偏量为 0：中心差本就没有边界退化。
//
// 这与补偿的实际行为逐位一致（见 GradientCompensated）。
func BiasEstimate(h *ScalarField, x, y int) (bx, by float64) {
	cxx, cyy := CurvatureEstimate(h, x, y)
	switch {
	case x == 0:
		bx = cxx / 2
	case x == h.Size-1:
		bx = -cxx / 2
	}
	switch {
	case y == 0:
		by = cyy / 2
	case y == h.Size-1:
		by = -cyy / 2
	}
	return bx, by
}
