package lod

import "errors"

// ErrBadGrid 表示高度图尺寸或参数不合法。
var ErrBadGrid = errors.New("lod: 网格尺寸或参数非法")

// ErrBadLevel 表示请求的 LOD 层级超出可构造范围。
var ErrBadLevel = errors.New("lod: LOD 层级非法")

// SpacingRatio 是细级 i 与粗级 j（j>=i）的格距比 h_j/h_i。
// 二倍格距方案下恒等于 2^(j-i)，只由层级差 d=j-i 决定。
func SpacingRatio(i, j int) float64 {
	return float64(int(1) << uint(j-i)) //nolint:gosec // 层级差很小
}

// PredictedRatio 是面积密度律给出的顶点数比预判值：
// E_i/E_j ≈ (h_j/h_i)^2 = 4^(j-i)。
func PredictedRatio(i, j int) float64 {
	r := SpacingRatio(i, j)
	return r * r
}

// ExactGridRatio 是有限 W×W 格网上的精确顶点数比。
// 每边顶点数 n_l = W/2^l + 1，有效顶点 E_l = n_l^2
// （规则网格中所有顶点都被三角形引用）。
func ExactGridRatio(w, i, j int) float64 {
	ni := float64(w)/float64(int(1)<<uint(i)) + 1
	nj := float64(w)/float64(int(1)<<uint(j)) + 1
	r := ni / nj
	return r * r
}

// SideVerts 给出第 level 级每边顶点数 W/2^level+1。
func SideVerts(w, level int) int {
	return w/(int(1)<<uint(level)) + 1
}

// GridEffective 给出理想规则网格第 level 级的有效顶点数。
func GridEffective(w, level int) int {
	n := SideVerts(w, level)
	return n * n
}
