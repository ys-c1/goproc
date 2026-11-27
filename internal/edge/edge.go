// Package edge 计算地形图边缘过渡带：在一阶导连续、|二阶导| 不超过
// 给定上限 M 的两条约束下，反解过渡曲线的最小像素宽度，并提供该曲线。
package edge

import "math"

// 过渡曲线记为 m(t)，t 为到边缘的像素距离，在 [0,W] 上从 0 单调升到 Δ，
// 两端与常值区 C¹ 拼接：m(0)=0, m(W)=Δ, m'(0)=m'(W)=0，且 |m''| ≤ M。
//
// 由 m'(0)=0 与 |m''| ≤ M 得 |m'(t)| ≤ M·min(t, W−t)，于是
//
//	Δ = ∫₀^W m'(t) dt ≤ M·∫₀^W min(t, W−t) dt = M·W²/4
//
// 即 W ≥ 2·√(Δ/M)。等号当且仅当 m'' 先取 +M 后取 −M（bang-bang 最优），
// 对应下面的对称抛物线 Profile。宽度再小，则任何满足端点条件的 C¹ 曲线都有
// max|m''| ≥ 4Δ/W² > M：二阶导必然越限，渲染上表现为灰度二阶差分出现
// 跳变点，即肉眼可见的亮带。

// MinWidth 返回连续意义下的最小过渡宽度 2·√(Δ/M)。
func MinWidth(delta, maxCurv float64) float64 {
	return 2 * math.Sqrt(delta/maxCurv)
}

// Width 返回满足约束的最小整数像素宽度（对 MinWidth 向上取整）。
func Width(delta, maxCurv float64) int {
	w := int(math.Ceil(MinWidth(delta, maxCurv)))
	if w < 1 {
		w = 1
	}
	return w
}

// Curvature 返回宽度 w 下 Profile 的 |m”|（每像素平方），恰为 4/w²。
func Curvature(w int) float64 { return 4 / float64(w*w) }

// Profile 是过渡曲线（bang-bang 最优的对称抛物线）。d 为到边缘的像素
// 距离，w 为过渡宽度。令 u = d/w：
//
//	u ≤ 1/2: m = 2u²
//	u ≥ 1/2: m = 1 − 2(1−u)²
//
// 满足 m(0)=0, m(w)=1, m'(0)=m'(w)=0，中点处 m 与 m' 均连续，
// |m”| = 4/w² ≤ M 当且仅当 w ≥ Width(1, M)。
func Profile(d float64, w int) float64 {
	if d <= 0 {
		return 0
	}
	if d >= float64(w) {
		return 1
	}
	u := d / float64(w)
	if u <= 0.5 {
		return 2 * u * u
	}
	v := 1 - u
	return 1 - 2*v*v
}

// Mask 生成 size×size 的边缘衰减掩膜，值域 [0,1]。
// 掩膜取两个轴向 Profile 的乘积 m(x,y) = fx(x)·fy(y)，这样沿任意一行
// 或一列的灰度剖面都是同一过渡曲线的常数缩放，逐行逐列的二阶差分上界
// 仍是 4/w²，不会因角部而引入新的曲率尖峰。
func Mask(size, w int) []float64 {
	m := make([]float64, size*size)
	for y := 0; y < size; y++ {
		fy := Profile(float64(min(y, size-1-y)), w)
		for x := 0; x < size; x++ {
			fx := Profile(float64(min(x, size-1-x)), w)
			m[y*size+x] = fx * fy
		}
	}
	return m
}

// Apply 把宽度 w 的边缘衰减掩膜乘到 size×size 的数据上（原地修改）。
func Apply(data []float64, size, w int) {
	m := Mask(size, w)
	for i := range data {
		data[i] *= m[i]
	}
}
