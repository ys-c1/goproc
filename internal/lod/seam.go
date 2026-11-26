// Package lod 给出地形 LOD 接缝的整除对齐判据、缝宽几何测量与高程重建误差。
//
// 坐标单位统一为“源高度图像素”，所有格距 d_k = 2^k 也是像素。
// 公共原点取 (0,0)，因此 d 级网格线就是坐标为 d 的整数倍的直线。
package lod

import (
	"math"
	"sort"
)

// Spacing 返回 LOD level 对应的格距 d_k = 2^k（像素）。
// level 0 为全分辨率，格距 1；二级、三级、四级分别为 4、8、16。
func Spacing(level int) int {
	if level < 0 {
		panic("lod: level 不能为负")
	}
	return 1 << uint(level)
}

// Divisible 报告整数坐标 coord 是否精确落在步长 d 的网格线上。
// 公共原点为 0 时，这正是“顶点坐标整除”条件：coord % d == 0。
func Divisible(coord, d int) bool {
	return d > 0 && coord%d == 0
}

// DivisibleF 是浮点版本的整除判定，容差 eps 用来吸收 2 的幂次连乘产生的舍入。
func DivisibleF(coord, d, eps float64) bool {
	if d <= 0 {
		return false
	}
	q := coord / d
	return math.Abs(q-math.Round(q)) <= eps
}

// Remainder 是朴素“截断到网格”构造在边界坐标 X 处留下的整除余数 r = X mod d（非负）。
// 网格最后一条线在 X-r，目标边界在 X，二者的像素距离就是 r。
func Remainder(X, d int) int {
	if d <= 0 {
		panic("lod: d 必须为正")
	}
	r := X % d
	if r < 0 {
		r += d
	}
	return r
}

// InsetGap 是“边界按格距比例 rho 内缩”登记约定下的缝宽定律：
//
//	w = rho * |H - h|
//
// rho 即比例系数：半格（像素中心）登记 rho=1/2，整格错位 rho=1，
// 正确的角点对齐 rho=0，此时无论两级格距差多少都有 w=0。
func InsetGap(h, H int, rho float64) float64 {
	return rho * math.Abs(float64(H-h))
}

// Point 是二维像素坐标点。
type Point struct{ X, Y float64 }

// Polyline 是按顺序连接的边界折线。
type Polyline []Point

// PointSegmentDist 返回点 p 到线段 ab 的欧氏距离（像素）。
func PointSegmentDist(p, a, b Point) float64 {
	vx, vy := b.X-a.X, b.Y-a.Y
	wx, wy := p.X-a.X, p.Y-a.Y
	l2 := vx*vx + vy*vy
	var t float64
	if l2 > 0 {
		t = (wx*vx + wy*vy) / l2
		if t < 0 {
			t = 0
		} else if t > 1 {
			t = 1
		}
	}
	cx, cy := a.X+t*vx, a.Y+t*vy
	return math.Hypot(p.X-cx, p.Y-cy)
}

// ToPolylineDistance 返回点 p 到折线 q 的最短距离。
func ToPolylineDistance(p Point, q Polyline) float64 {
	if len(q) == 0 {
		return math.Inf(1)
	}
	if len(q) == 1 {
		return math.Hypot(p.X-q[0].X, p.Y-q[0].Y)
	}
	best := math.Inf(1)
	for i := 0; i+1 < len(q); i++ {
		if d := PointSegmentDist(p, q[i], q[i+1]); d < best {
			best = d
		}
	}
	return best
}

// Hausdorff 返回两条接缝边界折线之间的最大距离（像素）。
//
// 实现取“每条折线的所有顶点到另一条折线距离”的双向最大值。本题所有边界都是
// x 跨度相同的水平平行边：任一边段内部点到对边折线的距离恒等于法向间距，
// 不会超过顶点值，因此顶点最大值就是完整连续 Hausdorff 距离。
// 对平行错位的直边，它就是缝宽；完全重合时严格为零。
func Hausdorff(a, b Polyline) float64 {
	maxD := 0.0
	for _, p := range a {
		if d := ToPolylineDistance(p, b); d > maxD {
			maxD = d
		}
	}
	for _, p := range b {
		if d := ToPolylineDistance(p, a); d > maxD {
			maxD = d
		}
	}
	return maxD
}

// seamXs 返回水平接缝 x∈[x0,x1] 上、步长为 d 的规则顶点 x 坐标，
// 并始终补上两个端点 x0、x1（去重、排序）。
func seamXs(x0, x1, d int) []float64 {
	set := map[int]bool{x0: true, x1: true}
	for x := x0 - (Remainder(x0, d)); x <= x1; x += d {
		if x >= x0 {
			set[x] = true
		}
	}
	xs := make([]int, 0, len(set))
	for x := range set {
		xs = append(xs, x)
	}
	sort.Ints(xs)
	out := make([]float64, len(xs))
	for i, x := range xs {
		out[i] = float64(x)
	}
	return out
}

// NaiveBoundaries 构造整除对齐修复“前”、两级各自独立截断到自身网格时的边界折线。
//
// 水平接缝名义上在 y=Y：步长 d 的一级把最后一条整格线放在 Y-r_d，
// r_d = Y mod d，且不补边界顶点。于是细级在 y_f=Y-r_h、粗级在 y_c=Y-r_H，
// 两条边界的 Hausdorff 距离为 |r_H-r_h|。
// 当 Y=M-1 且 h,H 都整除 M 时 r_d=d-1，故宽度恰为 |H-h|（系数 1）。
// h=1 时 r_h=0，细级恰在 Y，对应“近处全分辨率、远处 LOD-k”的主场景。
//
// 沿接缝方向各自按自身步长细分，因此还保留了 T 接点拓扑。
func NaiveBoundaries(x0, x1, Y, h, H int) (fine, coarse Polyline) {
	xsF := seamXs(x0, x1, h)
	xsC := seamXs(x0, x1, H)
	yf := Y - Remainder(Y, h)
	yc := Y - Remainder(Y, H)
	fine = make(Polyline, len(xsF))
	for i, x := range xsF {
		fine[i] = Point{X: x, Y: float64(yf)}
	}
	coarse = make(Polyline, len(xsC))
	for i, x := range xsC {
		coarse[i] = Point{X: x, Y: float64(yc)}
	}
	return fine, coarse
}

// InsetBoundaries 构造“边界按格距比例 rho 向同一侧内缩”的登记约定下的两条边界，
// 用于验证一般定律 w = rho*|H-h|：细级在 Y-rho*h，粗级在 Y-rho*H。
func InsetBoundaries(x0, x1, Y, h, H int, rho float64) (fine, coarse Polyline) {
	xsF := seamXs(x0, x1, h)
	xsC := seamXs(x0, x1, H)
	yf := float64(Y) - rho*float64(h)
	yc := float64(Y) - rho*float64(H)
	fine = make(Polyline, len(xsF))
	for i, x := range xsF {
		fine[i] = Point{X: x, Y: yf}
	}
	coarse = make(Polyline, len(xsC))
	for i, x := range xsC {
		coarse[i] = Point{X: x, Y: yc}
	}
	return fine, coarse
}

// StitchXs 是修复后接缝上的公共顶点 x 坐标集合：
// 两级规则顶点的并集（再加端点）。H 是 h 的整数倍时，粗级顶点集是细级的子集。
func StitchXs(x0, x1, h, H int) []float64 {
	set := map[int]bool{x0: true, x1: true}
	add := func(d int) {
		for x := x0 - Remainder(x0, d); x <= x1; x += d {
			if x >= x0 {
				set[x] = true
			}
		}
	}
	add(h)
	add(H)
	xs := make([]int, 0, len(set))
	for x := range set {
		xs = append(xs, x)
	}
	sort.Ints(xs)
	out := make([]float64, len(xs))
	for i, x := range xs {
		out[i] = float64(x)
	}
	return out
}

// StitchedBoundaries 构造修复“后”的两条边界折线：
// 两侧使用同一个公共顶点集合 xs、同一条接缝坐标 Y，
// 并在粗级长边缺少顶点处拆边（zipper）。两条折线逐点相同，Hausdorff 距离严格为 0。
func StitchedBoundaries(x0, x1, Y, h, H int) (fine, coarse Polyline) {
	xs := StitchXs(x0, x1, h, H)
	fine = make(Polyline, len(xs))
	coarse = make(Polyline, len(xs))
	for i, x := range xs {
		fine[i] = Point{X: x, Y: float64(Y)}
		coarse[i] = Point{X: x, Y: float64(Y)}
	}
	return fine, coarse
}

// CoarseSubsetOfFine 报告粗级边界顶点是否全部落在细级网格线上，
// 等价于格距整除 H % h == 0（公共原点、h>0）。
func CoarseSubsetOfFine(h, H int) bool {
	return h > 0 && H > 0 && H%h == 0
}
