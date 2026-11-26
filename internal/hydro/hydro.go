// Package hydro 在规则方格高度场上做确定性的 D8 流向、坡度与汇流累积分析。
//
// 约定（与 docs/gully-core-proof.md 一致）：
//   - 坐标 x 向右增、y 向下增，格点索引 i = y*N + x；
//   - D8 方向编号固定为 E,SE,S,SW,W,NW,N,NE（顺时针），并列时取小编号；
//   - 只接受严格更低的邻格，故流向边严格下降，全图流向图是 DAG；
//   - 无严格下坡邻格时 Recv=-1（洼地/平地/边界出口终点，不做填洼）。
package hydro

import (
	"math"
	"sort"
)

// 方向编号：顺序不可随意调整，并列裁决依赖它。
const (
	E  = 0
	SE = 1
	S  = 2
	SW = 3
	W  = 4
	NW = 5
	N  = 6
	NE = 7
)

// DX/DY 是八个方向的坐标增量，编号顺序 E,SE,S,SW,W,NW,N,NE。
var DX = [8]int{1, 1, 0, -1, -1, -1, 0, 1}
var DY = [8]int{0, 1, 1, 1, 0, -1, -1, -1}

// Dist 是相邻格的平面距离：正向 1，斜向 sqrt(2)。
var Dist = [8]float64{1, math.Sqrt2, 1, math.Sqrt2, 1, math.Sqrt2, 1, math.Sqrt2}

// Field 是一张高度图对应的水文场，全部字段逐格可复算。
type Field struct {
	N     int       // 边长
	H     []float64 // 原始高程（不归一化也可以，本项目用 0..1）
	Dir   []int     // 最陡方向编号；无去向为 -1
	Recv  []int     // 接收格索引；无去向为 -1
	Slope []float64 // 最陡下坡坡度（落差/距离），无下坡为 0
	Drop  []float64 // 相对接收格的落差，无去向为 0
	Acc   []int     // 汇流累积：上游格数（含本格，每格初始贡献 1）
}

// Idx 把坐标换成一维索引。
func Idx(n, x, y int) int { return y*n + x }

// XY 把一维索引换成坐标。
func XY(n, i int) (x, y int) { return i % n, i / n }

// Inside 判断坐标是否在 n×n 画幅内。
func Inside(n, x, y int) bool { return x >= 0 && y >= 0 && x < n && y < n }

// Build 从高程场计算 D8 流向、坡度与汇流累积。
// h 的长度必须为 n*n。
func Build(h []float64, n int) *Field {
	f := &Field{N: n, H: h, Dir: make([]int, n*n), Recv: make([]int, n*n),
		Slope: make([]float64, n*n), Drop: make([]float64, n*n), Acc: make([]int, n*n)}
	for i := range f.Dir {
		f.Dir[i] = -1
		f.Recv[i] = -1
		f.Acc[i] = 1
	}

	// 第一步：逐格只读邻格，求最陡下坡。格间无写依赖。
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			i := Idx(n, x, y)
			best := 0.0
			bestD := -1
			bestDrop := 0.0
			for d := 0; d < 8; d++ {
				nx, ny := x+DX[d], y+DY[d]
				if !Inside(n, nx, ny) {
					continue
				}
				j := Idx(n, nx, ny)
				drop := h[i] - h[j]
				if drop <= 0 { // 只接受严格更低
					continue
				}
				g := drop / Dist[d]
				if g > best { // 严格大于：并列保留先扫到的小编号方向
					best = g
					bestD = d
					bestDrop = drop
				}
			}
			if bestD >= 0 {
				f.Dir[i] = bestD
				f.Recv[i] = Idx(n, x+DX[bestD], y+DY[bestD])
				f.Slope[i] = best
				f.Drop[i] = bestDrop
			}
		}
	}

	// 第二步：按高程从高到低结算汇流。
	// 流向边严格下降，所以所有上游格必在本格之前处理。
	order := make([]int, n*n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		ia, ib := order[a], order[b]
		if h[ia] != h[ib] {
			return h[ia] > h[ib]
		}
		return ia < ib // 确定性并列裁决
	})
	for _, i := range order {
		if r := f.Recv[i]; r >= 0 {
			f.Acc[r] += f.Acc[i]
		}
	}
	return f
}

// SinkCount 统计无严格下坡去向的格子数（含边界出口终点与内陆洼地/平地）。
func (f *Field) SinkCount() int {
	c := 0
	for _, r := range f.Recv {
		if r < 0 {
			c++
		}
	}
	return c
}
