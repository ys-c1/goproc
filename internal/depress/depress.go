// Package depress 在 8-bit 灰度格网上做洼地/溢流分析与限深填洼验证。
//
// 与 analysis/depressions.py 相互独立，用于交叉复核同一条恒等式
// （详见 docs/depression-proof.md）：
//
//	strictMin4(F_tau) == { m in strictMin4(G) : spill(m)-G(m) > tau }
//
// 且这些点的坐标逐格不变。深度一律按灰度差计算。
package depress

import (
	"container/heap"
	"math"
)

// Point 是格网坐标，X 向右、Y 向下，均为 0 基。
type Point struct{ X, Y int }

// Pair 是一个预判点与填后实测点的一一配对及偏差。
type Pair struct {
	Pred      Point   `json:"pred"`
	Meas      *Point  `json:"meas"`
	DX        int     `json:"dx"`
	DY        int     `json:"dy"`
	ChebCells int     `json:"chebCells"`
	Euclid    float64 `json:"euclidCells"`
	OverOne   bool    `json:"flagOver1"`
}

// Minima4 枚举内部四邻域严格局部极小。
// 边界格没有四个邻格，且水文上是出画幅的出口，故不计入。
func Minima4(g []int, n int) []Point {
	var out []Point
	at := func(x, y int) int { return g[y*n+x] }
	for y := 1; y < n-1; y++ {
		for x := 1; x < n-1; x++ {
			v := at(x, y)
			if v < at(x-1, y) && v < at(x+1, y) &&
				v < at(x, y-1) && v < at(x, y+1) {
				out = append(out, Point{X: x, Y: y})
			}
		}
	}
	return out
}

type pqItem struct {
	lev int
	y   int
	x   int
	idx int
}

type priorityQueue []*pqItem

func (q priorityQueue) Len() int { return len(q) }
func (q priorityQueue) Less(i, j int) bool {
	if q[i].lev != q[j].lev {
		return q[i].lev < q[j].lev
	}
	if q[i].y != q[j].y {
		return q[i].y < q[j].y
	}
	return q[i].x < q[j].x
}
func (q priorityQueue) Swap(i, j int) {
	q[i], q[j] = q[j], q[i]
	q[i].idx = i
	q[j].idx = j
}
func (q *priorityQueue) Push(v any) {
	it := v.(*pqItem)
	it.idx = len(*q)
	*q = append(*q, it)
}
func (q *priorityQueue) Pop() any {
	old := *q
	it := old[len(old)-1]
	old[len(old)-1] = nil
	*q = old[:len(old)-1]
	return it
}

// Spill 用多源 Dijkstra/Priority-Flood 求每格 minimax 溢流高程：
//
//	S(y,x) = min_{p: (y,x)->边界} max_{c in p} G[c]
//
// 边界格以自身灰度为源。它同时是无 epsilon 完整填洼面。
func Spill(g []int, n int) []int {
	const inf = math.MaxInt32
	s := make([]int, n*n)
	for i := range s {
		s[i] = inf
	}
	q := &priorityQueue{}
	heap.Init(q)
	push := func(x, y int) {
		v := g[y*n+x]
		if v < s[y*n+x] {
			s[y*n+x] = v
			heap.Push(q, &pqItem{lev: v, x: x, y: y})
		}
	}
	for x := 0; x < n; x++ {
		push(x, 0)
		push(x, n-1)
	}
	for y := 0; y < n; y++ {
		push(0, y)
		push(n-1, y)
	}
	dirs := [...]Point{{0, -1}, {0, 1}, {-1, 0}, {1, 0}}
	for q.Len() > 0 {
		it := heap.Pop(q).(*pqItem)
		if it.lev != s[it.y*n+it.x] {
			continue
		}
		for _, d := range dirs {
			nx, ny := it.x+d.X, it.y+d.Y
			if nx < 0 || ny < 0 || nx >= n || ny >= n {
				continue
			}
			idx := ny*n + nx
			nl := g[idx]
			if it.lev > nl {
				nl = it.lev
			}
			if nl < s[idx] {
				s[idx] = nl
				heap.Push(q, &pqItem{lev: nl, x: nx, y: ny})
			}
		}
	}
	return s
}

// Depth 返回极小 m 的洼地（夺顶）深度 Δ=S[m]-G[m]，单位灰度级。
func Depth(g, s []int, m Point, n int) int {
	i := m.Y*n + m.X
	return s[i] - g[i]
}

// FillLimited 是深度阈值 τ 下的限深填洼面 Fτ=min(S, G+τ)：
// 每格相对原始灰度最多抬 τ 个灰度级。
func FillLimited(g, s []int, n, tau int) []int {
	f := make([]int, n*n)
	for i := range g {
		a := g[i] + tau
		if s[i] < a {
			a = s[i]
		}
		f[i] = a
	}
	return f
}

// Cheb 是“格偏差”：max(|dx|,|dy|)，落在八邻域一环内即 ≤1。
func Cheb(a, b Point) int {
	dx := abs(a.X - b.X)
	dy := abs(a.Y - b.Y)
	if dx > dy {
		return dx
	}
	return dy
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// Compare 把预判点与填后实测点一一配对：先锁同坐标，余下贪心取
// Cheb 最近（再比欧氏平方）。返回与 pred 同序的配对。
func Compare(pred, meas []Point) []Pair {
	unused := make(map[int]bool, len(meas))
	for j := range meas {
		unused[j] = true
	}
	out := make([]Pair, 0, len(pred))
	for _, p := range pred {
		best := -1
		bestCheb := math.MaxInt
		bestD2 := math.MaxInt
		// 同坐标优先
		for j := range meas {
			if !unused[j] {
				continue
			}
			if meas[j] == p {
				best = j
				bestCheb = 0
				bestD2 = 0
				break
			}
		}
		if best < 0 {
			for j := range meas {
				if !unused[j] {
					continue
				}
				c := Cheb(p, meas[j])
				dx := p.X - meas[j].X
				dy := p.Y - meas[j].Y
				d2 := dx*dx + dy*dy
				if c < bestCheb || (c == bestCheb && d2 < bestD2) {
					best, bestCheb, bestD2 = j, c, d2
				}
			}
		}
		pr := Pair{Pred: p}
		if best >= 0 {
			q := meas[best]
			unused[best] = false
			dx := q.X - p.X
			dy := q.Y - p.Y
			pr.Meas = &Point{X: q.X, Y: q.Y}
			pr.DX, pr.DY = dx, dy
			pr.ChebCells = Cheb(p, q)
			pr.Euclid = math.Round(math.Hypot(float64(dx), float64(dy))*1000) / 1000
			pr.OverOne = pr.ChebCells > 1
		} else {
			pr.ChebCells = -1
			pr.OverOne = true
		}
		out = append(out, pr)
	}
	return out
}

// SetEq 判定两个点集是否完全相同。
func SetEq(a, b []Point) bool {
	if len(a) != len(b) {
		return false
	}
	seen := make(map[Point]int, len(a))
	for _, p := range a {
		seen[p]++
	}
	for _, p := range b {
		seen[p]--
		if seen[p] < 0 {
			return false
		}
	}
	return true
}
