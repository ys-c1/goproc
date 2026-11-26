// Package lod 把高度图变成带索引的地形网格，并构造格距逐级翻倍的
// 离散 LOD。所有顶点计数一律以“被三角形索引引用过的不同顶点”
// （有效顶点）为准，而不是顶点缓冲/候选池长度。
package lod

import (
	"math"
	"sort"
)

// Grid 是一张 N×N 的规则高度网格。N 个顶点对应 N-1 个格距区间，
// 约定 N-1 必须能被最大二分级数整除。
type Grid struct {
	N      int       // 每边顶点数
	Cell   float64   // 0 级格距（世界单位）
	Relief float64   // 归一化高度到世界高程的垂直缩放
	H      []float64 // 长度 N*N 的归一化高度
}

// NewGrid 校验并构造规则网格。
func NewGrid(h []float64, n int, cell, relief float64) (*Grid, error) {
	if n < 2 || len(h) != n*n {
		return nil, ErrBadGrid
	}
	if cell <= 0 || relief <= 0 {
		return nil, ErrBadGrid
	}
	return &Grid{N: n, Cell: cell, Relief: relief, H: h}, nil
}

// SideCells 返回每边的 0 级格距区间数 W=N-1。
func (g *Grid) SideCells() int { return g.N - 1 }

// ID 返回整数格点 (ix,iy) 在共享顶点池中的编号。
func (g *Grid) ID(ix, iy int) uint32 { return uint32(iy*g.N + ix) }

// Position 返回池编号 id 对应的世界坐标 (x,y,z)。
func (g *Grid) Position(id uint32) (x, y, z float64) {
	ix := int(id) % g.N
	iy := int(id) / g.N
	return float64(ix) * g.Cell, float64(iy) * g.Cell, g.H[int(id)] * g.Relief
}

// PoolVertices 是共享顶点缓冲的分配顶点数（未按引用压缩，不能直接
// 当作 LOD 顶点规模使用）。
func (g *Grid) PoolVertices() int { return g.N * g.N }

// MaxLevel 返回仍能构成至少 2×2 顶点（1 个四边形）的最大二分级数。
func (g *Grid) MaxLevel() int {
	w := g.SideCells()
	l := 0
	for w%(1<<(l+1)) == 0 && w/(1<<(l+1)) >= 1 {
		l++
	}
	return l
}

// Level 是某一级 LOD 的实际网格化结果。Indices 是三角形列表，
// 引用共享顶点池 Grid 中的编号。
type Level struct {
	Level     int      // 层级号 ℓ，格距 = Cell * 2^ℓ
	Step      int      // 在 0 级高度图上的抽稀步长 = 2^ℓ
	SideVerts int      // 每边有效顶点数 = W/2^ℓ + 1
	Indices   []uint32 // 三角形索引（长度 = 3 * Triangles）

	// ErrMax 是该级双线性插值面相对原始高度图的最大绝对高程误差
	// （归一化高度单位）；ErrRMS 是均方根误差。
	ErrMax float64
	ErrRMS float64
}

// Triangles 返回三角形个数。
func (l *Level) Triangles() int { return len(l.Indices) / 3 }

// EffectiveVertices 按题目口径统计有效顶点：索引缓冲中被三角形
// 引用过的不同顶点编号数。顶点池里未被任何索引引用的顶点不计。
func EffectiveVertices(indices []uint32) int {
	seen := make(map[uint32]struct{}, len(indices)/3)
	for _, id := range indices {
		seen[id] = struct{}{}
	}
	return len(seen)
}

// EffectiveVerticesSorted 与 EffectiveVertices 等价的第二实现
// （排序去重），供报告端交叉核验，避免单一计数实现出错。
func EffectiveVerticesSorted(indices []uint32) int {
	if len(indices) == 0 {
		return 0
	}
	cp := append([]uint32(nil), indices...)
	sort.Slice(cp, func(i, j int) bool { return cp[i] < cp[j] })
	n := 1
	for i := 1; i < len(cp); i++ {
		if cp[i] != cp[i-1] {
			n++
		}
	}
	return n
}

// Effective 返回本级的有效顶点数。
func (l *Level) Effective() int { return EffectiveVertices(l.Indices) }

// Spacing 返回该级格距（世界单位）。
func (g *Grid) Spacing(level int) float64 {
	return g.Cell * float64(int(1)<<level)
}

// BuildLevel 实际执行一次二倍抽稀网格化：只保留行列号均为 Step
// 整数倍的格点，并为每个粗网格四边形生成两个三角形。这就是
// “逐级实际简化”的真实路径。
func (g *Grid) BuildLevel(level int) (*Level, error) {
	if level < 0 || level > g.MaxLevel() {
		return nil, ErrBadLevel
	}
	step := 1 << level
	w := g.SideCells()
	if w%step != 0 {
		return nil, ErrBadLevel
	}
	quads := w / step
	sideVerts := quads + 1

	idx := make([]uint32, 0, quads*quads*6)
	for j := 0; j < quads; j++ {
		y0, y1 := j*step, (j+1)*step
		for i := 0; i < quads; i++ {
			x0, x1 := i*step, (i+1)*step
			v00 := g.ID(x0, y0)
			v10 := g.ID(x1, y0)
			v01 := g.ID(x0, y1)
			v11 := g.ID(x1, y1)
			// 与高度图坐标系一致的绕向
			idx = append(idx, v00, v10, v11, v00, v11, v01)
		}
	}

	eMax, eRMS := g.MeasureError(level)
	return &Level{
		Level:     level,
		Step:      step,
		SideVerts: sideVerts,
		Indices:   idx,
		ErrMax:    eMax,
		ErrRMS:    eRMS,
	}, nil
}

// MeasureError 计算第 level 级粗网格的双线性插值面相对原始逐像素
// 高度的误差。格点本身误差为零，其余原始格点是真正的检验样本。
func (g *Grid) MeasureError(level int) (eMax, eRMS float64) {
	step := 1 << level
	if step <= 1 {
		return 0, 0
	}
	w := g.SideCells()
	var sumSq float64
	var n int
	for y := 0; y <= w; y++ {
		gy0 := (y / step) * step
		gy1 := gy0 + step
		if gy1 > w {
			gy1 = w
		}
		fy := 0.0
		if gy1 != gy0 {
			fy = float64(y-gy0) / float64(gy1-gy0)
		}
		for x := 0; x <= w; x++ {
			gx0 := (x / step) * step
			gx1 := gx0 + step
			if gx1 > w {
				gx1 = w
			}
			fx := 0.0
			if gx1 != gx0 {
				fx = float64(x-gx0) / float64(gx1-gx0)
			}
			h00 := g.H[gy0*g.N+gx0]
			h10 := g.H[gy0*g.N+gx1]
			h01 := g.H[gy1*g.N+gx0]
			h11 := g.H[gy1*g.N+gx1]
			top := h00 + (h10-h00)*fx
			bot := h01 + (h11-h01)*fx
			interp := top + (bot-top)*fy
			d := math.Abs(g.H[y*g.N+x] - interp)
			if d > eMax {
				eMax = d
			}
			sumSq += d * d
			n++
		}
	}
	return eMax, math.Sqrt(sumSq / float64(n))
}

// SelectLevel 返回误差不超过 threshold 的最粗二分级层级。
// 第 0 级误差恒为 0，所以任何非负阈值都能选到至少 0 级。
func SelectLevel(errMaxByLevel []float64, threshold float64) int {
	chosen := 0
	for l, e := range errMaxByLevel {
		if e <= threshold {
			chosen = l
		}
	}
	return chosen
}
