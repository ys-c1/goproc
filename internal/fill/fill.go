// Package fill 做洼地填平与水面分析。
//
// 数学依据见 docs/fill-proof.md：
//  1. 填洼后每片水面的高度只由出口格（与水体八邻接的非水格）高度的最小值决定；
//  2. 已知溢流高度 S，水面就是 {G < S} 中含洼底的四连通分量，
//     水面像素数与最大水深（S 减分量最小灰度）可直接读出，无需迭代填洼。
//
// 排水与溢流按八邻域（含对角），水面像素的连通统计按四邻域。
package fill

import (
	"container/heap"
	"fmt"
)

// Grid 是八位灰度高度场，行主序，值域零到二五五。
type Grid struct {
	Size int
	G    []int
}

// At 取某点灰度。
func (g *Grid) At(x, y int) int { return g.G[y*g.Size+x] }

// neighbors8 对八邻域（含对角）的每个合法坐标调用 fn。
func neighbors8(size, x, y int, fn func(nx, ny int)) {
	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			if dx == 0 && dy == 0 {
				continue
			}
			nx, ny := x+dx, y+dy
			if nx < 0 || ny < 0 || nx >= size || ny >= size {
				continue
			}
			fn(nx, ny)
		}
	}
}

type cell struct {
	level int
	idx   int
}

type levelHeap []cell

func (h levelHeap) Len() int            { return len(h) }
func (h levelHeap) Less(i, j int) bool  { return h[i].level < h[j].level }
func (h levelHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *levelHeap) Push(x interface{}) { *h = append(*h, x.(cell)) }
func (h *levelHeap) Pop() interface{} {
	old := *h
	n := len(old)
	c := old[n-1]
	*h = old[:n-1]
	return c
}

// PriorityFlood 从边界单遍优先泛滥，返回每格填洼水位 W，八邻域。
// W(c) = min_{P: c→边界} max_{p∈P} G(p)，即最小最大瓶颈高度；
// 每格恰好入堆出堆一次，不是迭代松弛。
func PriorityFlood(g *Grid) []int {
	size := g.Size
	w := make([]int, size*size)
	seen := make([]bool, size*size)
	pq := &levelHeap{}
	push := func(x, y, level int) {
		i := y*size + x
		if seen[i] {
			return
		}
		seen[i] = true
		w[i] = level
		heap.Push(pq, cell{level: level, idx: i})
	}
	for x := 0; x < size; x++ {
		push(x, 0, g.At(x, 0))
		push(x, size-1, g.At(x, size-1))
	}
	for y := 1; y < size-1; y++ {
		push(0, y, g.At(0, y))
		push(size-1, y, g.At(size-1, y))
	}
	for pq.Len() > 0 {
		c := heap.Pop(pq).(cell)
		x, y := c.idx%size, c.idx/size
		neighbors8(size, x, y, func(nx, ny int) {
			lv := g.At(nx, ny)
			if c.level > lv {
				lv = c.level
			}
			push(nx, ny, lv)
		})
	}
	return w
}

// IterativeFill 迭代松弛填洼，八邻域，作为优先泛滥的对照。
// 反复执行 W(c) = max(G(c), min_邻居 W) 直到收敛，边界格固定为 G。
func IterativeFill(g *Grid) []int {
	size := g.Size
	const inf = int(^uint(0) >> 2)
	w := make([]int, size*size)
	onBorder := func(x, y int) bool { return x == 0 || y == 0 || x == size-1 || y == size-1 }
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			if onBorder(x, y) {
				w[y*size+x] = g.At(x, y)
			} else {
				w[y*size+x] = inf
			}
		}
	}
	for changed := true; changed; {
		changed = false
		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				if onBorder(x, y) {
					continue
				}
				m := inf
				neighbors8(size, x, y, func(nx, ny int) {
					if v := w[ny*size+nx]; v < m {
						m = v
					}
				})
				v := g.At(x, y)
				if m > v {
					v = m
				}
				i := y*size + x
				if v < w[i] {
					w[i] = v
					changed = true
				}
			}
		}
	}
	return w
}

// WaterMask 标出被填高的格，即水面像素。
func WaterMask(g *Grid, w []int) []bool {
	m := make([]bool, len(w))
	for i := range w {
		m[i] = w[i] > g.G[i]
	}
	return m
}

// Components4 对掩膜做四邻域连通标号。
// 返回标号图（非掩膜格为 -1）与每块像素数。
func Components4(size int, mask []bool) ([]int, []int) {
	labels := make([]int, size*size)
	for i := range labels {
		labels[i] = -1
	}
	var sizes []int
	dirs := [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	for start := range mask {
		if !mask[start] || labels[start] >= 0 {
			continue
		}
		label := len(sizes)
		n := 0
		queue := []int{start}
		labels[start] = label
		for len(queue) > 0 {
			i := queue[0]
			queue = queue[1:]
			n++
			x, y := i%size, i/size
			for _, d := range dirs {
				nx, ny := x+d[0], y+d[1]
				if nx < 0 || ny < 0 || nx >= size || ny >= size {
					continue
				}
				j := ny*size + nx
				if mask[j] && labels[j] < 0 {
					labels[j] = label
					queue = append(queue, j)
				}
			}
		}
		sizes = append(sizes, n)
	}
	return labels, sizes
}

// Lake 是一片水面：填洼掩膜的四连通分量。
type Lake struct {
	Label     int      // 连通标号
	Level     int      // 水面高度 S（灰度）
	Pixels    int      // 水面像素数（四连通）
	MinGround int      // 洼底灰度
	MaxDepth  int      // 最大水深 S-MinGround（灰度差）
	OutletMin int      // 出口格高度的最小值（八邻域，含对角）
	Outlets   [][2]int // 高度等于 S 的出口格坐标
	Cells     []int    // 水面像素下标，行主序
}

// AnalyzeLakes 由水位场 W 提取全部水面，并核验命题1：
// 水位在每片湖上为常数，且等于出口格高度的最小值。
func AnalyzeLakes(g *Grid, w []int) ([]Lake, error) {
	size := g.Size
	mask := WaterMask(g, w)
	labels, sizes := Components4(size, mask)
	lakes := make([]Lake, 0, len(sizes))
	for label := range sizes {
		lk := Lake{Label: label}
		for i, l := range labels {
			if l == label {
				lk.Cells = append(lk.Cells, i)
			}
		}
		lk.Level = w[lk.Cells[0]]
		lk.MinGround = g.G[lk.Cells[0]]
		for _, i := range lk.Cells {
			if w[i] != lk.Level {
				return nil, fmt.Errorf("湖 %d 水位非常数: 格(%d,%d) W=%d 应为 %d",
					label, i%size, i/size, w[i], lk.Level)
			}
			if g.G[i] < lk.MinGround {
				lk.MinGround = g.G[i]
			}
		}
		lk.Pixels = len(lk.Cells)
		lk.MaxDepth = lk.Level - lk.MinGround

		// 溢流水体：含本湖、水位同为 S 的水格八连通簇。
		inCluster := make([]bool, size*size)
		queue := append([]int(nil), lk.Cells...)
		for _, i := range lk.Cells {
			inCluster[i] = true
		}
		for len(queue) > 0 {
			i := queue[0]
			queue = queue[1:]
			x, y := i%size, i/size
			neighbors8(size, x, y, func(nx, ny int) {
				j := ny*size + nx
				if !inCluster[j] && mask[j] && w[j] == lk.Level {
					inCluster[j] = true
					queue = append(queue, j)
				}
			})
		}
		// 出口格：与溢流水体八邻接的非水格。
		lk.OutletMin = -1
		seenOutlet := make([]bool, size*size)
		for i := range inCluster {
			if !inCluster[i] {
				continue
			}
			x, y := i%size, i/size
			neighbors8(size, x, y, func(nx, ny int) {
				j := ny*size + nx
				if inCluster[j] || mask[j] || seenOutlet[j] {
					return
				}
				seenOutlet[j] = true
				if lk.OutletMin < 0 || g.G[j] < lk.OutletMin {
					lk.OutletMin = g.G[j]
				}
				if g.G[j] == lk.Level {
					lk.Outlets = append(lk.Outlets, [2]int{nx, ny})
				}
			})
		}
		lakes = append(lakes, lk)
	}
	return lakes, nil
}

// PredictLake 按命题2 预判水面：{G < S} 中含种子格的四连通分量。
// 只用高度场 G 与溢流高度 S，不碰任何填洼结果。
// 返回分量像素与分量最小灰度（洼底），最大水深即 S 减洼底。
func PredictLake(g *Grid, seed, s int) (cells []int, minGround int) {
	size := g.Size
	seen := make([]bool, size*size)
	dirs := [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}
	queue := []int{seed}
	seen[seed] = true
	minGround = g.G[seed]
	for len(queue) > 0 {
		i := queue[0]
		queue = queue[1:]
		cells = append(cells, i)
		if g.G[i] < minGround {
			minGround = g.G[i]
		}
		x, y := i%size, i/size
		for _, d := range dirs {
			nx, ny := x+d[0], y+d[1]
			if nx < 0 || ny < 0 || nx >= size || ny >= size {
				continue
			}
			j := ny*size + nx
			if !seen[j] && g.G[j] < s {
				seen[j] = true
				queue = append(queue, j)
			}
		}
	}
	return cells, minGround
}
