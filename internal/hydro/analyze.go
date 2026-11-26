package hydro

import "sort"

// EventKind 是水位扫描中一个台地入水时的临界点类型。
type EventKind int

const (
	// KindRegular 普通台地：入水时接到恰好一个更低水域分量，分量数不变。
	KindRegular EventKind = iota
	// KindMinimum 局部极小台地：入水时没有任何更低邻格，分量数 +1。
	KindMinimum
	// KindSaddle 合并台地（离散鞍点）：入水时接上 k>=2 个不同的更低分量，分量数减少 k-1。
	KindSaddle
)

func (k EventKind) String() string {
	switch k {
	case KindMinimum:
		return "minimum"
	case KindSaddle:
		return "saddle"
	default:
		return "regular"
	}
}

// Event 是按高度扫描得到的一个临界事件，对应一个同高台地（四连通等高峰）。
// 无同高时每个台地恰好一个格点，事件即逐格点事件。
type Event struct {
	Y, X   int // 台地内行主序最小的代表格点
	Height float64
	Kind   EventKind
	// Cells 是台地包含的格点索引。
	Cells []int
	// K 是台地入水前，四邻域中严格更低邻格所属的不同水域分量数。
	// 极小台地 K=0（Δ=+1），普通台地 K=1（Δ=0），鞍点台地 K>=2（Δ=1-K<=-1）。
	K int
	// Delta 是台地入水导致的分量数增量，恒等于 1-K。
	Delta int
	// Order 是全局扫描序号（高度升序，同高按代表格行主序），从 1 开始。
	Order int
	// GeomSaddle 是代表格点的八邻域几何判定鞍点（仅作对照，不参与分量预测）。
	GeomSaddle bool
	// LocalMax 是代表格点的四邻域严格局部极大（仅作对照）。
	LocalMax bool
	// Pattern 按 U D L R 给出代表格点四邻格高低：'<' 更低 '>' 更高 '=' 同高 '#' 网格外。
	Pattern string
}

// HeightRecord 是每一个不同高度档的扫描结果。
type HeightRecord struct {
	Height     float64
	Components int // 该高度处理完后的水域连通分量数（扫描预测值）
	Births     int // 本高度档的极小台地数
	Merges     int // 本高度档的鞍点台地数
	Regular    int
	Delta      int // 分量数净变化 = Births - Σ(K-1)
	Events     []*Event
}

// Analysis 是整张图的扫描结果。
type Analysis struct {
	Size int
	// Records 按高度升序，每个不同高度一条。
	Records []*HeightRecord
	Events  []*Event
	// Minima/Saddles/Regular 是各类事件（台地）总数。
	Minima, Saddles, Regular int
	// EqualNeighborPairs 是四邻域同高相邻对数（>0 说明存在台地）。
	EqualNeighborPairs int
}

var dirs4 = [4][2]int{{-1, 0}, {1, 0}, {0, -1}, {0, 1}} // U D L R

// dirs8Ring 按顺时针环序排列八个邻格方向，用于几何鞍点的环符号判定。
var dirs8Ring = [8][2]int{
	{-1, -1}, {-1, 0}, {-1, 1}, {0, 1},
	{1, 1}, {1, 0}, {1, -1}, {0, -1},
}

func (a *Analysis) idx(y, x int) int { return y*a.Size + x }

func (a *Analysis) in(y, x int) bool {
	return y >= 0 && y < a.Size && x >= 0 && x < a.Size
}

// Analyze 对高度图做一次按高度升序的并查集扫描。
//
// 事件以同高台地（四连通等高峰）为单位：处理高度 h 时，所有 h 台地同时激活。
// 每个台地先统计严格更低邻格所属的不同分量根集合 L（|L|=K），再整体并入并查集：
// 台地自身连通后逐一对接 K 个根，分量数变化恰为 1-K。
// 同高度的不同台地互不四邻接（否则就是同一个台地），故同档台地之间相互独立。
func Analyze(size int, h []float64) *Analysis {
	n := size * size
	an := &Analysis{Size: size}

	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		ia, ib := order[a], order[b]
		if h[ia] != h[ib] {
			return h[ia] < h[ib]
		}
		return ia < ib // 同高按行主序，保证确定性
	})

	active := make([]bool, n)
	d := newDSU(n)

	equalPairs := 0
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			if x+1 < size && h[an.idx(y, x)] == h[an.idx(y, x+1)] {
				equalPairs++
			}
			if y+1 < size && h[an.idx(y, x)] == h[an.idx(y+1, x)] {
				equalPairs++
			}
		}
	}
	an.EqualNeighborPairs = equalPairs

	plateauID := make([]int, n)
	for i := range plateauID {
		plateauID[i] = -1
	}

	for i := 0; i < n; {
		level := h[order[i]]
		j := i
		for j < n && h[order[j]] == level {
			j++
		}
		batch := order[i:j]
		inBatch := make([]bool, n)
		for _, v := range batch {
			inBatch[v] = true
		}

		// 在本批格点上求四连通台地（洪水填充）。
		type plateau struct{ cells []int }
		var plateaus []plateau
		for _, seed := range batch {
			if plateauID[seed] != -1 {
				continue
			}
			id := len(plateaus)
			stack := []int{seed}
			plateauID[seed] = id
			cells := []int{seed}
			for len(stack) > 0 {
				v := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				y, x := v/size, v%size
				for _, q := range dirs4 {
					ny, nx := y+q[0], x+q[1]
					if !an.in(ny, nx) {
						continue
					}
					w := an.idx(ny, nx)
					if inBatch[w] && plateauID[w] == -1 {
						plateauID[w] = id
						stack = append(stack, w)
						cells = append(cells, w)
					}
				}
			}
			sort.Ints(cells)
			plateaus = append(plateaus, plateau{cells: cells})
		}

		rec := &HeightRecord{Height: level}
		before := d.ncomp

		for _, p := range plateaus {
			rep := p.cells[0]
			ry, rx := rep/size, rep%size

			// 统计严格更低邻格的不同分量根；同高邻格必在本台地内，不会出现在根集合里。
			roots := map[int]struct{}{}
			for _, v := range p.cells {
				y, x := v/size, v%size
				for _, q := range dirs4 {
					ny, nx := y+q[0], x+q[1]
					if !an.in(ny, nx) {
						continue
					}
					w := an.idx(ny, nx)
					if active[w] { // 已激活意味着高度严格更低（同高格点此刻尚未激活）
						roots[d.find(w)] = struct{}{}
					}
				}
			}
			k := len(roots)

			// 台地整体激活：首格建分量，其余格并入；随后对接 k 个低分量根。
			head := p.cells[0]
			d.add(head)
			active[head] = true
			for _, v := range p.cells[1:] {
				d.add(v)
				active[v] = true
				d.union(head, v)
			}
			for r := range roots {
				d.union(head, r)
			}

			kind := KindRegular
			switch {
			case k == 0:
				kind = KindMinimum
			case k >= 2:
				kind = KindSaddle
			}
			ev := &Event{
				Y: ry, X: rx, Height: level, Kind: kind, Cells: p.cells,
				K: k, Delta: 1 - k, Pattern: an.pattern(ry, rx, h),
			}
			ev.GeomSaddle = an.geomSaddle(ry, rx, h)
			ev.LocalMax = an.localMax4(ry, rx, h)
			an.Events = append(an.Events, ev)
			rec.Events = append(rec.Events, ev)
		}

		for _, ev := range rec.Events {
			switch ev.Kind {
			case KindMinimum:
				rec.Births++
			case KindSaddle:
				rec.Merges++
			default:
				rec.Regular++
			}
		}
		rec.Components = d.ncomp
		rec.Delta = d.ncomp - before
		an.Records = append(an.Records, rec)
		i = j
	}

	for idx := range an.Events {
		an.Events[idx].Order = idx + 1
		switch an.Events[idx].Kind {
		case KindMinimum:
			an.Minima++
		case KindSaddle:
			an.Saddles++
		default:
			an.Regular++
		}
	}
	return an
}

func (a *Analysis) pattern(y, x int, h []float64) string {
	b := make([]byte, 4)
	for i, q := range dirs4 {
		ny, nx := y+q[0], x+q[1]
		if !a.in(ny, nx) {
			b[i] = '#'
			continue
		}
		switch nh := h[a.idx(ny, nx)]; {
		case nh < h[a.idx(y, x)]:
			b[i] = '<'
		case nh > h[a.idx(y, x)]:
			b[i] = '>'
		default:
			b[i] = '='
		}
	}
	return string(b)
}

// localMax4：四邻域都存在且全部严格低于本点。
func (a *Analysis) localMax4(y, x int, h []float64) bool {
	cur := h[a.idx(y, x)]
	for _, q := range dirs4 {
		ny, nx := y+q[0], x+q[1]
		if !a.in(ny, nx) {
			return false
		}
		if h[a.idx(ny, nx)] >= cur {
			return false
		}
	}
	return true
}

// geomSaddle 是八邻域环上的几何鞍点判定：沿顺时针邻环（网格外跳过，环在
// 有效邻格上闭合）读严格低(-1)/高(+1)符号，符号沿环交替变化即判为鞍点——
// 内点八邻环要求 4 次变化（低、高各两段分离的弧）；边界点有效邻格更少，
// 小环要求 2 次变化。同高邻格使环符号无定义，判否。
// 这是连续曲面鞍点的离散近似，仅作对照；分量预测只认扫描得到的
// KindSaddle（台地 K>=2），两者口径不同、不等价。
func (a *Analysis) geomSaddle(y, x int, h []float64) bool {
	cur := h[a.idx(y, x)]
	boundary := false
	ring := make([]int8, 0, 8)
	for _, q := range dirs8Ring {
		ny, nx := y+q[0], x+q[1]
		if !a.in(ny, nx) {
			boundary = true
			continue // 网格外：不进环
		}
		v := h[a.idx(ny, nx)]
		switch {
		case v < cur:
			ring = append(ring, -1)
		case v > cur:
			ring = append(ring, +1)
		default:
			return false // 同高邻格：环符号无定义
		}
	}
	n := len(ring)
	want := 4
	if boundary {
		want = 2
	}
	if n < want {
		return false
	}
	changes := 0
	for i := 0; i < n; i++ {
		if ring[i] != ring[(i+1)%n] {
			changes++
		}
	}
	return changes == want
}
