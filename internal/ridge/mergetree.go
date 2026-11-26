package ridge

import "sort"

// Saddle 是一次合并事件：高到低扫描时，新格点同时接通 arity 个已存在的
// 连通分量。
type Saddle struct {
	X, Y  int
	Arity int
}

// Arc 是一条上升脊弧：从鞍点格出发，沿最陡上升路径终止于一个局部极大点。
//
// Path 按顺序给出经过的格点（含鞍点与终点），用于叠加图绘制。
type Arc struct {
	FromX, FromY int
	ToX, ToY     int
	Path         [][2]int
}

// JoinTree 是高度图的上升合并树。
//
// P 是成分创建次数，等于严格局部极大点数；
// Arcs 的长度等于 Σ arity，也就是上升脊弧的总数。
type JoinTree struct {
	P        int
	Saddles  []Saddle
	Arcs     []Arc
	Arities  map[int]int // arity -> 次数
	TieCount int         // 等高相邻的格点对数（非通用情形，仅作告警）
}

// BuildJoinTree 用并查集按高度从高到低扫描构建上升合并树。
//
// 计数恒等式（docs/ridges.md 定理 3）：
//
//	P - 1 = Σ_saddles (arity - 1)
//	|Arcs| = Σ arity = P + S - 1
//
// 当所有合并都是二元（arity=2）时 S = P-1，于是 |Arcs| = 2P-2。
func (f *Field) BuildJoinTree() JoinTree {
	n := f.N
	m := n * n
	order := make([]int, m)
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		return f.F[order[a]] > f.F[order[b]]
	})

	parent := make([]int, m)
	added := make([]bool, m)
	for i := range parent {
		parent[i] = i
	}
	var find func(int) int
	find = func(x int) int {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}

	// 最陡上升指针：指向严格更高的邻点中最高者。
	ptr := make([]int, m)
	for i := range ptr {
		ptr[i] = -1
	}
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			best := -1
			bv := f.F[y*n+x]
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					xx, yy := x+dx, y+dy
					if xx < 0 || yy < 0 || xx >= n || yy >= n {
						continue
					}
					w := f.F[yy*n+xx]
					if w > bv {
						bv = w
						best = yy*n + xx
					}
				}
			}
			ptr[y*n+x] = best
		}
	}
	ascentRoot := func(i int) int {
		for ptr[i] >= 0 {
			i = ptr[i]
		}
		return i
	}
	// ascentPath 返回从 i 出发沿最陡上升路径经过的格点。
	// i 本身指向严格更高的邻点，因此 ptr[i] 为 -1 时 i 就是局部极大点。
	ascentPath := func(i int) [][2]int {
		var p [][2]int
		for {
			p = append(p, [2]int{i % n, i / n})
			if ptr[i] < 0 {
				return p
			}
			i = ptr[i]
		}
	}

	t := JoinTree{Arities: make(map[int]int)}
	for _, idx := range order {
		x, y := idx%n, idx/n
		added[idx] = true
		rootRep := make(map[int]int) // 根 -> 一个代表邻点
		for dy := -1; dy <= 1; dy++ {
			for dx := -1; dx <= 1; dx++ {
				if dx == 0 && dy == 0 {
					continue
				}
				xx, yy := x+dx, y+dy
				if xx < 0 || yy < 0 || xx >= n || yy >= n {
					continue
				}
				j := yy*n + xx
				if !added[j] {
					continue
				}
				r := find(j)
				if _, ok := rootRep[r]; !ok {
					rootRep[r] = j
				}
			}
		}
		if len(rootRep) == 0 {
			t.P++
			continue
		}
		if len(rootRep) == 1 {
			// 直接接到已有分量上：这个格点是通向某个极大点的坡面格。
			for r := range rootRep {
				parent[idx] = r
			}
			continue
		}
		// 合并：新格点成为已存在各根的父节点，它就是本棵弧树的鞍点。
		// 按格点下标排序，保证鞍点与弧的报告顺序稳定。
		reps := make([]int, 0, len(rootRep))
		for r := range rootRep {
			reps = append(reps, r)
		}
		sort.Ints(reps)
		for _, r := range reps {
			parent[r] = idx
		}
		parent[idx] = idx
		arity := len(reps)
		t.Saddles = append(t.Saddles, Saddle{X: x, Y: y, Arity: arity})
		t.Arities[arity]++
		// 每个被合并的分量贡献一条上升弧：从鞍点格出发，经过该分量中
		// 与鞍点相接的代表邻点，再沿最陡上升路径到局部极大点。
		for _, rep := range reps {
			root := ascentRoot(rep)
			path := ascentPath(rep)
			path = append([][2]int{{x, y}}, path...)
			t.Arcs = append(t.Arcs, Arc{
				FromX: x, FromY: y,
				ToX: root % n, ToY: root / n,
				Path: path,
			})
		}
	}

	// 统计等高相邻对，用于判断数据是否满足通用性假设。
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			v := f.F[y*n+x]
			if x+1 < n && f.F[y*n+x+1] == v {
				t.TieCount++
			}
			if y+1 < n && f.F[(y+1)*n+x] == v {
				t.TieCount++
			}
		}
	}
	return t
}
