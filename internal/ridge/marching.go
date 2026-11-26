package ridge

import (
	"math"
	"sort"
)

// Segment 是裁剪后脊线折线的一段，只连接两个顶点。
type Segment struct{ A, B int }

// Curve 是一条连通脊线段。
//
// Vertex 是折线顶点下标；Closed 表示首尾相接；
// Endpoints 是度为 1 的顶点坐标（开曲线必为 2 个）。
type Curve struct {
	Vertex    []int
	Closed    bool
	Endpoints [][2]float64
}

// Ridge 是 G=0 等值线裁剪后的全部脊线段。
type Ridge struct {
	Points   [][2]float64
	Segments []Segment
	Curves   []Curve
}

// CompareMode 决定 G=0 等值线保留哪些部分。
type CompareMode int

const (
	// ModeMasked 只保留落在任务书判据 Mask 内的线段。
	ModeMasked CompareMode = iota
	// ModeCrest 只保留满足 L1<0（横截面凹，即脊而非谷）的线段。
	ModeCrest
	// ModeAll 不裁剪：输出全部 G=0 等值线（脊与谷都在内）。
	ModeAll
)

// Extract 用 marching squares 提取 G=0 等值线，并按 mode 裁剪连通段。
//
// 模糊格（四角符号二二对半）用格心 G 的平均值消歧；这一步只影响
// 拓扑连接，不改变“线段中点是否通过裁剪”的判断。
//
// 每个格边至多产生一个顶点，且每格贡献 0 或 2 条线段，因此结果图
// 每个顶点度数 ≤ 2：只可能是开链或闭环。
func (f *Field) Extract(mode CompareMode) Ridge {
	n := f.N
	nodes := make(map[[2]int]int)
	var pts [][2]float64
	var segs []Segment

	// 水平边 (x0,y0)-(x0+1,y0) 用 kind=0，垂直边 (x0,y0)-(x0,y0+1) 用 kind=1。
	edgeKey := func(kind, x0, y0 int) [2]int { return [2]int{kind, y0*n + x0} }
	node := func(key [2]int, g0, g1 float64) int {
		if id, ok := nodes[key]; ok {
			return id
		}
		t := 0.0
		if g0 != g1 {
			t = g0 / (g0 - g1)
		}
		if t < 0 {
			t = 0
		}
		if t > 1 {
			t = 1
		}
		x0, y0 := key[1]%n, key[1]/n
		if key[0] == 0 {
			pts = append(pts, [2]float64{float64(x0) + t, float64(y0)})
		} else {
			pts = append(pts, [2]float64{float64(x0), float64(y0) + t})
		}
		id := len(pts) - 1
		nodes[key] = id
		return id
	}

	for y := 0; y < n-1; y++ {
		for x := 0; x < n-1; x++ {
			a := f.G[y*n+x]       // 左下 (x,y)
			b := f.G[y*n+x+1]     // 右下 (x+1,y)
			c := f.G[(y+1)*n+x+1] // 右上 (x+1,y+1)
			d := f.G[(y+1)*n+x]   // 左上 (x,y+1)

			// 依次记录四条边上的交点；-1 表示无交点。
			var bottom, right, top, left int = -1, -1, -1, -1
			if (a < 0) != (b < 0) {
				bottom = node(edgeKey(0, x, y), a, b)
			}
			if (b < 0) != (c < 0) {
				right = node(edgeKey(1, x+1, y), b, c)
			}
			if (d < 0) != (c < 0) {
				top = node(edgeKey(0, x, y+1), d, c)
			}
			if (a < 0) != (d < 0) {
				left = node(edgeKey(1, x, y), a, d)
			}

			var cross []int
			if bottom >= 0 {
				cross = append(cross, bottom)
			}
			if right >= 0 {
				cross = append(cross, right)
			}
			if top >= 0 {
				cross = append(cross, top)
			}
			if left >= 0 {
				cross = append(cross, left)
			}

			switch len(cross) {
			case 0:
				// 无交点。
			case 2:
				// 恰好两个交点，直接相连。
				segs = append(segs, Segment{cross[0], cross[1]})
			case 4:
				// 模糊格：格心符号与被 a 同号的负角点一致时，
				// 负角点连通，等值线隔离另外两个正角点。
				ctr := 0.25 * (a + b + c + d)
				if (a < 0) == (ctr < 0) {
					// 隔离正角点 b（右下）与 d（左上）。
					segs = append(segs, Segment{bottom, right})
					segs = append(segs, Segment{top, left})
				} else {
					// 隔离负角点 a（左下）与 c（右上）。
					segs = append(segs, Segment{bottom, left})
					segs = append(segs, Segment{right, top})
				}
			}
		}
	}

	// 按线段中点做裁剪。
	kept := make([]Segment, 0, len(segs))
	for _, s := range segs {
		p, q := pts[s.A], pts[s.B]
		mx := 0.5 * (p[0] + q[0])
		my := 0.5 * (p[1] + q[1])
		ok := false
		switch mode {
		case ModeMasked:
			l1 := f.Bilinear(f.L1, mx, my)
			l2 := f.Bilinear(f.L2, mx, my)
			ok = l1 < 0 && math.Abs(l1) > math.Abs(l2)
		case ModeCrest:
			ok = f.Bilinear(f.L1, mx, my) < 0
		case ModeAll:
			ok = true
		}
		if ok {
			kept = append(kept, s)
		}
	}
	return Ridge{Points: pts, Segments: kept, Curves: buildCurves(pts, kept)}
}

func buildCurves(pts [][2]float64, segs []Segment) []Curve {
	adj := make(map[int][]int, len(pts))
	for _, s := range segs {
		adj[s.A] = append(adj[s.A], s.B)
		adj[s.B] = append(adj[s.B], s.A)
	}
	seen := make(map[int]bool, len(pts))
	ids := make([]int, 0, len(adj))
	for id := range adj {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	var curves []Curve
	for _, start := range ids {
		if seen[start] {
			continue
		}
		queue := []int{start}
		seen[start] = true
		var comp []int
		for len(queue) > 0 {
			v := queue[0]
			queue = queue[1:]
			comp = append(comp, v)
			for _, w := range adj[v] {
				if !seen[w] {
					seen[w] = true
					queue = append(queue, w)
				}
			}
		}
		sort.Ints(comp)
		c := Curve{Vertex: comp}
		for _, v := range comp {
			if len(adj[v]) == 1 {
				c.Endpoints = append(c.Endpoints, pts[v])
			}
		}
		if len(c.Endpoints) == 0 {
			c.Closed = true
		}
		curves = append(curves, c)
	}
	return curves
}

// Degree 返回顶点度数，用于自检脊线图是否处处度 ≤2。
func (r Ridge) Degree() map[int]int {
	deg := make(map[int]int, len(r.Points))
	for _, s := range r.Segments {
		deg[s.A]++
		deg[s.B]++
	}
	return deg
}

// ExtractFull 返回未经裁剪的全部 G=0 等值线（脊与谷）。
func (f *Field) ExtractFull() Ridge { return f.Extract(ModeAll) }
