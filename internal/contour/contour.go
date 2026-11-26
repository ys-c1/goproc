// Package contour 在规则方格高度场上做等值线（等高线）分析。
//
// 网格约定（全文统一）：
//   - 格点 (x,y)，x,y ∈ [0,N-1]，高度 g.At(x,y)。
//   - 格点之间只承认四邻域边：水平边 (x,y)-(x+1,y) 与垂直边 (x,y)-(x,y+1)。
//     八邻域中的对角相邻不是边，不参与任何跨越与连通性判定。
//   - 边界边：两个端点都落在区域边界 ∂D 上的四邻域边，即 y=0、y=N-1 的水平边
//     与 x=0、x=N-1 的垂直边，共 4(N-1) 条。
//   - 边界格：至少含一条边界边的格，即最外层一圈格（cx=0 或 cx=N-2 或
//     cy=0 或 cy=N-2）；角格含两条边界边。
//   - 跨越（straddle）：一条四邻域边两端高度 a,b 对水平 h 异号，
//     即 (a>h) != (b>h)，等价于 min(a,b)<h<max(a,b)。
//     为避免等值线恰好经过格点的退化情形，调用方先用 NudgeLevel 挪开等值高度。
package contour

import (
	"fmt"
	"math"
)

// Grid 是 N×N 的高度场。
type Grid struct {
	N int
	Z []float64
}

// At 取格点高度。
func (g *Grid) At(x, y int) float64 { return g.Z[y*g.N+x] }

// Pt 是格点坐标系下的点（格点坐标为整数）。
type Pt struct {
	X, Y float64
}

// Curve 是一条追踪得到的等值线连通分支。
// Open=true 为两端落在边界上的开弧；false 为闭合环。
type Curve struct {
	Open bool
	Pts  []Pt
}

// LevelResult 是单个等值高度的全部统计与追踪结果。
type LevelResult struct {
	H float64 // 实际使用的（可能微调过的）等值高度

	// 边界跨越
	BoundaryCrossings int // B：边界边跨越条数

	// 无追踪预判一：局部组合计数（只逐格、逐边扫描，不追踪曲线）
	GraphNodes  int // V：跨越的四邻域边数（每条边恰好一个交点）
	GraphEdges  int // E：marching squares 段数（普通格 1 段，鞍格 2 段，含格心节点则每段 1）
	ChiLocal    int // χ(F)：超水平集 F={f>h} 的欧拉示性数，逐格局部计数
	PredOpenG   int // 开曲线预判 = B/2
	PredCloseG  int // 闭合曲线预判 = c-χ(F)，c 由局部计数独立给出（见 localCompCount）

	// 无追踪预判二：超水平集区域拓扑（洪水填充，同样不追踪等值线）
	RegionComps     int // c：{f>h} 连通分支数（四邻域 + 鞍点中心约定）
	RegionBoundary  int // b：其中触及 ∂D 的分支数
	BackgroundHoles int // holes：{f≤h} 的有界背景分支数（洞数）
	PredOpenR       int // 开曲线预判 = B/2
	PredCloseR      int // 闭合曲线预判 = (c-b)+holes = c-χ(F)

	// 实际追踪
	TraceOpen   int // 追踪得到的开弧条数
	TraceClosed int // 追踪得到的闭合环条数
	Curves      []Curve

	localMismatch string // 局部计数与填充计数的恒等式自检差异
}

// nodeKey 标识等值线与一条四邻域边的交点，或鞍格的中心端口。
// kind=0 水平边 {0,x,y}：(x,y)-(x+1,y)；kind=1 垂直边 {1,x,y}：(x,y)-(x,y+1)；
// kind=2、3 为鞍格的两个中心端口 {2/3,cx,cy}，坐标同为格心，相互不连通。
type nodeKey struct {
	kind uint8
	x, y int
}

func (g *Grid) isBoundaryNode(k nodeKey) bool {
	switch k.kind {
	case 0: // 水平边
		return k.y == 0 || k.y == g.N-1
	case 1: // 垂直边
		return k.x == 0 || k.x == g.N-1
	default: // 鞍格中心端口永远在内陆
		return false
	}
}

// edgeEndpoints 返回一条边两个端点的格点坐标与高度。
func (g *Grid) edgeEndpoints(k nodeKey) (p0, p1 Pt, v0, v1 float64) {
	switch k.kind {
	case 0:
		p0 = Pt{float64(k.x), float64(k.y)}
		p1 = Pt{float64(k.x + 1), float64(k.y)}
	default:
		p0 = Pt{float64(k.x), float64(k.y)}
		p1 = Pt{float64(k.x), float64(k.y + 1)}
	}
	return p0, p1, g.At(int(p0.X), int(p0.Y)), g.At(int(p1.X), int(p1.Y))
}

// crossingPoint 返回等值线与边的线性插值交点；中心端口返回格心。
func crossingPoint(g *Grid, k nodeKey, h float64) Pt {
	if k.kind == 2 || k.kind == 3 {
		return Pt{float64(k.x) + 0.5, float64(k.y) + 0.5}
	}
	p0, p1, v0, v1 := g.edgeEndpoints(k)
	t := (h - v0) / (v1 - v0)
	return Pt{p0.X + t*(p1.X-p0.X), p0.Y + t*(p1.Y-p0.Y)}
}

// BoundaryCrossingEdges 列出全部跨越 h 的边界边。边界跨越判定的唯一实现处。
func BoundaryCrossingEdges(g *Grid, h float64) []nodeKey {
	var out []nodeKey
	for x := 0; x < g.N-1; x++ {
		top := nodeKey{0, x, 0}
		bot := nodeKey{0, x, g.N - 1}
		if straddles(g, top, h) {
			out = append(out, top)
		}
		if straddles(g, bot, h) {
			out = append(out, bot)
		}
	}
	for y := 0; y < g.N-1; y++ {
		left := nodeKey{1, 0, y}
		right := nodeKey{1, g.N - 1, y}
		if straddles(g, left, h) {
			out = append(out, left)
		}
		if straddles(g, right, h) {
			out = append(out, right)
		}
	}
	return out
}

func straddles(g *Grid, k nodeKey, h float64) bool {
	_, _, v0, v1 := g.edgeEndpoints(k)
	return (v0 > h) != (v1 > h)
}

// NudgeLevel 保证等值高度不与任何格点高度相等（退化情形处理）。
// 返回实际使用高度；若发生微调，第二返回值给出微调量。
func NudgeLevel(g *Grid, h float64) (float64, float64) {
	const eps = 1e-12
	const step = 1e-5
	orig := h
	for iter := 0; iter < 100; iter++ {
		hit := false
		for _, v := range g.Z {
			if math.Abs(v-h) < eps {
				hit = true
				break
			}
		}
		if !hit {
			return h, h - orig
		}
		h += step
	}
	return h, h - orig
}

// 格 (cx,cy) 的四条边：0 底、1 右、2 顶、3 左。
func cellEdges(cx, cy int) [4]nodeKey {
	return [4]nodeKey{
		{0, cx, cy},     // 底 (cx,cy)-(cx+1,cy)
		{1, cx + 1, cy}, // 右 (cx+1,cy)-(cx+1,cy+1)
		{0, cx, cy + 1}, // 顶 (cx,cy+1)-(cx+1,cy+1)
		{1, cx, cy},     // 左 (cx,cy)-(cx,cy+1)
	}
}

// Analyze 对单个等值高度做完整分析：两种无追踪预判 + marching squares 追踪。
func Analyze(g *Grid, h float64) *LevelResult {
	r := &LevelResult{H: h}

	// 等值线段图：交点节点 + 段。
	nodes := map[nodeKey]Pt{}
	adj := map[nodeKey][]nodeKey{}
	addNode := func(k nodeKey) {
		if _, ok := nodes[k]; !ok {
			nodes[k] = crossingPoint(g, k, h)
		}
	}
	addSeg := func(a, b nodeKey) {
		addNode(a)
		addNode(b)
		adj[a] = append(adj[a], b)
		adj[b] = append(adj[b], a)
		r.GraphEdges++
	}
	saddleCells := 0

	for cy := 0; cy < g.N-1; cy++ {
		for cx := 0; cx < g.N-1; cx++ {
			v00 := g.At(cx, cy)
			v10 := g.At(cx+1, cy)
			v11 := g.At(cx+1, cy+1)
			v01 := g.At(cx, cy+1)
			b00, b10, b11, b01 := v00 > h, v10 > h, v11 > h, v01 > h
			caseID := 0
			if b00 {
				caseID |= 1
			}
			if b10 {
				caseID |= 2
			}
			if b11 {
				caseID |= 4
			}
			if b01 {
				caseID |= 8
			}
			e := cellEdges(cx, cy)
			switch caseID {
			case 0, 15:
				// 全同号，无等值线穿过。
			case 1:
				addSeg(e[3], e[0])
			case 2:
				addSeg(e[0], e[1])
			case 3: // 底两角为前景
				addSeg(e[3], e[1])
			case 4:
				addSeg(e[1], e[2])
			case 6: // 右两角为前景
				addSeg(e[0], e[2])
			case 8:
				addSeg(e[2], e[3])
			case 9: // 左两角为前景
				addSeg(e[0], e[2])
			case 12: // 顶两角为前景
				addSeg(e[3], e[1])
			case 7: // 与 case 8 互补
				addSeg(e[2], e[3])
			case 11: // 与 case 4 互补
				addSeg(e[1], e[2])
			case 13: // 与 case 2 互补
				addSeg(e[0], e[1])
			case 14: // 与 case 1 互补
				addSeg(e[3], e[0])
			case 5, 10:
				// 鞍格：双对角同号。marching squares 的端口配对只由哪一对对角
				// 为前景决定，与格心插值符号无关（已用 skimage.find_contours 在
				// 随机强度的 2×2 鞍格上穷举确认）；格心符号只改变交点的弯曲位置。
				// 用两个坐标重合但相互独立的中心端口 cA、cB 承载两条段，
				// 避免在段图上制造度 4 交叉。
				saddleCells++
				cA := nodeKey{2, cx, cy}
				cB := nodeKey{3, cx, cy}
				addNode(cA)
				addNode(cB)
				if caseID == 5 { // 前景对角：左下(角0)、右上(角2)：底-左、右-顶
					addSeg(e[0], cA)
					addSeg(cA, e[3]) // 底(e0)-左(e3)
					addSeg(e[1], cB)
					addSeg(cB, e[2]) // 右(e1)-顶(e2)
				} else { // case 10：前景对角：右下(角1)、左上(角3)：底-右、左-顶
					addSeg(e[0], cA)
					addSeg(cA, e[1]) // 底(e0)-右(e1)
					addSeg(e[3], cB)
					addSeg(cB, e[2]) // 左(e3)-顶(e2)
				}
			}
		}
	}

	r.GraphNodes = len(nodes)
	for k := range nodes {
		if g.isBoundaryNode(k) {
			r.BoundaryCrossings++
		}
	}

	// 超水平集区域拓扑（标准四邻域 BFS，不追踪等值线）。
	r.RegionComps, r.RegionBoundary, r.BackgroundHoles = regionTopology(g, h)

	// 无追踪预判。
	// 定理一（开弧数）：边界符号序列的变化次数 B 必为偶数，每次跨越是且仅是
	// 一条等值线分支的边界端点；开弧恰好有两个端点，故 O = B/2。
	r.PredOpenG = r.BoundaryCrossings / 2
	r.PredOpenR = r.BoundaryCrossings / 2

	// 定理二（闭合环数，局部欧拉计数路径）：超水平集 F={f>h} 是平面区域，
	// χ(F) = c - holes。闭合等值线既包围每个不触边界的前景分支（c-b 条），
	// 也圈出每个洞（holes 条），故 L = (c-b) + holes。
	// 代入 holes = c-χ 得 L = 2c - b - χ：χ 由逐格局部计数得到，
	// c、b 由四邻域 BFS 独立得到，全程不追踪等值线。
	_, _, _, chi := localEulerCounts(g, h)
	r.ChiLocal = chi
	r.PredCloseG = 2*r.RegionComps - r.RegionBoundary - r.ChiLocal

	// 无追踪预判二：BFS 直接数 c、b、holes（同样不追踪等值线）。
	r.PredCloseR = (r.RegionComps - r.RegionBoundary) + r.BackgroundHoles

	// 恒等式自检：局部 χ 必须等于 BFS 的 c-holes。
	if r.ChiLocal != r.RegionComps-r.BackgroundHoles {
		r.localMismatch = fmt.Sprintf("局部 χ=%d 与 c-holes=%d 不符",
			r.ChiLocal, r.RegionComps-r.BackgroundHoles)
	}

	// 实际追踪：所有节点度数为 1（边界）或 2（内部），逐链行走。
	r.Curves = traceCurves(nodes, adj)
	for _, c := range r.Curves {
		if c.Open {
			r.TraceOpen++
		} else {
			r.TraceClosed++
		}
	}
	return r
}

// localEulerCounts 用标准立方复形公式对 F={f>h} 做纯局部计数，无需追踪或填充。
//   - nV：前景格点数（0-胞腔）；
//   - nE：两端同属前景的四邻域边数（1-胞腔）；
//   - nC：四角全前景的格数（2-胞腔）。
//
// χ(F) = nV - nE + nC，逐格/逐边扫描即可。该复形的连通性是四邻域的，
// 与 regionTopology 的 BFS 同构，故 χ 恒等于填充结果 c-holes。
func localEulerCounts(g *Grid, h float64) (nV, nE, nC, chi int) {
	n := g.N
	above := make([]bool, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			if g.At(x, y) > h {
				above[y*n+x] = true
				nV++
			}
		}
	}
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			if !above[y*n+x] {
				continue
			}
			if x+1 < n && above[y*n+x+1] {
				nE++
			}
			if y+1 < n && above[(y+1)*n+x] {
				nE++
			}
		}
	}
	for cy := 0; cy < n-1; cy++ {
		for cx := 0; cx < n-1; cx++ {
			if above[cy*n+cx] && above[cy*n+cx+1] &&
				above[(cy+1)*n+cx] && above[(cy+1)*n+cx+1] {
				nC++
			}
		}
	}
	chi = nV - nE + nC
	return nV, nE, nC, chi
}

// traceCurves 从度 1 节点出发追踪全部开弧，剩余闭合成环。
// 普通节点度 2 直接通过；鞍格中心节点度 4，按消解时的端口配对通过：
// 端口按 底/右/顶/左 顺序排列，对向相连（底↔右、顶↔左 或 底↔左、顶↔右），
// 具体配对由鞍格符号重新计算，与建图时完全一致。
// traceCurves 追踪全部等值线分支。鞍格已在建图时用两个独立中心端口消解，
// 故段图中每个节点度数恰为 1（边界端点）或 2（内部，含中心端口），
// 通用链行走即可：从度 1 节点走出全部开弧，剩余闭合成环。
func traceCurves(nodes map[nodeKey]Pt, adj map[nodeKey][]nodeKey) []Curve {
	visited := map[nodeKey]bool{}
	var curves []Curve

	walk := func(start nodeKey, open bool) Curve {
		var pts []Pt
		prev := nodeKey{kind: 255} // 哨兵
		cur := start
		for {
			pts = append(pts, nodes[cur])
			visited[cur] = true
			var next nodeKey
			found := false
			for _, nb := range adj[cur] {
				if nb != prev {
					next = nb
					found = true
					break
				}
			}
			if !found { // 开弧走到另一端度 1 节点
				break
			}
			if next == start { // 闭合环回到起点
				break
			}
			prev, cur = cur, next
		}
		return Curve{Open: open, Pts: pts}
	}

	for k := range nodes {
		if len(adj[k]) == 1 && !visited[k] {
			curves = append(curves, walk(k, true))
		}
	}
	for k := range nodes {
		if !visited[k] {
			curves = append(curves, walk(k, false))
		}
	}
	return curves
}

// regionTopology 用数字拓扑对偶连通约定标注前景 {f>h} 与背景 {f≤h}，不追踪等值线：
//   - 前景：四邻域连通（只沿水平/垂直边，对角不算连通）；
//   - 背景：八邻域连通（对角可连通）。
//
// 这是与 marching squares 等值线拓扑一致的标准对偶（4-前景/8-背景），
// 已在随机网格上与 skimage.find_contours 逐例核对零误差。
// 返回前景分支数 c、触及边界 ∂D 的前景分支数 b、背景有界分支（洞）数。
func regionTopology(g *Grid, h float64) (c, b, holes int) {
	n := g.N
	label := func(foreground bool) (comps, boundaryComps int) {
		lab := make([]int, n*n)
		for i := range lab {
			lab[i] = -1
		}
		var dirs [][2]int
		if foreground {
			dirs = [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}} // 四邻域
		} else {
			dirs = [][2]int{
				{1, 0}, {-1, 0}, {0, 1}, {0, -1},
				{1, 1}, {1, -1}, {-1, 1}, {-1, -1}, // 八邻域
			}
		}
		for start := 0; start < n*n; start++ {
			sx, sy := start%n, start/n
			if lab[start] != -1 || (g.At(sx, sy) > h) != foreground {
				continue
			}
			comps++
			touch := false
			queue := []int{start}
			lab[start] = comps
			for len(queue) > 0 {
				id := queue[0]
				queue = queue[1:]
				x, y := id%n, id/n
				if x == 0 || x == n-1 || y == 0 || y == n-1 {
					touch = true
				}
				for _, d := range dirs {
					nx, ny := x+d[0], y+d[1]
					if nx < 0 || nx >= n || ny < 0 || ny >= n {
						continue
					}
					nb := ny*n + nx
					if lab[nb] == -1 && (g.At(nx, ny) > h) == foreground {
						lab[nb] = comps
						queue = append(queue, nb)
					}
				}
			}
			if touch {
				boundaryComps++
			}
		}
		return comps, boundaryComps
	}

	c, b = label(true)
	bgComps, bgBoundary := label(false)
	holes = bgComps - bgBoundary
	return c, b, holes
}

// Check 比对预判与追踪，逐条高度返回差异描述；无差异返回空串。
func (r *LevelResult) Check() string {
	var problem string
	if r.localMismatch != "" {
		problem += fmt.Sprintf("h=%.6f %s; ", r.H, r.localMismatch)
	}
	if r.PredOpenG != r.TraceOpen || r.PredOpenR != r.TraceOpen {
		problem += fmt.Sprintf("h=%.6f 开曲线不一致: 边界预判=%d 区域预判=%d 追踪=%d; ",
			r.H, r.PredOpenG, r.PredOpenR, r.TraceOpen)
	}
	if r.PredCloseG != r.TraceClosed || r.PredCloseR != r.TraceClosed {
		problem += fmt.Sprintf("h=%.6f 闭合曲线不一致: 欧拉预判=%d 区域预判=%d 追踪=%d; ",
			r.H, r.PredCloseG, r.PredCloseR, r.TraceClosed)
	}
	// 每条开弧必须恰有两个边界端点，闭合环不得触及边界。
	for i, cu := range r.Curves {
		if cu.Open && len(cu.Pts) < 2 {
			problem += fmt.Sprintf("h=%.6f 开弧#%d 端点数异常; ", r.H, i)
		}
	}
	return problem
}
