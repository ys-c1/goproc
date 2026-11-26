// Package ridge 在高度图上按横截面曲率的一维极小值检测「曲率脊」像素，
// 并由全图曲率差分分布预判 RidgePixels 的总数，无需逐条追踪脊线。
//
// 重要术语：本包检出的是横截面曲率 κ(x) 的局部极小点，记作「曲率脊」。
// 它不等价于地形意义上的山脊（高度 h(x) 的局部极大点，记作「分水脊」）。
// 两者都是某条一维序列的符号翻转，可以用同一个公式预判，但像素集合不同；
// 差异见 Analysis 中的 Crest 与 Both 字段、以及 docs/ridge.md 的反例。
package ridge

import (
	"math"
	"sort"
)

// Curvature 保存逐像素的横截面曲率估计
//
//	κ(x,y) = h(x-1,y) - 2h(x,y) + h(x+1,y)
//
// 最左、最右各一列缺少完整邻域，置为 NaN，不参与判定。
type Curvature struct {
	Size int
	V    []float64
}

// CurvatureOf 用二阶中心差分估计曲率。data 按行优先存放，长度须为 size*size。
func CurvatureOf(size int, data []float64) Curvature {
	c := Curvature{Size: size, V: make([]float64, size*size)}
	for i := range c.V {
		c.V[i] = math.NaN()
	}
	for y := 0; y < size; y++ {
		row := y * size
		for x := 1; x+1 < size; x++ {
			c.V[row+x] = data[row+x-1] - 2*data[row+x] + data[row+x+1]
		}
	}
	return c
}

// At 取 (x,y) 处的曲率。
func (c Curvature) At(x, y int) float64 { return c.V[y*c.Size+x] }

// Mask 是脊线像素掩码。
type Mask struct {
	Size int
	Data []bool
}

// At 报告 (x,y) 是否为脊线像素。
func (m Mask) At(x, y int) bool { return m.Data[y*m.Size+x] }

// Count 返回掩码中脊线像素数。
func (m Mask) Count() int {
	n := 0
	for _, v := range m.Data {
		if v {
			n++
		}
	}
	return n
}

// Detect 逐行做一维极小值检测。对候选列 x 记
//
//	dL = κ(x) - κ(x-1),  dR = κ(x+1) - κ(x)
//
// 判据为 dL < 0 且 dR > 0，即曲率差分的左右符号翻转。
// 判定列为 x∈[2,Size-3]：两端各两列没有完整的 κ 邻域，不参与判定。
// 返回的 ties 是 dL 或 dR 恰为零的候选数：这些位置不属于严格极小，不计入脊线。
func Detect(c Curvature) (m Mask, ties int) {
	n := c.Size
	m = Mask{Size: n, Data: make([]bool, n*n)}
	if n < 5 {
		return m, 0
	}
	for y := 0; y < n; y++ {
		for x := 2; x <= n-3; x++ {
			dL := c.At(x, y) - c.At(x-1, y)
			dR := c.At(x+1, y) - c.At(x, y)
			if dL == 0 || dR == 0 {
				ties++
			}
			if dL < 0 && dR > 0 {
				m.Data[y*n+x] = true
			}
		}
	}
	return m, ties
}

// DetectCrest 逐行检出「分水脊」：高度在横截面上的严格局部极大。
// 判据为 h(x) > h(x-1) 且 h(x) > h(x+1)，等价于
// Δh(x-1) > 0 且 Δh(x) < 0，即差分序列的符号翻转。
// 判定列与 Detect 相同。
func DetectCrest(size int, data []float64) Mask {
	m := Mask{Size: size, Data: make([]bool, size*size)}
	if size < 5 {
		return m
	}
	for y := 0; y < size; y++ {
		for x := 2; x <= size-3; x++ {
			if data[y*size+x] > data[y*size+x-1] && data[y*size+x] > data[y*size+x+1] {
				m.Data[y*size+x] = true
			}
		}
	}
	return m
}

// Component 是八连通意义下的一个脊线连通段。
type Component struct {
	ID             int
	Pixels         int
	X0, Y0, X1, Y1 int
}

// Label 对掩码做八连通标记，返回逐像素标签（0 为背景），
// 以及按像素数降序排列的连通段清单。段 ID 为发现顺序，从 1 开始。
func Label(m Mask) (labels []int32, comps []Component) {
	n := m.Size
	labels = make([]int32, n*n)
	var stack []int
	for i, on := range m.Data {
		if !on || labels[i] != 0 {
			continue
		}
		id := int32(len(comps) + 1)
		stack = append(stack[:0], i)
		labels[i] = id
		x, y := i%n, i/n
		comp := Component{ID: int(id), X0: x, Y0: y, X1: x, Y1: y}
		for len(stack) > 0 {
			v := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			comp.Pixels++
			vx, vy := v%n, v/n
			if vx < comp.X0 {
				comp.X0 = vx
			}
			if vx > comp.X1 {
				comp.X1 = vx
			}
			if vy < comp.Y0 {
				comp.Y0 = vy
			}
			if vy > comp.Y1 {
				comp.Y1 = vy
			}
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					nx, ny := vx+dx, vy+dy
					if nx < 0 || nx >= n || ny < 0 || ny >= n {
						continue
					}
					j := ny*n + nx
					if m.Data[j] && labels[j] == 0 {
						labels[j] = id
						stack = append(stack, j)
					}
				}
			}
		}
		comps = append(comps, comp)
	}
	sort.Slice(comps, func(a, b int) bool {
		if comps[a].Pixels != comps[b].Pixels {
			return comps[a].Pixels > comps[b].Pixels
		}
		if comps[a].Y0 != comps[b].Y0 {
			return comps[a].Y0 < comps[b].Y0
		}
		return comps[a].X0 < comps[b].X0
	})
	// 按像素数重排 ID，使 ID 顺序即像素数降序。
	remap := make([]int32, len(comps)+1)
	for r, c := range comps {
		remap[c.ID] = int32(r + 1)
	}
	for i, v := range labels {
		if v != 0 {
			labels[i] = remap[v]
		}
	}
	for r := range comps {
		comps[r].ID = r + 1
	}
	return labels, comps
}

// PairStats 是序列一阶差分的分布统计。
type PairStats struct {
	Count     int     // 相邻差分对数量
	Mean      float64 // 差分均值
	Sigma     float64 // 差分标准差
	Skew      float64 // 偏度
	Kurt      float64 // 超额峰度，正态为 0
	VarLeft   float64 // 左差分方差
	VarRight  float64 // 右差分方差
	Rho       float64 // 左、右差分相关系数
	BandShare float64 // |差分| ≤ BandStep·σ 的占比，衡量符号不可分的数值歧义
}

// BandStep 是数值歧义带的半宽，单位为差分标准差。
const BandStep = 1e-3

// modelDensity 是相邻差分 (X,Y) 联合高斯的异号概率：
//
//	P(X<0, Y>0) = P(X>0, Y<0) = 1/4 - arcsin(ρ)/(2π)
//
// 要求 X、Y 零均值、等方差。ρ=0 时为 1/4；ρ→1 时为 0；ρ=-1 时为 1/2。
func modelDensity(rho float64) float64 {
	rho = math.Max(-1, math.Min(1, rho))
	return 0.25 - math.Asin(rho)/(2*math.Pi)
}

// pairStatsOf 在按行存放的序列 series（行宽 stride）上统计：
// 序列有效下标 [i0, i1]，相邻对 (i, i+1)，i∈[i0, i1-1]。
// 每行的统计合并成全局统计，行间不做跨行配对。
func pairStatsOf(series []float64, stride, i0, i1 int) PairStats {
	var st PairStats
	if i1-i0 < 2 {
		return st
	}
	var sx, sy []float64
	for y := 0; (y+1)*stride <= len(series); y++ {
		base := y * stride
		for i := i0; i < i1; i++ {
			sx = append(sx, series[base+i])
			sy = append(sy, series[base+i+1])
		}
	}
	n := float64(len(sx))
	if n == 0 {
		return st
	}
	st.Count = len(sx)
	var mx, my float64
	for i := range sx {
		mx += sx[i]
		my += sy[i]
	}
	mx /= n
	my /= n
	var vx, vy, cxy, m3, m4 float64
	for i := range sx {
		a := sx[i] - mx
		b := sy[i] - my
		vx += a * a
		vy += b * b
		cxy += a * b
		m3 += a * a * a
		m4 += a * a * a * a
	}
	vx /= n
	vy /= n
	cxy /= n
	st.Mean = (mx + my) / 2
	st.VarLeft = vx
	st.VarRight = vy
	st.Sigma = math.Sqrt((vx + vy) / 2)
	if st.Sigma > 0 {
		st.Skew = (m3 / n) / (st.Sigma * st.Sigma * st.Sigma)
		st.Kurt = (m4/n)/(st.Sigma*st.Sigma*st.Sigma*st.Sigma) - 3
	}
	if vx > 0 && vy > 0 {
		st.Rho = cxy / math.Sqrt(vx*vy)
	}
	band := BandStep * st.Sigma
	var inBand int
	for i := range sx {
		if math.Abs(sx[i]-mx) <= band {
			inBand++
		}
		if math.Abs(sy[i]-my) <= band {
			inBand++
		}
	}
	st.BandShare = float64(inBand) / (2 * n)
	return st
}

// Model 是单个判定公式的预判结果。
type Model struct {
	Name     string
	Pairs    PairStats
	P        float64 // 单点符号翻转概率
	Count    float64 // 全图预判像素数
	Measured int     // 全图实测像素数
}

// Deviation 返回 (预判-实测)/实测。实测为零时返回 NaN。
func (m Model) Deviation() float64 {
	if m.Measured == 0 {
		return math.NaN()
	}
	return (m.Count - float64(m.Measured)) / float64(m.Measured)
}

// Frame 是一次完整分析的结果。
type Frame struct {
	Size       int
	Data       []float64
	Candidates int // 参与判定的像素数
	Curvature  Curvature
	Mask       Mask
	Ties       int

	KMin  Model // 曲率 κ 的局部极小
	Crest Model // 高度 h 的局部极大（分水脊）

	Both     int // 两类像素的重合数
	Comps    []Component
	Labels   []int32
	IIDCount float64 // 独立同分布假设（只用边际分布）的预判
}

// AnalyzeFrame 完成曲率、检测、连通段与两种预判。
func AnalyzeFrame(size int, data []float64) Frame {
	f := Frame{Size: size, Data: data}
	if size < 5 {
		return f
	}
	f.Candidates = size * (size - 4)
	f.Curvature = CurvatureOf(size, data)
	f.Mask, f.Ties = Detect(f.Curvature)
	f.Labels, f.Comps = Label(f.Mask)

	// 曲率差分序列 D_i = κ(i+1)-κ(i)，i∈[1, size-3]。
	// 候选列 x∈[2,size-3] 使用 (D_{x-1}, D_x)，即 pairs i∈[1,size-4]；
	// pairStatsOf 的区间写作 [1, size-3)，与候选像素一一对应。
	dK := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for i := 1; i <= size-3; i++ {
			dK[y*size+i] = f.Curvature.At(i+1, y) - f.Curvature.At(i, y)
		}
	}
	f.KMin = Model{
		Name:     "曲率极小",
		Pairs:    pairStatsOf(dK, size, 1, size-3),
		Measured: f.Mask.Count(),
	}
	f.KMin.P = modelDensity(f.KMin.Pairs.Rho)
	f.KMin.Count = f.KMin.P * float64(f.Candidates)

	// 高度差分序列 Δh_i = h(i+1)-h(i)，i∈[1,size-2]。
	// 候选列 x∈[2,size-3] 使用 (Δh_{x-1}, Δh_x)，即 pairs i∈[1,size-4]；
	// 同样写作 [1, size-3) 以对齐候选像素数。
	dH := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for i := 1; i <= size-2; i++ {
			dH[y*size+i] = data[y*size+i+1] - data[y*size+i]
		}
	}
	crestMask := DetectCrest(size, data)
	f.Crest = Model{
		Name:     "高度极大",
		Pairs:    pairStatsOf(dH, size, 1, size-3),
		Measured: crestMask.Count(),
	}
	f.Crest.P = modelDensity(f.Crest.Pairs.Rho)
	f.Crest.Count = f.Crest.P * float64(f.Candidates)

	for i := range f.Mask.Data {
		if f.Mask.Data[i] && crestMask.Data[i] {
			f.Both++
		}
	}

	// 独立同分布假设：任意连续分布下，三点中「中间点严格最小/最大」的概率
	// 都是 1/3，与边际分布形状无关。只用边际分布时这是唯一可得的结论。
	f.IIDCount = float64(f.Candidates) / 3

	return f
}

// TopComponents 返回像素数最大的 k 个连通段（不足则全部）。
func TopComponents(comps []Component, k int) []Component {
	if k > len(comps) {
		k = len(comps)
	}
	return comps[:k]
}

// ComponentHistogram 按像素数分桶：1、2–3、4–10、11–50、>50。
func ComponentHistogram(comps []Component) [5]int {
	var h [5]int
	for _, c := range comps {
		switch {
		case c.Pixels == 1:
			h[0]++
		case c.Pixels <= 3:
			h[1]++
		case c.Pixels <= 10:
			h[2]++
		case c.Pixels <= 50:
			h[3]++
		default:
			h[4]++
		}
	}
	return h
}

type pixelPair struct{ a, b int }

// TotalLength 估计脊线总长（单位：像素）：
// 每个连通段内取相邻像素对的欧氏距离，八连通对角步长为 √2，直步长为 1，
// 用最小生成树把它们连起来，累加边权即为该段的折线长度下界。
// 单像素段长度为 0。
func TotalLength(m Mask) (length float64, edges int) {
	n := m.Size
	// 收集边：(段内相邻像素对, 权重)。
	edgesW := make([]float64, 0, 4*len(m.Data))
	pairs := make([]pixelPair, 0, 4*len(m.Data))
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			i := y*n + x
			if !m.Data[i] {
				continue
			}
			// 只看向右下四个方向，避免重复。
			for _, d := range [4][2]int{{1, 0}, {0, 1}, {1, 1}, {-1, 1}} {
				nx, ny := x+d[0], y+d[1]
				if nx < 0 || nx >= n || ny < 0 || ny >= n {
					continue
				}
				j := ny*n + nx
				if !m.Data[j] {
					continue
				}
				w := 1.0
				if d[0] != 0 && d[1] != 0 {
					w = math.Sqrt2
				}
				pairs = append(pairs, pixelPair{i, j})
				edgesW = append(edgesW, w)
			}
		}
	}
	return kruskal(n*n, pairs, edgesW)
}

// kruskal 用并查集求最小生成树总权；pairs 的边权只有 1 和 √2 两种，按权排序后选边。
func kruskal(n int, pairs []pixelPair, w []float64) (total float64, edges int) {
	parent := make([]int32, n)
	for i := range parent {
		parent[i] = int32(i)
	}
	var find func(int32) int32
	find = func(x int32) int32 {
		for parent[x] != x {
			parent[x] = parent[parent[x]]
			x = parent[x]
		}
		return x
	}
	// 两类权重分别处理，直步优先。
	for _, want := range []float64{1, math.Sqrt2} {
		for k, ww := range w {
			if ww != want {
				continue
			}
			a, b := pairs[k].a, pairs[k].b
			ra, rb := find(int32(a)), find(int32(b))
			if ra == rb {
				continue
			}
			parent[ra] = rb
			total += ww
			edges++
		}
	}
	return total, edges
}
