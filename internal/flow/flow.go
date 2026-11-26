// Package flow 在规则网格高度图上按 D8 八邻域规则判定每格流向。
//
// 每格的流向只依赖自身与八个邻格的高度，因此全图流向可以一次扫描
// 算出，无需迭代累积。坡度为零（无严格下坡）的格子不指派方向，
// 标记为 Sink，单独处理。
package flow

import "math"

// Dir 是离散流向，取值 0..7 对应八邻域，255 表示无流向（汇/平地）。
//
//	0=E 1=SE 2=S 3=SW 4=W 5=NW 6=N 7=NE
type Dir uint8

const (
	E  Dir = 0
	SE Dir = 1
	S  Dir = 2
	SW Dir = 3
	W  Dir = 4
	NW Dir = 5
	N  Dir = 6
	NE Dir = 7
	// Sink 表示该格没有严格更低的邻格（局部洼地或平地），水不流出。
	Sink Dir = 255
)

// dx/dy 按 Dir 数值排列；对角方向步长为 sqrt2。
var dx = [8]int{1, 1, 0, -1, -1, -1, 0, 1}
var dy = [8]int{0, 1, 1, 1, 0, -1, -1, -1}
var stepLen = [8]float64{
	1, math.Sqrt2, 1, math.Sqrt2,
	1, math.Sqrt2, 1, math.Sqrt2,
}

// Neighbor 返回方向 d 的邻格坐标，d 必须是 0..7。
func Neighbor(x, y int, d Dir) (nx, ny int) {
	return x + dx[d], y + dy[d]
}

// TerminusKind 标记逐格追踪的终点类型。
type TerminusKind uint8

const (
	// TermSink 追踪终止于内陆汇格（无下坡，且不在图幅边界）。
	TermSink TerminusKind = iota
	// TermOutlet 追踪终止于图幅边界格，水由此流出画幅。
	TermOutlet
)

// Terminus 是一条流路的终点。
type Terminus struct {
	X, Y int
	Kind TerminusKind
}

// Map 保存全图流向及尺寸。
type Map struct {
	Size int
	Dirs []Dir
}

// At 取某格流向，越界返回 Sink。
func (m *Map) At(x, y int) Dir {
	if x < 0 || y < 0 || x >= m.Size || y >= m.Size {
		return Sink
	}
	return m.Dirs[y*m.Size+x]
}

// Direct 计算单格流向：在所有存在的邻格中取坡度（单位水平距离的
// 高度落差）最大者；同坡度按方向编号（先正东、顺时针，即正方向先
// 于斜方向）取编号最小者，保证结果唯一且可复算。无严格下坡（最大
// 坡度为零，含八邻格全部同高的零坡度格）返回 Sink。
func Direct(h []float64, size, x, y int) Dir {
	center := h[y*size+x]
	best := Sink
	bestSlope := 0.0
	for d := Dir(0); d < 8; d++ {
		nx, ny := x+dx[d], y+dy[d]
		if nx < 0 || ny < 0 || nx >= size || ny >= size {
			continue
		}
		drop := center - h[ny*size+nx]
		if drop <= 0 { // 只接受严格更低的邻格
			continue
		}
		slope := drop / stepLen[d]
		if slope > bestSlope { // 严格大于才替换；并列保留编号最小的方向
			bestSlope = slope
			best = d
		}
	}
	return best
}

// Build 一次扫描预判全图流向。每个格子的判定只读 3x3 邻域，格间
// 没有读写依赖，复杂度 O(N)，N=size^2，不含任何迭代累积。
func Build(h []float64, size int) *Map {
	m := &Map{Size: size, Dirs: make([]Dir, size*size)}
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			m.Dirs[y*size+x] = Direct(h, size, x, y)
		}
	}
	return m
}

// NaiveTrace 不使用预判结果，从 (x,y) 出发逐格重算流向并行走，直到
// 到达边界格（出口）或汇格（内陆洼地/平地）。这是参照实现，用来
// 与一次性预判的结论逐格比对。
func NaiveTrace(h []float64, size, x, y int) Terminus {
	for {
		if x == 0 || y == 0 || x == size-1 || y == size-1 {
			return Terminus{X: x, Y: y, Kind: TermOutlet}
		}
		d := Direct(h, size, x, y)
		if d == Sink {
			return Terminus{X: x, Y: y, Kind: TermSink}
		}
		x, y = x+dx[d], y+dy[d]
	}
}

// ResolveAll 基于预判流向一次性给出每格的追踪终点。方向指针构成
// 函数图；Direct 只指向严格更低格，高度沿指针严格递减，故图中无环，
// 记忆化深度优先在线性时间 O(N) 内结算全部格子。
func (m *Map) ResolveAll() []Terminus {
	n := m.Size * m.Size
	out := make([]Terminus, n)
	done := make([]bool, n)
	seen := make([]bool, n) // 防御性环标记；规则本身保证不会出现环

	for start := 0; start < n; start++ {
		if done[start] {
			continue
		}
		path := path[:0]
		idx := start
		var t Terminus
		for {
			if done[idx] {
				t = out[idx]
				break
			}
			if seen[idx] { // 理论不可达：出现环则按汇处理
				t = Terminus{X: idx % m.Size, Y: idx / m.Size, Kind: TermSink}
				break
			}
			seen[idx] = true
			path = append(path, idx)

			x, y := idx%m.Size, idx/m.Size
			if x == 0 || y == 0 || x == m.Size-1 || y == m.Size-1 {
				t = Terminus{X: x, Y: y, Kind: TermOutlet}
				break
			}
			d := m.Dirs[idx]
			if d == Sink {
				t = Terminus{X: x, Y: y, Kind: TermSink}
				break
			}
			nx, ny := x+dx[d], y+dy[d]
			idx = ny*m.Size + nx
		}
		// 同一条路径上的所有格子共享同一个终点，一次回填。
		for _, p := range path {
			out[p] = t
			done[p] = true
			seen[p] = false
		}
	}
	return out
}

// path 复用路径缓冲，避免每个起点重新分配。
var path []int

// dirGray 把流向映射成可复算的灰度：方向 d 编码为 d*32+1，
// 取值 1..225；Sink 编码为 0，不与任何方向冲突。
// 解码：g==0 -> Sink，否则 d=(g-1)/32。
var dirGray = [8]byte{1, 33, 65, 97, 129, 161, 193, 225}

// Gray 返回流向对应的灰度值，Sink 为 0。
func (d Dir) Gray() byte {
	if d == Sink {
		return 0
	}
	return dirGray[int(d)]
}

// FromGray 是 Gray 的逆映射，0 还原为 Sink。
func FromGray(g byte) Dir {
	if g == 0 {
		return Sink
	}
	return Dir((g - 1) / 32)
}

// EncodePGM 把流向图编码为八位 PGM（P5），头注释记录编码规则，
// 任意读取者都能据注释逐格复算流向。
func (m *Map) EncodePGM() []byte {
	head := "P5\n" +
		"# goproc flow D8: gray=dir*32+1, dir 0=E 1=SE 2=S 3=SW 4=W 5=NW 6=N 7=NE; gray=0 sink/flat\n" +
		itoa(m.Size) + " " + itoa(m.Size) + "\n255\n"
	out := make([]byte, 0, len(head)+len(m.Dirs))
	out = append(out, head...)
	for _, d := range m.Dirs {
		out = append(out, d.Gray())
	}
	return out
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var buf [20]byte
	n := len(buf)
	for i > 0 {
		n--
		buf[n] = byte('0' + i%10)
		i /= 10
	}
	return string(buf[n:])
}

// Cell 是一个格子的坐标。
type Cell struct{ X, Y int }

// Outlets 汇总所有被至少一格追踪到的边界出口，按行优先顺序去重排列。
func Outlets(term []Terminus, size int) []Cell {
	seen := make([]bool, size*size)
	var cells []Cell
	for _, t := range term {
		if t.Kind != TermOutlet {
			continue
		}
		id := t.Y*size + t.X
		if !seen[id] {
			seen[id] = true
			cells = append(cells, Cell{X: t.X, Y: t.Y})
		}
	}
	return cells
}

// Sinks 汇总所有内陆汇格（自身流向为 Sink 且不在边界），按行优先排列。
func (m *Map) Sinks() []Cell {
	var cells []Cell
	for y := 1; y < m.Size-1; y++ {
		for x := 1; x < m.Size-1; x++ {
			if m.Dirs[y*m.Size+x] == Sink {
				cells = append(cells, Cell{X: x, Y: y})
			}
		}
	}
	return cells
}

// FlatCells 返回八邻格全部不高于自身、即最大坡度为零的格子坐标
// （含边界），这是坡度为零、需要单独处理的格子集合。它与
// Direct 返回 Sink 的格子完全一致，单独列出便于核查。
func FlatCells(h []float64, size int) []Cell {
	var cells []Cell
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			if Direct(h, size, x, y) == Sink {
				cells = append(cells, Cell{X: x, Y: y})
			}
		}
	}
	return cells
}
