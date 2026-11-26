// Package divide 在高度图上直接描出山与山之间的分界线（分水线）。
//
// 核心思想：每格沿最陡下降方向唯一指向一个汇（sink）。一条内部格边两侧的格若分属
// 不同汇，水就从该边两侧流向不同山坳，这条边正是两座山的分界。逐格求流向、再用记忆化
// 指针追踪得到每格所属汇，全程 O(n) 且不做逐格洪泛标号。
package divide

import "errors"

// 八个流向，按固定优先级排列；等下降量（tie）取最先出现者，保证结果确定。
// 斜向邻居的下降量按欧氏距离 √2 归一化。
const (
	N    = iota // (0,-1)
	NE          // (1,-1)
	E           // (1,0)
	SE          // (1,1)
	S           // (0,1)
	SW          // (-1,1)
	W           // (-1,0)
	NW          // (-1,-1)
	none = -1   // 无去向：汇（含闭洼平地）
)

var d8 = [8][2]int{
	{0, -1}, {1, -1}, {1, 0}, {1, 1},
	{0, 1}, {-1, 1}, {-1, 0}, {-1, -1},
}

// Edge 是一条格边。Kind 为 H（水平格边）或 V（竖直格边）。
//
// 坐标约定与高度图格 (x,y) 的网格线对齐：
//   - H(x,y)：第 y 条水平网格线上、横跨列 x 的水平格边，北侧格 (x,y-1)、南侧格 (x,y)；
//   - V(x,y)：第 x 条竖直网格线上、纵跨行 y 的竖直格边，西侧格 (x-1,y)、东侧格 (x,y)。
//
// 内部格边范围：H 的 y∈[1,n-1], x∈[0,n-1]；V 的 x∈[1,n-1], y∈[0,n-1]。
type Edge struct {
	Kind byte // 'H' 或 'V'
	X    int
	Y    int
}

// ErrBadSize 表示高度图尺寸非法。
var ErrBadSize = errors.New("divide: 尺寸非法")

// Flow 计算每格的 D8 最陡下降流向，返回长度 n*n 的方向数组（N..NW/none）。
//
// 选向规则：在八个邻居中取归一化下降量 (h(cur)-h(nb))/dist 最大者，dist 正交为 1、
// 斜向为 √2；只有严格为正的下降才算可流。多邻居并列时按 N,NE,E,...,NW 的固定优先级
// 取第一个，同输入必然同输出。无更低邻居的格先标 none，交给 ResolveFlats。
func Flow(n int, h []float64) ([]int, error) {
	if n <= 0 || len(h) != n*n {
		return nil, ErrBadSize
	}
	dir := make([]int, n*n)
	sqrt2 := 1.4142135623730951
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			cur := y*n + x
			ch := h[cur]
			bestDrop := 0.0
			best := none
			for d := N; d <= NW; d++ {
				nx, ny := x+d8[d][0], y+d8[d][1]
				if nx < 0 || nx >= n || ny < 0 || ny >= n {
					continue
				}
				dist := 1.0
				if d8[d][0] != 0 && d8[d][1] != 0 {
					dist = sqrt2
				}
				drop := (ch - h[ny*n+nx]) / dist
				if drop > bestDrop { // 严格下降；相等不覆盖，天然实现固定优先级
					bestDrop = drop
					best = d
				}
			}
			dir[cur] = best
		}
	}
	return dir, nil
}

// ResolveFlats 处理平地格：把与更低区域连通的等高水平台地整体指向其排水方向。
//
// 动机：Flow 只沿严格下降选向，台地内部互为等高、没有更低邻居，初始为 none。
// 斜台地（同高度连通体边缘有更低格）应能排水；被等高/更高格完全包围的闭洼才是真汇。
//
// 做法（有界 BFS，非逐格洪泛）：
//  1. 排水口 = 有严格更低邻居的格，携朝低方向入队；
//  2. 沿同高度连通体漫延，给初始为 none 的台地格统一赋该排水方向；漫延只在同高度体内，
//     不会跨高度，也不会改动普通坡格；
//  3. 闭洼连通体无排水口，格保持 none 作为汇。
//
// 同台地多排水口时按扫描顺序取首个方向，结果确定。
func ResolveFlats(n int, h []float64, dir []int) {
	type cell struct{ x, y int }
	unresolved := make([]bool, n*n)
	for i, d := range dir {
		if d == none {
			unresolved[i] = true
		}
	}
	visited := make([]bool, n*n)
	out := make([]int, n*n)
	for i := range out {
		out[i] = none
	}
	var queue []cell

	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			cur := y*n + x
			for d := N; d <= NW; d++ {
				nx, ny := x+d8[d][0], y+d8[d][1]
				if nx < 0 || nx >= n || ny < 0 || ny >= n {
					continue
				}
				if h[ny*n+nx] < h[cur] {
					visited[cur] = true
					out[cur] = d
					queue = append(queue, cell{x, y})
					break
				}
			}
		}
	}

	for head := 0; head < len(queue); head++ {
		c := queue[head]
		cur := c.y*n + c.x
		for d := N; d <= NW; d++ {
			nx, ny := c.x+d8[d][0], c.y+d8[d][1]
			if nx < 0 || nx >= n || ny < 0 || ny >= n {
				continue
			}
			ni := ny*n + nx
			if visited[ni] || h[ni] != h[cur] {
				continue
			}
			visited[ni] = true
			out[ni] = out[cur]
			if unresolved[ni] {
				dir[ni] = out[cur]
			}
			queue = append(queue, cell{nx, ny})
		}
	}
}

// Sinks 用记忆化流向指针追踪每格所属汇，返回汇编号数组与汇的数量。
//
// 每个格沿 dir 走到 none 即抵达汇；用 sink 数组记忆已解析格，使每条流向链只走一次，
// 总体 O(n)。这不是逐格洪泛：没有从汇向全图扩散的标号过程，只是顺指针读取每格自己的终点。
func Sinks(n int, dir []int) ([]int, int) {
	sink := make([]int, n*n)
	for i := range sink {
		sink[i] = -1
	}
	id := 0
	idOf := make(map[int]int)

	var resolve func(start int) int
	resolve = func(start int) int {
		// 迭代收集链，避免深递归；遇已解析格或汇即可回填。
		path := []int{}
		cur := start
		for sink[cur] < 0 {
			// 自环保护：正常 D8 严格下降不会成环，这里仅做防御。
			prev := len(path)
			for _, p := range path {
				if p == cur {
					root := id
					id++
					for _, q := range path[prev:] {
						sink[q] = root
					}
					return root
				}
			}
			path = append(path, cur)
			d := dir[cur]
			if d == none {
				if r, ok := idOf[cur]; ok {
					for _, q := range path {
						sink[q] = r
					}
					return r
				}
				r := id
				id++
				idOf[cur] = r
				for _, q := range path {
					sink[q] = r
				}
				return r
			}
			x, y := cur%n, cur/n
			cur = (y+d8[d][1])*n + (x + d8[d][0])
		}
		root := sink[cur]
		for _, q := range path {
			sink[q] = root
		}
		return root
	}

	for i := 0; i < n*n; i++ {
		if sink[i] < 0 {
			resolve(i)
		}
	}
	return sink, id
}

// OppositeEdges 只保留“两侧流向在该边法向上严格相反、各自背离该边”的格边。
//
// 这是定义最严格、可直接由流向判定、无需任何汇信息的分水边：
//   - H(x,y)：北侧格流向含北分量（N/NE/NW）且南侧格流向含南分量（S/SE/SW）；
//   - V(x,y)：西侧格流向含西分量（W/SW/NW）且东侧格流向含东分量（E/SE/NE）。
//
// 它是 SinkEdges 的子集（相背流必然汇入不同汇），用于对照证明中的“两侧流向相反”命题。
func OppositeEdges(n int, dir []int) (horiz, vert []Edge) {
	northish := map[int]bool{N: true, NE: true, NW: true}
	southish := map[int]bool{S: true, SE: true, SW: true}
	westish := map[int]bool{W: true, SW: true, NW: true}
	eastish := map[int]bool{E: true, SE: true, NE: true}
	for y := 1; y < n; y++ {
		for x := 0; x < n; x++ {
			a, b := dir[(y-1)*n+x], dir[y*n+x]
			if a != none && b != none && northish[a] && southish[b] {
				horiz = append(horiz, Edge{'H', x, y})
			}
		}
	}
	for x := 1; x < n; x++ {
		for y := 0; y < n; y++ {
			a, b := dir[y*n+x-1], dir[y*n+x]
			if a != none && b != none && westish[a] && eastish[b] {
				vert = append(vert, Edge{'V', x, y})
			}
		}
	}
	return horiz, vert
}

// SinkEdges 返回完整分水网：相邻格分属不同汇的内部格边。
// 水平、竖直两类分别返回，只含内部格边（图幅外轮廓不计）。
func SinkEdges(n int, sink []int) (horiz, vert []Edge) {
	for y := 1; y < n; y++ {
		for x := 0; x < n; x++ {
			if sink[(y-1)*n+x] != sink[y*n+x] {
				horiz = append(horiz, Edge{'H', x, y})
			}
		}
	}
	for x := 1; x < n; x++ {
		for y := 0; y < n; y++ {
			if sink[y*n+x-1] != sink[y*n+x] {
				vert = append(vert, Edge{'V', x, y})
			}
		}
	}
	return horiz, vert
}

// Compute 是一站式入口：求流向、处理平地、解析汇、提取完整分水线，
// 同时返回严格“流向相反”的边子集，供证明与对照使用。
func Compute(n int, h []float64) (dir []int, horiz, vert, oppH, oppV []Edge, sinkCount int, err error) {
	dir, err = Flow(n, h)
	if err != nil {
		return nil, nil, nil, nil, nil, 0, err
	}
	ResolveFlats(n, h, dir)
	sink, sinkCount := Sinks(n, dir)
	horiz, vert = SinkEdges(n, sink)
	oppH, oppV = OppositeEdges(n, dir)
	return dir, horiz, vert, oppH, oppV, sinkCount, nil
}
