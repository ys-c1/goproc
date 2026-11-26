// Package hydro 在离散高度图上做水位—连通分量分析。
//
// 约定（全项目统一）：
//   - 水域 W(w) = {格点 p | h(p) <= w}；
//   - 连通 = 四邻域（上下左右），八邻域只用于几何鞍点对照；
//   - 网格外既不是水域也不是陆地，不参与任何连接，边界不环绕、无外部海洋。
package hydro

// dsu 是按高度压缩、按秩合并的并查集，只在已“入水”的格点上操作。
type dsu struct {
	parent []int
	rank   []uint8
	// ncomp 是当前并查集内连通分量（不同根）的数量。
	ncomp int
}

func newDSU(n int) *dsu {
	return &dsu{parent: make([]int, n), rank: make([]uint8, n)}
}

func (d *dsu) add(v int) {
	d.parent[v] = v
	d.ncomp++
}

func (d *dsu) find(v int) int {
	for d.parent[v] != v {
		d.parent[v] = d.parent[d.parent[v]]
		v = d.parent[v]
	}
	return v
}

// union 合并两个分量；两者原本不同根时返回 true，分量数减一。
func (d *dsu) union(a, b int) bool {
	ra, rb := d.find(a), d.find(b)
	if ra == rb {
		return false
	}
	switch {
	case d.rank[ra] < d.rank[rb]:
		ra, rb = rb, ra
	case d.rank[ra] == d.rank[rb]:
		d.rank[ra]++
	}
	d.parent[rb] = ra
	d.ncomp--
	return true
}
