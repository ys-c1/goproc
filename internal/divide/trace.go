package divide

// Segment 是一条连通的分界线，由首尾在网格顶点相接的格边组成。
type Segment struct {
	Edges []Edge
}

// vertex 是网格顶点坐标。水平格边 H(x,y) 的端点为 (x,y)-(x+1,y)；
// 竖直格边 V(x,y) 的端点为 (x,y)-(x,y+1)。
type vertex struct{ x, y int }

func endpoints(e Edge) (vertex, vertex) {
	if e.Kind == 'H' {
		return vertex{e.X, e.Y}, vertex{e.X + 1, e.Y}
	}
	return vertex{e.X, e.Y}, vertex{e.X, e.Y + 1}
}

// Trace 把格边集合按共享网格顶点描成若干连通线段。
//
// 这是对“边—顶点邻接图”求连通分量：每个格边是图的一条边，端点相同即相连，
// 全程只在已经判定出的格边集合上遍历，不向普通地形格做任何洪泛或盆地标号。
// 孤立的单条格边也作为长度为 1 的线段返回。
func Trace(edges []Edge) []Segment {
	adj := make(map[vertex][]Edge, len(edges)*2)
	for _, e := range edges {
		a, b := endpoints(e)
		adj[a] = append(adj[a], e)
		adj[b] = append(adj[b], e)
	}

	seen := make(map[Edge]bool, len(edges))
	var segments []Segment
	for _, start := range edges {
		if seen[start] {
			continue
		}
		seen[start] = true
		stack := []Edge{start}
		var seg Segment
		for len(stack) > 0 {
			e := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			seg.Edges = append(seg.Edges, e)
			a, b := endpoints(e)
			for _, nb := range adj[a] {
				if !seen[nb] {
					seen[nb] = true
					stack = append(stack, nb)
				}
			}
			for _, nb := range adj[b] {
				if !seen[nb] {
					seen[nb] = true
					stack = append(stack, nb)
				}
			}
		}
		segments = append(segments, seg)
	}
	return segments
}
