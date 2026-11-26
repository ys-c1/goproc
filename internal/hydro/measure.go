package hydro

import "sort"

// CountComponentsBFS 独立于扫描预测，直接对 W(w)={p:h(p)<=w} 做四邻域连通标记。
// 网格外的格点不存在，不参与标记、不环绕。返回连通分量数。
func CountComponentsBFS(size int, h []float64, w float64) int {
	n := size * size
	visited := make([]bool, n)
	components := 0
	var stack []int
	for start := 0; start < n; start++ {
		if visited[start] || h[start] > w {
			continue
		}
		components++
		stack = append(stack[:0], start)
		visited[start] = true
		for len(stack) > 0 {
			v := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			y, x := v/size, v%size
			for _, q := range dirs4 {
				ny, nx := y+q[0], x+q[1]
				if ny < 0 || ny >= size || nx < 0 || nx >= size {
					continue // 网格外不是水域
				}
				u := ny*size + nx
				if !visited[u] && h[u] <= w {
					visited[u] = true
					stack = append(stack, u)
				}
			}
		}
	}
	return components
}

// ComponentsAt 返回水位 w 下的预测分量数。
// 取高度不超过 w 的最后一条扫描记录（C 是右连续阶梯函数：C(w)=C(h_k)，
// h_k 为满足 h_k<=w 的最大格点高度）；w 低于最小高度时返回 0。
func (a *Analysis) ComponentsAt(w float64) int {
	idx := sort.Search(len(a.Records), func(i int) bool {
		return a.Records[i].Height > w
	}) - 1
	if idx < 0 {
		return 0
	}
	return a.Records[idx].Components
}

// Band 把水位 w 映射到其跨过的扫描记录下标（满足 height<=w 的最大下标）；
// w 低于最小高度时返回 -1。
func (a *Analysis) Band(w float64) int {
	idx := sort.Search(len(a.Records), func(i int) bool {
		return a.Records[i].Height > w
	}) - 1
	return idx
}

// EventsAt 返回水位 w 已跨过（高度 <=w）的全部临界事件。
func (a *Analysis) EventsAt(w float64) []*Event {
	k := 0
	for k < len(a.Events) && a.Events[k].Height <= w {
		k++
	}
	return a.Events[:k]
}
