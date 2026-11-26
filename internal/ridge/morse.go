package ridge

import "sort"

// MorseCount 是 Mask 连通段数的离散 Morse 计数。
//
// 把 Mask 视作 Δf<0 的负值集合，按 Δf 从低到高把格点加入并查集：
//   - 没有更低的 Mask 邻点：新开一个段（birth）；
//   - 接到 1 个已有段：不改变段数；
//   - 接到 k>=2 个已有段：合并，段数减少 k-1。
//
// 于是
//
//	段数 = Births - Σ_merge (k-1)
//
// Births 恰好是 Δf 在全网格（按存在邻点比较）意义下的严格局部极小点
// 中 Δf<0 的那些；merge 格是 Δf<0 的鞍型格点。这就是把段数写成
// Δf 局部极值点数的函数。
type MorseCount struct {
	Births     int
	MergeLoss  int
	Components int
	Saddles    []Saddle
	// BirthCells 是每个段的出生格点。
	BirthCells []Extremum
}

// ComputeMorse 在 mask 上按 Δf 递增顺序做并查集，返回精确段数及其分解。
func (f *Field) ComputeMorse(mask []bool) MorseCount {
	n := f.N
	lap := f.Laplacian()
	order := make([]int, 0, n*n)
	for i, b := range mask {
		if b {
			order = append(order, i)
		}
	}
	// 按 Δf 升序，值相同时按下标，保证确定性。
	sort.SliceStable(order, func(a, b int) bool {
		ia, ib := order[a], order[b]
		if lap[ia] != lap[ib] {
			return lap[ia] < lap[ib]
		}
		return ia < ib
	})

	parent := make([]int, n*n)
	added := make([]bool, n*n)
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

	mc := MorseCount{}
	for _, idx := range order {
		x, y := idx%n, idx/n
		added[idx] = true
		parent[idx] = idx
		roots := make(map[int]struct{}, 4)
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
				if added[j] {
					roots[find(j)] = struct{}{}
				}
			}
		}
		if len(roots) == 0 {
			mc.Births++
			mc.BirthCells = append(mc.BirthCells, Extremum{X: x, Y: y, Value: lap[idx]})
			continue
		}
		reps := make([]int, 0, len(roots))
		for r := range roots {
			reps = append(reps, r)
		}
		sort.Ints(reps)
		for _, r := range reps {
			parent[r] = idx
		}
		parent[idx] = idx
		if len(reps) >= 2 {
			mc.MergeLoss += len(reps) - 1
			mc.Saddles = append(mc.Saddles, Saddle{X: x, Y: y, Arity: len(reps)})
		}
	}
	mc.Components = mc.Births - mc.MergeLoss
	return mc
}

// NegativeLaplacianMinima 返回 Δf 严格局部极小且 Δf<0 的格点数。
// 在无平台数据上它应与 ComputeMorse 的 Births 相等。
func (f *Field) NegativeLaplacianMinima() int {
	lap := f.Laplacian()
	n := f.N
	count := 0
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			v := lap[y*n+x]
			if v >= 0 {
				continue
			}
			ok := true
			for dy := -1; dy <= 1 && ok; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					xx, yy := x+dx, y+dy
					if xx < 0 || yy < 0 || xx >= n || yy >= n {
						continue
					}
					if lap[yy*n+xx] <= v {
						ok = false
						break
					}
				}
			}
			if ok {
				count++
			}
		}
	}
	return count
}
