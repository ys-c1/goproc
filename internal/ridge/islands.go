package ridge

import (
	"math"
	"sort"
)

// Island 是 Mask 在 8 连通意义下的一个连通分量。
type Island struct {
	ID     int
	Size   int
	MinX   int
	MinY   int
	MaxX   int
	MaxY   int
	HasMax bool    // 分量内是否含严格局部极大点
	Maxima []int   // 含有的极大点在 maxima 列表中的下标
	Frac   float64 // 面积占比
	CellsX []int   // 分量内格点 x
	CellsY []int   // 分量内格点 y
}

// Islands 返回 mask 的 8 连通分量，并标注每个分量含哪些局部极大点。
func (f *Field) Islands(mask []bool, maxima []Extremum) []Island {
	n := f.N
	label := make([]int, n*n)
	for i := range label {
		label[i] = -1
	}
	at := make(map[int]int, len(maxima))
	for k, m := range maxima {
		at[m.Y*n+m.X] = k
	}
	var out []Island
	queue := make([]int, 0, 64)
	for start := 0; start < n*n; start++ {
		if !mask[start] || label[start] >= 0 {
			continue
		}
		id := len(out)
		is := Island{ID: id, MinX: n, MinY: n, MaxX: -1, MaxY: -1}
		queue = queue[:0]
		queue = append(queue, start)
		label[start] = id
		for len(queue) > 0 {
			v := queue[0]
			queue = queue[1:]
			x, y := v%n, v/n
			is.Size++
			is.CellsX = append(is.CellsX, x)
			is.CellsY = append(is.CellsY, y)
			if x < is.MinX {
				is.MinX = x
			}
			if y < is.MinY {
				is.MinY = y
			}
			if x > is.MaxX {
				is.MaxX = x
			}
			if y > is.MaxY {
				is.MaxY = y
			}
			if k, ok := at[v]; ok {
				is.HasMax = true
				is.Maxima = append(is.Maxima, k)
			}
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					xx, yy := x+dx, y+dy
					if xx < 0 || yy < 0 || xx >= n || yy >= n {
						continue
					}
					w := yy*n + xx
					if mask[w] && label[w] < 0 {
						label[w] = id
						queue = append(queue, w)
					}
				}
			}
		}
		is.Frac = float64(is.Size) / float64(n*n)
		out = append(out, is)
	}
	return out
}

// SortIslandsBySizeDesc 按分量大小降序排序。
func SortIslandsBySizeDesc(v []Island) {
	sort.SliceStable(v, func(i, j int) bool { return v[i].Size > v[j].Size })
}

// BBoxDistance 返回到分量的欧氏距离的包围盒下界；点在盒内时为 0。
func (is Island) BBoxDistance(x, y int) float64 {
	dx := 0
	if x < is.MinX {
		dx = is.MinX - x
	} else if x > is.MaxX {
		dx = x - is.MaxX
	}
	dy := 0
	if y < is.MinY {
		dy = is.MinY - y
	} else if y > is.MaxY {
		dy = y - is.MaxY
	}
	if dx == 0 && dy == 0 {
		return 0
	}
	return math.Hypot(float64(dx), float64(dy))
}
