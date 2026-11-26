package ridge

// Laplacian 返回逐点离散拉普拉斯 Δf = Fxx + Fyy。
//
// 注意：边界格点的 Fxx、Fyy 用单侧二阶模板，与内部中心差分不是
// 同一个算子，因此 Δf 在边界只有一阶相容性；关于 Δf 的拓扑结论
// 只对内部格点使用。
func (f *Field) Laplacian() []float64 {
	out := make([]float64, len(f.F))
	for i := range out {
		out[i] = f.Fxx[i] + f.Fyy[i]
	}
	return out
}

// Critical 是 Δf 的一个严格局部极值点，用于给出连通段数上界。
type Critical struct {
	X, Y  int
	Value float64
	IsMin bool
}

// CriticalPoints 返回 Δf 的内部严格局部极值点（边界格点的单侧模板
// 与内部算子不同，不参与计数）。
func (f *Field) CriticalPoints() []Critical {
	lap := f.Laplacian()
	n := f.N
	var out []Critical
	for y := 1; y < n-1; y++ {
		for x := 1; x < n-1; x++ {
			v := lap[y*n+x]
			isMax, isMin := true, true
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					w := lap[(y+dy)*n+x+dx]
					if w >= v {
						isMax = false
					}
					if w <= v {
						isMin = false
					}
				}
			}
			if isMin {
				out = append(out, Critical{X: x, Y: y, Value: v, IsMin: true})
			}
			if isMax {
				out = append(out, Critical{X: x, Y: y, Value: v})
			}
		}
	}
	return out
}
