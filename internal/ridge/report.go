package ridge

import (
	"fmt"
	"sort"
)

// Prediction 是“Mask 连通分量数”的可证上界。
//
// 依据见 docs/ridges.md：Δf<0 与任务书判据等价；内部格点上 Δf<0 的每个
// 8 连通分量都必须包含 Δf 的一个局部极小点，因此内部段数 ≤ Δf 的内部
// 局部极小点数。触边分量由边界条件产生，不在此上界内，单独计数。
type Prediction struct {
	InteriorMinima  int
	BoundaryIslands int
}

// Actual 是 Mask 分量的实际统计。
type Actual struct {
	Total          int
	Interior       int
	Boundary       int
	MaximaTotal    int
	InteriorMaxima int
	MaximaInMask   int
}

// IslandReport 是单个分量的明细。
type IslandReport struct {
	ID               int
	Size             int
	BBox             [4]int // minX,minY,maxX,maxY
	CentroidX        float64
	CentroidY        float64
	TouchesBorder    bool
	Maxima           []Extremum // 分量内含的全部严格局部极大点
	InteriorMaxCount int        // 其中位于内部的个数
}

// Report 汇总一个画幅的预测与实际。
type Report struct {
	Name       string
	Prediction Prediction
	Actual     Actual
	Islands    []IslandReport

	// Morse 是 Mask 段数的精确分解：Components = Births - MergeLoss。
	Morse MorseCount

	// Slack 是“多出段”数量：总段数 − 落在 Mask 内的内部极大点数。
	// 当且仅当 Slack>0 时，Mask 的段数多于极大点能解释的数目。
	Slack int
	// NoMaxSegments 是不含任何内部极大点的段数。
	NoMaxSegments int
	// WithMaxSegments 是含至少一个内部极大点的段数。
	WithMaxSegments int
	// SharedSurplus 是“一 段多极大点”造成的冗余：MaximaInMask − WithMaxSegments。
	SharedSurplus int

	// raw 保存分量格点，供最近距离查询复用；不对外。
	raw []Island
}

// Analyze 统计一个画幅的分量与极值。
func (f *Field) Analyze(name string, mask []bool, maxima []Extremum) Report {
	n := f.N
	isl := f.Islands(mask, maxima)
	r := Report{Name: name}

	for _, c := range f.CriticalPoints() {
		if c.IsMin {
			r.Prediction.InteriorMinima++
		}
	}

	for _, m := range maxima {
		interior := m.X > 0 && m.Y > 0 && m.X < n-1 && m.Y < n-1
		if interior {
			r.Actual.InteriorMaxima++
			if mask[m.Y*n+m.X] {
				r.Actual.MaximaInMask++
			}
		}
	}
	r.Actual.MaximaTotal = len(maxima)

	for _, is := range isl {
		touch := is.MinX == 0 || is.MinY == 0 || is.MaxX == n-1 || is.MaxY == n-1
		if touch {
			r.Actual.Boundary++
			r.Prediction.BoundaryIslands++
		} else {
			r.Actual.Interior++
		}
		mx := make([]Extremum, 0, len(is.Maxima))
		interiorCount := 0
		for _, k := range is.Maxima {
			m := maxima[k]
			mx = append(mx, m)
			if m.X > 0 && m.Y > 0 && m.X < n-1 && m.Y < n-1 {
				interiorCount++
			}
		}
		sort.SliceStable(mx, func(i, j int) bool {
			if mx[i].Y != mx[j].Y {
				return mx[i].Y < mx[j].Y
			}
			return mx[i].X < mx[j].X
		})
		r.Islands = append(r.Islands, IslandReport{
			ID:               is.ID,
			Size:             is.Size,
			BBox:             [4]int{is.MinX, is.MinY, is.MaxX, is.MaxY},
			CentroidX:        centroid(is.CellsX),
			CentroidY:        centroid(is.CellsY),
			TouchesBorder:    touch,
			Maxima:           mx,
			InteriorMaxCount: interiorCount,
		})
	}
	r.Actual.Total = len(isl)
	r.raw = isl
	r.Morse = f.ComputeMorse(mask)
	r.Slack = r.Actual.Total - r.Actual.MaximaInMask
	for _, is := range r.Islands {
		if is.InteriorMaxCount == 0 {
			r.NoMaxSegments++
		} else {
			r.WithMaxSegments++
		}
	}
	r.SharedSurplus = r.Actual.MaximaInMask - r.WithMaxSegments
	return r
}

func centroid(v []int) float64 {
	if len(v) == 0 {
		return 0
	}
	var s int
	for _, x := range v {
		s += x
	}
	return float64(s) / float64(len(v))
}

// Extra 返回不含任何内部极大点的分量：按“局部极大点数”预测时会漏掉的那些段。
func (r Report) Extra() []IslandReport {
	var out []IslandReport
	for _, is := range r.Islands {
		if is.InteriorMaxCount == 0 {
			out = append(out, is)
		}
	}
	return out
}

// WithMax 返回含至少一个内部极大点的分量。
func (r Report) WithMax() []IslandReport {
	var out []IslandReport
	for _, is := range r.Islands {
		if is.InteriorMaxCount > 0 {
			out = append(out, is)
		}
	}
	return out
}

// Line 输出单行摘要，用于逐行比对。
func (r Report) Line() string {
	return fmt.Sprintf("%-14s | 内部段数 %d ≤ Δf内部极小 %d → %v | total=%d interior=%d boundary=%d | interiorMax=%d inMask=%d | 多出(slack)=%d",
		r.Name, r.Actual.Interior, r.Prediction.InteriorMinima,
		r.Actual.Interior <= r.Prediction.InteriorMinima,
		r.Actual.Total, r.Actual.Interior, r.Actual.Boundary,
		r.Actual.InteriorMaxima, r.Actual.MaximaInMask, r.Slack)
}

// MorseLine 输出段数的精确函数关系：段数 = 出生 − 合并损失。
func (r Report) MorseLine() string {
	return fmt.Sprintf("段数=%d = Births %d − MergeLoss %d（鞍点 %d）",
		r.Morse.Components, r.Morse.Births, r.Morse.MergeLoss, len(r.Morse.Saddles))
}

// CompactSlack 供测试断言：slack 必须等于 (总段数 − 内部极大点在 Mask 内数)。
func (r Report) CompactSlack() int { return r.Actual.Total - r.Actual.MaximaInMask }
