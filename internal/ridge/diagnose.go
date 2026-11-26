package ridge

import (
	"math"
	"sort"
)

// NearestIsland 返回 (x,y) 到报告内各分量的最近分量 ID 与欧氏距离。
// 无分量时返回 -1, +Inf。
func (f *Field) NearestIsland(rep Report, x, y int) (int, float64) {
	return f.nearestIsland(rep.raw, x, y)
}

// nearestIsland 返回 (x,y) 到各分量的最近欧氏距离与分量编号。
func (f *Field) nearestIsland(isl []Island, x, y int) (int, float64) {
	bestID, best := -1, math.Inf(1)
	for _, is := range isl {
		// 先用包围盒下界剪枝。
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
		if float64(dx*dx+dy*dy) >= best {
			continue
		}
		for k := range is.CellsX {
			dd := math.Hypot(float64(is.CellsX[k]-x), float64(is.CellsY[k]-y))
			if dd < best {
				best = dd
				bestID = is.ID
			}
		}
	}
	return bestID, best
}

// DistToRidge 计算 (x,y) 到脊线折线顶点的最近距离。
func DistToRidge(r Ridge, x, y float64) float64 {
	best := math.Inf(1)
	for _, p := range r.Points {
		dd := math.Hypot(p[0]-x, p[1]-y)
		if dd < best {
			best = dd
		}
	}
	return best
}

// ExtraIslandMeta 记录一个不含内部极大点的分量，即“多出来的段”。
type ExtraIslandMeta struct {
	IslandReport
	IslandFrac      float64
	NearestMaxDist  float64
	NearestMaxLabel string
}

// ExtraMeta 返回“多出来的段”的逐项定位。
func (f *Field) ExtraMeta(rep Report, maxima []Extremum) []ExtraIslandMeta {
	var out []ExtraIslandMeta
	for _, is := range rep.Islands {
		if is.InteriorMaxCount > 0 {
			continue
		}
		best, label := math.Inf(1), ""
		for _, m := range maxima {
			d := math.Hypot(float64(m.X)-is.CentroidX, float64(m.Y)-is.CentroidY)
			if d < best {
				best = d
				label = fmtPoint(m.X, m.Y)
			}
		}
		out = append(out, ExtraIslandMeta{
			IslandReport:    is,
			NearestMaxDist:  best,
			NearestMaxLabel: label,
		})
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Size > out[j].Size })
	return out
}

func fmtPoint(x, y int) string {
	return "(" + itoa(x) + "," + itoa(y) + ")"
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}

// CrestEndpoint 是裁剪后脊线的一条开曲线端点。
type CrestEndpoint struct {
	CurveIndex int
	X, Y       float64
}

// CrestEndpoints 返回所有开曲线的端点，按坐标排序。
func (r Ridge) CrestEndpoints() []CrestEndpoint {
	var out []CrestEndpoint
	for i, c := range r.Curves {
		if c.Closed {
			continue
		}
		for _, e := range c.Endpoints {
			out = append(out, CrestEndpoint{CurveIndex: i, X: e[0], Y: e[1]})
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Y != out[j].Y {
			return out[i].Y < out[j].Y
		}
		return out[i].X < out[j].X
	})
	return out
}
