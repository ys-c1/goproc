// Package gully 定义冲沟核心格的双阈值合取判定、八邻域连通段、
// 阈值门控侵蚀模型以及侵蚀前后的逐格比对。
//
// 核心命题（详见 docs/gully-core-proof.md）：
//
//	Core(c) = (Acc(c) > τAcc) ∧ (Slope(c) > τSlope)
//
// 侵蚀模型的核心下切只作用于 Core 为真的格子，沟岸拓宽只作用于
// Core 的八邻域，因此侵蚀后的实测核心格（下切深度过检测杠）在参数
// 允许区间内与侵蚀前预判的 Core 逐格相等。
package gully

import (
	"math"

	"goproc/internal/hydro"
)

// Thresholds 是合取判定的两个阈值。
type Thresholds struct {
	Acc   int     // 汇流累积阈值 τAcc（上游格数；判定用严格 >）
	Slope float64 // 坡度阈值 τSlope（D8 落差/距离；判定用严格 >）
}

// CoreMask 是冲沟核心格的唯一判定入口：两个阈值的合取。
//
//	core[i] = (Acc[i] > τ.Acc) && (Slope[i] > τ.Slope)
func CoreMask(f *hydro.Field, t Thresholds) []bool {
	n := f.N
	core := make([]bool, n*n)
	for i := 0; i < n*n; i++ {
		core[i] = f.Acc[i] > t.Acc && f.Slope[i] > t.Slope
	}
	return core
}

// Segment 是一个八邻域连通段的统计信息。
type Segment struct {
	ID        int     // 从 1 开始，按扫描中首次出现顺序编号
	Count     int     // 格数
	Cells     []int   // 一维索引
	MinX      int     `json:"minX"`
	MinY      int     `json:"minY"`
	MaxX      int     `json:"maxX"`
	MaxY      int     `json:"maxY"`
	CentroidX float64 `json:"centroidX"`
	CentroidY float64 `json:"centroidY"`
}

// Components8 按八邻域统计连通段。
// comp 返回每格段号（背景为 0，段号从 1 开始），segs 为各段信息。
// 扫描顺序固定为行优先，BFS 邻格顺序固定为 hydro.DX/DY 的编号顺序，
// 因此同一张掩膜每次得到完全相同的段号。
func Components8(mask []bool, n int) (comp []int, seg []Segment) {
	comp = make([]int, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			start := hydro.Idx(n, x, y)
			if !mask[start] || comp[start] != 0 {
				continue
			}
			id := len(seg) + 1
			s := Segment{ID: id, MinX: x, MinY: y, MaxX: x, MaxY: y}
			queue := []int{start}
			comp[start] = id
			for head := 0; head < len(queue); head++ {
				i := queue[head]
				s.Cells = append(s.Cells, i)
				cx, cy := hydro.XY(n, i)
				s.Count++
				if cx < s.MinX {
					s.MinX = cx
				}
				if cx > s.MaxX {
					s.MaxX = cx
				}
				if cy < s.MinY {
					s.MinY = cy
				}
				if cy > s.MaxY {
					s.MaxY = cy
				}
				for d := 0; d < 8; d++ {
					nx, ny := cx+hydro.DX[d], cy+hydro.DY[d]
					if !hydro.Inside(n, nx, ny) {
						continue
					}
					j := hydro.Idx(n, nx, ny)
					if mask[j] && comp[j] == 0 {
						comp[j] = id
						queue = append(queue, j)
					}
				}
			}
			var sumX, sumY float64
			for _, i := range s.Cells {
				xx, yy := hydro.XY(n, i)
				sumX += float64(xx)
				sumY += float64(yy)
			}
			s.CentroidX = sumX / float64(s.Count)
			s.CentroidY = sumY / float64(s.Count)
			seg = append(seg, s)
		}
	}
	return comp, seg
}

// Dilate8 返回掩膜按切比雪夫距离 1（即自身加八邻域）膨胀 r 次的结果。
func Dilate8(mask []bool, n, r int) []bool {
	cur := append([]bool(nil), mask...)
	for k := 0; k < r; k++ {
		nxt := append([]bool(nil), cur...)
		for i, v := range cur {
			if !v {
				continue
			}
			x, y := hydro.XY(n, i)
			for d := 0; d < 8; d++ {
				nx, ny := x+hydro.DX[d], y+hydro.DY[d]
				if hydro.Inside(n, nx, ny) {
					nxt[hydro.Idx(n, nx, ny)] = true
				}
			}
		}
		cur = nxt
	}
	return cur
}

// ErodeParams 是一组侵蚀强度参数。两组强度用两个不同的 K / 沟岸上限。
type ErodeParams struct {
	Name            string  // 英文标识，如 weak / strong
	Label           string  // 中文标识，如 弱强度组
	K               float64 // 每轮核心下切系数
	Rounds          int     // 下切轮数（冻结汇流/坡度场）
	CapAccRatio     float64 // Acc/τAcc 的截断上限，保证驱动因子有界
	CapSlopeRatio   float64 // Slope/τSlope 的截断上限
	BankPerNeighbor float64 // 每个相邻核心格给沟岸格的拓宽深度
	BankCap         float64 // 单个沟岸格拓宽总深度上限
	Floor           float64 // 高程下限（一般 0）
}

// Drive 是归一化、有界的下切驱动因子：
//
//	drive(c) = sqrt( min(Acc/τAcc, CapA) * min(Slope/τSlope, CapS) )
//
// 在核心格上两个比值都严格大于 1，故 drive 严格大于 1；
// 全图最大不超过 sqrt(CapA*CapS)。
func Drive(f *hydro.Field, t Thresholds, p ErodeParams, i int) float64 {
	u := float64(f.Acc[i]) / float64(t.Acc)
	v := f.Slope[i] / t.Slope
	if u > p.CapAccRatio {
		u = p.CapAccRatio
	}
	if v > p.CapSlopeRatio {
		v = p.CapSlopeRatio
	}
	return math.Sqrt(u * v)
}

// Result 是一次完整侵蚀的逐格结果，全部数组长度 n*n。
type Result struct {
	H1        []float64 // 侵蚀后高程
	CoreCut   []float64 // 核心下切深度（仅核心格非零）
	BankCut   []float64 // 沟岸拓宽深度（仅非核心邻格可能非零）
	Cut       []float64 // 总下切 = H0 - H1
	Clipped   int       // 撞到 Floor 被截断的格子数（参数得当时应为 0）
	BankCells int       // 实际发生沟岸拓宽的格子数
	MaxDrive  float64
	MaxCut    float64
	TotalCut  float64
}

// Erode 跑完整侵蚀：Rounds 轮阈值门控下切 + 一次沟岸拓宽。
// 汇流累积与坡度始终取自侵蚀前的 f（单场降雨事件内冻结场）。
func Erode(h0 []float64, f *hydro.Field, t Thresholds, core []bool, p ErodeParams) *Result {
	n := f.N
	N := n * n
	res := &Result{H1: append([]float64(nil), h0...),
		CoreCut: make([]float64, N), BankCut: make([]float64, N), Cut: make([]float64, N)}

	// 多轮下切；驱动场冻结，轮次仅线性累加，循环保留“跑侵蚀”的过程。
	for r := 0; r < p.Rounds; r++ {
		for i := 0; i < N; i++ {
			if !core[i] {
				continue // 合取为假：不发生核心下切
			}
			d := p.K * Drive(f, t, p, i)
			if dr := Drive(f, t, p, i); dr > res.MaxDrive {
				res.MaxDrive = dr
			}
			newH := res.H1[i] - d
			if newH < p.Floor {
				d = res.H1[i] - p.Floor
				newH = p.Floor
				res.Clipped++
			}
			res.H1[i] = newH
			res.CoreCut[i] += d
		}
	}

	// 沟岸拓宽：只可能落在核心格的八邻域非核心格上。
	for i := 0; i < N; i++ {
		if core[i] {
			continue
		}
		x, y := hydro.XY(n, i)
		q := 0
		for d := 0; d < 8; d++ {
			nx, ny := x+hydro.DX[d], y+hydro.DY[d]
			if hydro.Inside(n, nx, ny) && core[hydro.Idx(n, nx, ny)] {
				q++
			}
		}
		if q == 0 {
			continue
		}
		b := p.BankPerNeighbor * float64(q)
		if b > p.BankCap {
			b = p.BankCap
		}
		newH := res.H1[i] - b
		if newH < p.Floor {
			b = res.H1[i] - p.Floor
			newH = p.Floor
			res.Clipped++
		}
		if b > 0 {
			res.H1[i] = newH
			res.BankCut[i] = b
			res.BankCells++
		}
	}

	for i := 0; i < N; i++ {
		c := h0[i] - res.H1[i]
		res.Cut[i] = c
		if c > res.MaxCut {
			res.MaxCut = c
		}
		res.TotalCut += c
	}
	return res
}

// MeasuredCore 是“侵蚀后逐格统计”的核心格判据：
// 用前后高程差量出的实际下切深度达到检测杠 DCore。
// 该判据不读取预判掩膜，只读取侵蚀产物 Cut。
func MeasuredCore(cut []float64, dCore float64) []bool {
	m := make([]bool, len(cut))
	for i, c := range cut {
		m[i] = c >= dCore
	}
	return m
}

// Diff 是两张布尔掩膜的逐格差异。
type Diff struct {
	Missing []int // 预判为核心、实测不是（漏报）
	Extra   []int // 实测为核心、预判不是（多报）
}

// Match 表示逐格完全一致。
func (d Diff) Match() bool { return len(d.Missing) == 0 && len(d.Extra) == 0 }

// Compare 逐格比对预判掩膜 pred 与实测掩膜 meas。
func Compare(pred, meas []bool) Diff {
	var d Diff
	for i := 0; i < len(pred); i++ {
		switch {
		case pred[i] && !meas[i]:
			d.Missing = append(d.Missing, i)
		case !pred[i] && meas[i]:
			d.Extra = append(d.Extra, i)
		}
	}
	return d
}

// NearCheck 核对所有实际受侵蚀格（cut>Eps）是否都落在
// Core 的半径 radius 八邻域膨胀带内；返回越界格索引。
func NearCheck(core []bool, cut []float64, n, radius int, eps float64) (zone []bool, violations []int) {
	zone = Dilate8(core, n, radius)
	for i, c := range cut {
		if c > eps && !zone[i] {
			violations = append(violations, i)
		}
	}
	return zone, violations
}
