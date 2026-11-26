// Package ridge 分析高度图的 Hessian 结构，提取山脊线并核对段数。
//
// 坐标约定与 hmap 一致：x 向右，y 向下，V[y*N+x] 是第 y 行第 x 列。
// 所有导数由 internal/diff 提供，内部格点用二阶中心差分，
// 边界格点用单侧二阶模板。
package ridge

import (
	"math"

	"goproc/internal/diff"
)

// Frame 是待分析的高度图。
type Frame struct {
	N int
	F []float64
}

// Field 保存高度图、一阶与二阶差分场、Hessian 特征值以及脊线标量场。
//
// L1 <= L2 为 Hessian 的两个特征值，(E1x,E1y) 是 L1 对应的单位特征向量。
// G = ∇f·e1，正确脊线判据是 G=0 且 L1<0。
type Field struct {
	N             int
	F             []float64
	Fx, Fy        []float64
	Fxx, Fyy, Fxy []float64
	L1, L2        []float64
	E1x, E1y      []float64
	G             []float64
}

// NewField 计算全部差分场与逐点特征分解。
func NewField(fr Frame) (*Field, error) {
	g := diff.Grid{N: fr.N, V: fr.F}
	if err := diff.Validate(g); err != nil {
		return nil, err
	}
	f := &Field{N: fr.N, F: fr.F}
	f.Fx = diff.D1X(g)
	f.Fy = diff.D1Y(g)
	f.Fxx = diff.D2X(g)
	f.Fyy = diff.D2Y(g)
	f.Fxy = diff.DXY(g)
	n2 := fr.N * fr.N
	f.L1 = make([]float64, n2)
	f.L2 = make([]float64, n2)
	f.E1x = make([]float64, n2)
	f.E1y = make([]float64, n2)
	f.G = make([]float64, n2)
	for i := 0; i < n2; i++ {
		a, b, c := f.Fxx[i], f.Fxy[i], f.Fyy[i]
		mean := 0.5 * (a + c)
		d := 0.5 * (a - c)
		r := math.Hypot(d, b)
		l1 := mean - r
		l2 := mean + r
		// (b, l1-a) 是 (H-l1·I) 的零空间向量。
		vx, vy := b, l1-a
		norm := math.Hypot(vx, vy)
		if norm == 0 {
			// H 是数量矩阵，方向任取。
			vx, vy, norm = 1, 0, 1
		}
		f.L1[i], f.L2[i] = l1, l2
		f.E1x[i], f.E1y[i] = vx/norm, vy/norm
		f.G[i] = f.Fx[i]*f.E1x[i] + f.Fy[i]*f.E1y[i]
	}
	return f, nil
}

// At 取格点标量。
func (f *Field) At(v []float64, x, y int) float64 { return v[y*f.N+x] }

// Extremum 是一个离散极值点。
type Extremum struct {
	X, Y  int
	Value float64
}

// LocalMaxima 返回 8 邻域意义下的严格局部极大点：比每个存在的邻点都大。
// 边界点只比较实际存在的邻点，因此边界上也可以出现极值。
func (f *Field) LocalMaxima() []Extremum { return f.localExtrema(true) }

// LocalMinima 返回严格局部极小点。
func (f *Field) LocalMinima() []Extremum { return f.localExtrema(false) }

func (f *Field) localExtrema(wantMax bool) []Extremum {
	n := f.N
	var out []Extremum
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			v := f.F[y*n+x]
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
					w := f.F[yy*n+xx]
					if wantMax && w >= v {
						ok = false
						break
					}
					if !wantMax && w <= v {
						ok = false
						break
					}
				}
			}
			if ok {
				out = append(out, Extremum{X: x, Y: y, Value: v})
			}
		}
	}
	return out
}

// Trace 返回逐点 Δf = Fxx + Fyy。
func (f *Field) Trace() []float64 {
	out := make([]float64, len(f.F))
	for i := range out {
		out[i] = f.Fxx[i] + f.Fyy[i]
	}
	return out
}

// Mask 是任务书给出的逐格判据：L1<0 且 |L1|>|L2|。
//
// 该判据与 Δf<0 恒等（见 docs/ridges.md 引理 1），
// 因此它只标记“净凹”区域，不是脊线曲线。
func (f *Field) Mask() []bool {
	m := make([]bool, len(f.F))
	for i := range m {
		m[i] = f.L1[i] < 0 && math.Abs(f.L1[i]) > math.Abs(f.L2[i])
	}
	return m
}

// Bilinear 对逐点场做双线性插值，坐标会被夹到网格范围内。
func (f *Field) Bilinear(v []float64, x, y float64) float64 {
	n := f.N
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	if x > float64(n-1) {
		x = float64(n - 1)
	}
	if y > float64(n-1) {
		y = float64(n - 1)
	}
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	if x0 > n-2 {
		x0 = n - 2
	}
	if y0 > n-2 {
		y0 = n - 2
	}
	fx := x - float64(x0)
	fy := y - float64(y0)
	i00 := y0*n + x0
	i10 := i00 + 1
	i01 := i00 + n
	i11 := i01 + 1
	return v[i00]*(1-fx)*(1-fy) + v[i10]*fx*(1-fy) + v[i01]*(1-fx)*fy + v[i11]*fx*fy
}
