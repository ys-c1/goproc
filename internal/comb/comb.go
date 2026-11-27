// Package comb 构造多层叠加产生梳齿（同相列）的受控反例并检测梳齿。
//
// 模型：两层一维余弦波沿 x 方向叠加，再用弱二维值噪声做纵向调制，
// 使每条同相亮列呈现为有起伏的竖直亮段，而不是无结构的整片亮带。
//
//	h(x,y) = A[cos(2π f1 x) + cos(2π f2 x + δ)] + w·n(x/s, y/s)
//
// 其中 f2 = r·f1，r 为两层频率之比，δ 为第二层的全局相位。
package comb

import (
	"fmt"
	"math"

	"goproc/internal/noise"
)

// Params 是一次梳齿实验的全部参数。
type Params struct {
	N          int     // 图像边长（像素），实验取 256
	F1         float64 // 第一层频率（周期/像素），实验取 1/16
	A          float64 // 每层余弦振幅，实验取 1
	W          float64 // 值噪声调制幅度，实验取 0.10
	NoiseScale float64 // 值噪声特征尺度（像素），实验取 4
	T          float64 // 点亮阈值（原始场），实验取 1.70
	HMin       int     // 段高阈值：同列连续点亮像素数下限，实验取 64
	MergeGap   int     // 相邻多少列内的合格亮段合并为同一梳齿，实验取 8
	PhiTol     float64 // 相位判据容差（弧度），实验取 1 度
	Seed       uint64  // 值噪声种子
}

// DefaultParams 返回标定后的实验参数。
func DefaultParams() Params {
	return Params{
		N: 256, F1: 1.0 / 16.0, A: 1.0, W: 0.10, NoiseScale: 4.0,
		T: 1.70, HMin: 64, MergeGap: 8, PhiTol: math.Pi / 180.0, Seed: 11,
	}
}

// Group 描述一组频率比与相位。
type Group struct {
	Name  string
	R     float64 // f2/f1
	Delta float64 // 第二层全局相位（弧度）
}

// Tooth 是一个被检出的梳齿。
type Tooth struct {
	Center float64 // 合并后亮列中心（像素，可为半像素）
	Height int     // 该簇内最高亮段的段高
}

// Result 是一组实验的检测结果。
type Result struct {
	Group          Group
	RawRuns        int     // 几何检测器报告的合格亮段总数
	Teeth          []Tooth // 通过相位判据并合并后的梳齿
	Predicted      int     // 理论预测梳齿数
	SpacingPred    float64 // 理论间距（像素），无理数组为 0
	SpacingMeas    []float64
	MinResidualDeg float64 // 无理数组：c>=1 列的最小相位差（度）
}

// Field 保存一次实验的原始高度场。
type Field struct {
	P        Params
	R, Delta float64
	H        []float64 // 行主序 N*N
}

func wrapPi(d float64) float64 {
	for d > math.Pi {
		d -= 2 * math.Pi
	}
	for d < -math.Pi {
		d += 2 * math.Pi
	}
	return d
}

// PhaseDiff 返回第 c 列处两层相位差，wrap 到 [-π, π]。
//
//	Δφ(c) = 2π(r−1)f1·c + δ
func (p Params) PhaseDiff(c int, r, delta float64) float64 {
	return wrapPi(2*math.Pi*(r-1)*p.F1*float64(c) + delta)
}

// GeomEpsilon 返回点亮阈值 T 隐含的几何相位容差：
//
//	2A·cos(|δ|/2) − w ≥ T  ⟺  |δ| ≤ 2·acos((T+w)/(2A))
//
// 该容差只用于相位扫描的理论对照；分类使用更严格的 PhiTol。
func (p Params) GeomEpsilon() float64 {
	return 2 * math.Acos((p.T+p.W)/(2*p.A))
}

// Build 按频率比 r 与相位 delta 构造高度场。
func Build(p Params, r, delta float64) *Field {
	src := noise.New(p.Seed)
	h := make([]float64, p.N*p.N)
	f2 := r * p.F1
	for y := 0; y < p.N; y++ {
		for x := 0; x < p.N; x++ {
			l1 := math.Cos(2 * math.Pi * p.F1 * float64(x))
			l2 := math.Cos(2*math.Pi*f2*float64(x) + delta)
			nv := src.Value(float64(x)/p.NoiseScale, float64(y)/p.NoiseScale)
			h[y*p.N+x] = p.A*(l1+l2) + p.W*nv
		}
	}
	return &Field{P: p, R: r, Delta: delta, H: h}
}

// litRuns 逐列扫描高度场，返回段高 ≥ HMin 的连续点亮段：[(列, 段高)]。
func (f *Field) litRuns() [][2]int {
	var runs [][2]int
	for c := 0; c < f.P.N; c++ {
		run := 0
		for y := 0; y < f.P.N; y++ {
			if f.H[y*f.P.N+c] >= f.P.T {
				run++
				continue
			}
			if run >= f.P.HMin {
				runs = append(runs, [2]int{c, run})
			}
			run = 0
		}
		if run >= f.P.HMin {
			runs = append(runs, [2]int{c, run})
		}
	}
	return runs
}

// Detect 执行"几何亮段 ∧ 精确相位"双条件检测，合并相邻列，排除边界瞬态。
func (f *Field) Detect() (rawRuns int, teeth []Tooth) {
	runs := f.litRuns()
	rawRuns = len(runs)

	// 条件 2（相位判据）：|Δφ(c)| ≤ PhiTol
	type seg struct{ c, h int }
	var aligned []seg
	for _, rn := range runs {
		if math.Abs(f.P.PhaseDiff(rn[0], f.R, f.Delta)) <= f.P.PhiTol {
			aligned = append(aligned, seg{rn[0], rn[1]})
		}
	}

	// 合并相邻列（同一条余弦亮峰可跨 1–3 列）为梳齿簇
	type cluster struct {
		cols []seg
	}
	var clusters []cluster
	for _, s := range aligned {
		if len(clusters) > 0 && s.c-clusters[len(clusters)-1].cols[len(clusters[len(clusters)-1].cols)-1].c <= f.P.MergeGap {
			clusters[len(clusters)-1].cols = append(clusters[len(clusters)-1].cols, s)
		} else {
			clusters = append(clusters, cluster{cols: []seg{s}})
		}
	}

	// 边界瞬态排除：全图仅一个同相簇且贴在 c=0 边上时，
	// 它是 δ=0 在原点的平凡重合，窗口内无公共周期的第二个证据，不计梳齿。
	if len(clusters) == 1 && clusters[0].cols[0].c <= f.P.MergeGap {
		return rawRuns, nil
	}

	for _, cl := range clusters {
		sum, maxH := 0, 0
		for _, s := range cl.cols {
			sum += s.c
			if s.h > maxH {
				maxH = s.h
			}
		}
		teeth = append(teeth, Tooth{
			Center: float64(sum) / float64(len(cl.cols)),
			Height: maxH,
		})
	}
	return rawRuns, teeth
}

// MinResidual 返回 c≥1 各列中最小的 |Δφ|（度）。c=0 在 δ=0 时恒为零，无信息量，故排除。
func (p Params) MinResidual(r, delta float64) float64 {
	best := math.Pi
	for c := 1; c < p.N; c++ {
		if d := math.Abs(p.PhaseDiff(c, r, delta)); d < best {
			best = d
		}
	}
	return best * 180 / math.Pi
}

// rational 用连分数把 r 展开为既约分数 p/q，分母上限 maxQ。
// 返回 p、q 以及该收敛分数与 r 的误差。
func rational(r float64, maxQ int) (p, q int, err float64) {
	// 连分数展开
	rem := r
	var cf []int
	for i := 0; i < 64; i++ {
		a := int(math.Floor(rem))
		cf = append(cf, a)
		frac := rem - float64(a)
		if math.Abs(frac) < 1e-12 {
			break
		}
		rem = 1 / frac
	}
	// 收敛分数递推，取分母不超过 maxQ 的最后一个
	np, nq := 1, 0
	dp, dq := 0, 1
	for _, a := range cf {
		pp := a*np + dp
		qq := a*nq + dq
		if qq > maxQ {
			break
		}
		dp, dq = np, nq
		np, nq = pp, qq
	}
	return np, nq, math.Abs(float64(np)/float64(nq) - r)
}

// PredictRational 对有理频率比 p/q 预测同相列：
// 间距 d = q/f1 像素，同相列 c = k·d；要求窗口内至少两列，否则预测 0（边界瞬态）。
func (p Params) PredictRational(r float64) (num, den int, spacing float64, count int) {
	num, den, _ = rational(r, 10000)
	spacing = float64(den) / p.F1
	c := 0
	for c < p.N {
		if math.Abs(p.PhaseDiff(c, r, 0)) > 1e-9 {
			break // 理论上不会发生，仅防御浮点
		}
		count++
		c += int(math.Round(spacing))
	}
	if count < 2 {
		count = 0
	}
	return num, den, spacing, count
}

// IsRational 判断 r 在实验精度内是否为有理数（连分数精确命中）。
func IsRational(r float64) bool {
	_, _, err := rational(r, 10000)
	return err < 1e-12
}

// RunGroup 构造并检测一组实验，附带理论预测。
func RunGroup(p Params, g Group) *Result {
	f := Build(p, g.R, g.Delta)
	raw, teeth := f.Detect()
	res := &Result{Group: g, RawRuns: raw, Teeth: teeth, MinResidualDeg: p.MinResidual(g.R, g.Delta)}

	// 有理数组（A、C）给预测；无理数组（B）预测为 0
	if IsRational(g.R) {
		_, _, spacing, count := p.PredictRational(g.R)
		res.Predicted = count
		res.SpacingPred = spacing
	}
	for i := 1; i < len(teeth); i++ {
		res.SpacingMeas = append(res.SpacingMeas, teeth[i].Center-teeth[i-1].Center)
	}
	return res
}

// StandardGroups 返回三组正式实验。
func StandardGroups() []Group {
	return []Group{
		{Name: "A counterexample r=4/1 delta=0", R: 4.0, Delta: 0},
		{Name: "B irrational r=sqrt(2) delta=0", R: math.Sqrt2, Delta: 0},
		{Name: "C rational r=4/3 delta=0", R: 4.0 / 3.0, Delta: 0},
	}
}

// Diff 返回实测梳齿数与预测值之差的绝对值。
func (r *Result) Diff() int {
	d := len(r.Teeth) - r.Predicted
	if d < 0 {
		d = -d
	}
	return d
}

// String 给出一行汇总。
func (r *Result) String() string {
	return fmt.Sprintf("%-34s raw=%3d teeth=%2d pred=%2d diff=%d",
		r.Group.Name, r.RawRuns, len(r.Teeth), r.Predicted, r.Diff())
}

// 显示窗：高度场理论范围约 [-2-w, 2+w]，固定映射到 [0,255]，
// 不做逐图 min-max 归一化，保证三组图的灰度口径一致、阈值可复现。
const displayWindow = 2.15

// EncodePGM 把高度场编码为八位灰度 PGM（P5）。
func (f *Field) EncodePGM() []byte {
	head := fmt.Sprintf("P5\n%d %d\n255\n", f.P.N, f.P.N)
	out := make([]byte, 0, len(head)+len(f.H))
	out = append(out, head...)
	for _, v := range f.H {
		g := int((v+displayWindow)/(2*displayWindow)*255 + 0.5)
		if g < 0 {
			g = 0
		}
		if g > 255 {
			g = 255
		}
		out = append(out, byte(g))
	}
	return out
}
