// Package fog 实现山地雾衰减、可见比例的高度分布预判，以及带雾灰度渲染。
//
// 物理模型（Beer–Lambert / Koschmieder 大气透视）：
//
//	对比度透射率 C(p) = exp(-beta * L(p))
//
// 其中 beta 是雾浓度（消光系数），L 是该像素对应气柱中的光程。
// 取雾从地面一直充填到山顶参考面 hRef=1（归一化高度），则
//
//	L(p) = R * (1 - h(p))
//	C(h) = exp(-chi * (1 - h)), chi = beta * R
//
// 可见判据为 C > epsilon（对比度阈值），等价于光学厚度
// tau = chi*(1-h) < tau* = -ln(epsilon)。
package fog

import (
	"math"
	"sort"
	"strconv"
)

// DefaultEpsilon 是默认的最小可分辨对比度（Koschmieder/WMO 常用 5%）。
const DefaultEpsilon = 0.05

// TauThreshold 返回对比度阈值对应的光学厚度界限 -ln(eps)。
func TauThreshold(eps float64) float64 { return -math.Log(eps) }

// Chi 返回无量纲乘积：雾浓度 beta 乘地形起伏 R。
func Chi(beta, relief float64) float64 { return beta * relief }

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// ContrastAt 返回归一化高度 h 处的残余对比度 C=exp(-chi*(1-h))。
func ContrastAt(h, chi float64) float64 {
	return math.Exp(-chi * (1 - clamp01(h)))
}

// HeightCutoff 返回可见所需的严格高度下限：h > 1 - tau*/chi。
// 当 chi <= tau* 时最低点也满足阈值，截断值为负，全部可见。
func HeightCutoff(chi, eps float64) float64 {
	return 1 - TauThreshold(eps)/chi
}

// Count 是一次可见比例统计的完整结果。
type Count struct {
	Visible      int     `json:"visible"`
	Total        int     `json:"total"`
	Fraction     float64 `json:"fraction"`
	Chi          float64 `json:"chi"`
	Epsilon      float64 `json:"epsilon"`
	TauStar      float64 `json:"tau_star"`
	CutoffHeight float64 `json:"cutoff_height"`
}

// Predict 仅凭高度分布（逐高度值的经验分布）与 chi 预判可见比例，
// 不做任何渲染。判据与渲染器完全一致：ContrastAt(h,chi) > eps（严格高于）。
func Predict(heights []float64, chi, eps float64) Count {
	cut := HeightCutoff(chi, eps)
	n := 0
	for _, h := range heights {
		if ContrastAt(h, chi) > eps {
			n++
		}
	}
	return Count{
		Visible:      n,
		Total:        len(heights),
		Fraction:     float64(n) / float64(len(heights)),
		Chi:          chi,
		Epsilon:      eps,
		TauStar:      TauThreshold(eps),
		CutoffHeight: cut,
	}
}

// PredictViaCDF 是“按高度分布预判”的严格实现：
// 先把高度排序得到经验 CDF，再查严格高于截断高度 hCut 的质量比例。
// 全程不接触任何渲染/图像缓冲。
func PredictViaCDF(heights []float64, chi, eps float64) Count {
	sorted := append([]float64(nil), heights...)
	sort.Float64s(sorted)
	cut := HeightCutoff(chi, eps)
	// 第一个严格大于 cut 的位置；其后所有高度可见（与 C>eps 严格判据一致）。
	firstAbove := sort.Search(len(sorted), func(i int) bool { return sorted[i] > cut })
	vis := len(sorted) - firstAbove
	return Count{
		Visible:      vis,
		Total:        len(sorted),
		Fraction:     float64(vis) / float64(len(sorted)),
		Chi:          chi,
		Epsilon:      eps,
		TauStar:      TauThreshold(eps),
		CutoffHeight: cut,
	}
}

// Histogram 是等宽高度直方图。
type Histogram struct {
	Bins  int     `json:"bins"`
	Lo    float64 `json:"lo"`
	Hi    float64 `json:"hi"`
	Count []int   `json:"count"`
}

// BuildHistogram 在 [0,1] 上构造等宽直方图（渲染前即可得到）。
func BuildHistogram(heights []float64, bins int) Histogram {
	hist := Histogram{Bins: bins, Lo: 0, Hi: 1, Count: make([]int, bins)}
	for _, h := range heights {
		b := int(clamp01(h) * float64(bins))
		if b >= bins {
			b = bins - 1
		}
		hist.Count[b]++
	}
	return hist
}

// PredictHistogramStep 是“按高度分布预判”的粗粒度版本：
// 整个直方柱的中点高度代表该柱。用于量化分布分箱粒度带来的偏差。
func PredictHistogramStep(hist Histogram, chi, eps float64) Count {
	total := 0
	vis := 0
	for b, cnt := range hist.Count {
		total += cnt
		h := (float64(b) + 0.5) / float64(hist.Bins)
		if ContrastAt(h, chi) > eps {
			vis += cnt
		}
	}
	frac := 0.0
	if total > 0 {
		frac = float64(vis) / float64(total)
	}
	return Count{
		Visible:      vis,
		Total:        total,
		Fraction:     frac,
		Chi:          chi,
		Epsilon:      eps,
		TauStar:      TauThreshold(eps),
		CutoffHeight: HeightCutoff(chi, eps),
	}
}

// HeightStats 是地形高度分布与起伏标量的统计。
type HeightStats struct {
	N        int                `json:"n"`
	Min      float64            `json:"min"`
	Max      float64            `json:"max"`
	Range    float64            `json:"range"`
	Mean     float64            `json:"mean"`
	Std      float64            `json:"std"`
	Quantile map[string]float64 `json:"quantile"`
}

// Quantile 取已排序样本的 p 分位数（最近秩法）。
func Quantile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	i := int(p*float64(len(sorted)-1) + 0.5)
	if i < 0 {
		i = 0
	}
	if i >= len(sorted) {
		i = len(sorted) - 1
	}
	return sorted[i]
}

// Stats 计算高度分布统计与地形起伏（极差、标准差）。
func Stats(heights []float64) HeightStats {
	st := HeightStats{N: len(heights), Quantile: map[string]float64{}}
	if len(heights) == 0 {
		return st
	}
	lo, hi := heights[0], heights[0]
	sum := 0.0
	for _, h := range heights {
		if h < lo {
			lo = h
		}
		if h > hi {
			hi = h
		}
		sum += h
	}
	st.Min, st.Max = lo, hi
	st.Range = hi - lo
	st.Mean = sum / float64(len(heights))
	ss := 0.0
	for _, h := range heights {
		d := h - st.Mean
		ss += d * d
	}
	st.Std = math.Sqrt(ss / float64(len(heights)))

	sorted := append([]float64(nil), heights...)
	sort.Float64s(sorted)
	for _, p := range []float64{0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99} {
		st.Quantile[quantileKey(p)] = Quantile(sorted, p)
	}
	return st
}

func quantileKey(p float64) string {
	switch p {
	case 0.01:
		return "p01"
	case 0.05:
		return "p05"
	case 0.10:
		return "p10"
	case 0.25:
		return "p25"
	case 0.50:
		return "p50"
	case 0.75:
		return "p75"
	case 0.90:
		return "p90"
	case 0.95:
		return "p95"
	case 0.99:
		return "p99"
	}
	return "p" + strconv.FormatFloat(p*100, 'f', 0, 64)
}

// PredictUniform 是“只靠乘积 + 高度在 [0,1] 均匀分布”的标量速判：
// 可见比例 = 1 - hCut = tau*/chi（chi>=tau*），否则为 1。
func PredictUniform(chi, eps float64) float64 {
	tau := TauThreshold(eps)
	if chi <= tau {
		return 1
	}
	return tau / chi
}

// PredictGaussian 用高度分布的均值与标准差做正态近似，
// 按 h > hCut 求高斯右尾概率。这是“只使用分布参数”的预判。
func PredictGaussian(mean, std, chi, eps float64) float64 {
	if std <= 0 {
		if HeightCutoff(chi, eps) <= mean {
			return 1
		}
		return 0
	}
	z := (HeightCutoff(chi, eps) - mean) / std
	return 0.5 * math.Erfc(z/math.Sqrt2)
}

// RelativeDiff 返回 |a-b|/b（以渲染实测 b 为基准）；b 为零时返回 NaN。
func RelativeDiff(pred, measured float64) float64 {
	if measured == 0 {
		return math.NaN()
	}
	return math.Abs(pred-measured) / measured
}
