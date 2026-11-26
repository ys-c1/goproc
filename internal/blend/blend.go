// Package blend 处理低频/高频两层地形在过渡带 [a,b] 内的一维混合，
// 并给出以“棱能量”为目标泛函时、端点固定约束下的最优权重。
//
// 记号（沿混合方向 x，逐行 y 独立处理）：
//
//	L(x)  低频层，H(x)  高频层（入口处已对齐：H(a)=L(a)）
//	Δ(x) = H(x)-L(x)，故 Δ(a)=0 且 Δ'(x)=q(x)=H'(x)-L'(x)
//	w(x)  混合权重，w(a)=0、w(b)=1；混合 B=(1-w)L+wH
//
// 混合相对理想带限结果 B*（满足 B*'=(1-w)L'+wH'、B*(a)=L(a)）的差为
//
//	B-B* = ∫_a^x w'Δ ds = wΔ - ∫_a^x w q ds，
//
// 而可见的棱就是混合梯度中多出的一项 r=w'Δ。
package blend

import "math"

// Weight 把归一化坐标 t∈[0,1] 映到权重 [0,1]。
type Weight func(t float64) float64

// Linear 是线性权重 w=t。
func Linear(t float64) float64 { return t }

// Smoothstep 是经典三次平滑权重 3t²-2t³，端点导数为零。
func Smoothstep(t float64) float64 { return t * t * (3 - 2*t) }

// OptimalWeights 解正则化后的最小棱能量问题
//
//	min dx·Σ (Δmid_i²+eps²)(w'_i)²,  w_0=0, w_m=1,
//
// 连续形式为 E[w]=∫(w'Δ)² dx，Euler-Lagrange 方程 (Δ²w')'=0，
// 即 Δ²w'=const；离散解为
//
//	w_k = Σ_{i<k} 1/c_i / Σ_j 1/c_j,  c_i = Δmid_i² + eps²。
//
// deltaMid 为过渡带边中点层差（长度 m），返回长度 m+1 的权重，
// 首尾严格为 0、1（最后一个分量强制归一，消除舍入误差）。
func OptimalWeights(deltaMid []float64, eps2 float64) []float64 {
	m := len(deltaMid)
	w := make([]float64, m+1)
	inv := make([]float64, m)
	total := 0.0
	for i, d := range deltaMid {
		c := d*d + eps2
		inv[i] = 1 / c
		total += inv[i]
	}
	acc := 0.0
	for i := 0; i < m; i++ {
		acc += inv[i]
		w[i+1] = acc / total
	}
	w[m] = 1.0
	return w
}

// Eps2FromD 按相对阈值 rel 把正则项折算成 Δ 的量级：eps2=(rel·rms(Δmid))²。
func Eps2FromD(deltaMid []float64, rel float64) float64 {
	sum := 0.0
	for _, d := range deltaMid {
		sum += d * d
	}
	rms := math.Sqrt(sum / float64(len(deltaMid)))
	e := rel * rms
	return e * e
}

// BlendRow 用给定函数型权重混合整行，a、b 为过渡带端点下标（含）。
// 行外在 x<a 取 L、x>b 取 H；调用方需保证 H[a]=L[a]。
func BlendRow(l, h []float64, a, b int, wf Weight) []float64 {
	out := make([]float64, len(l))
	for i := range l {
		switch {
		case i < a:
			out[i] = l[i]
		case i > b:
			out[i] = h[i]
		default:
			t := float64(i-a) / float64(b-a)
			w := wf(t)
			out[i] = (1-w)*l[i] + w*h[i]
		}
	}
	return out
}

// BlendRowSamples 用逐采样权重 w（长度 b-a+1）混合整行。
func BlendRowSamples(l, h []float64, a, b int, w []float64) []float64 {
	out := make([]float64, len(l))
	for i := range l {
		switch {
		case i < a:
			out[i] = l[i]
		case i > b:
			out[i] = h[i]
		default:
			wi := w[i-a]
			out[i] = (1-wi)*l[i] + wi*h[i]
		}
	}
	return out
}

// EdgeMidDeltas 返回过渡带内各边中点的层差 Δmid，长度 m=b-a。
func EdgeMidDeltas(l, h []float64, a, b int) []float64 {
	m := b - a
	d := make([]float64, m)
	for i := 0; i < m; i++ {
		d0 := h[a+i] - l[a+i]
		d1 := h[a+i+1] - l[a+i+1]
		d[i] = 0.5 * (d0 + d1)
	}
	return d
}

// RidgeEnergy 计算目标泛函 E=dx·Σ (Δmid_i²+eps²)(w'_i)²，
// 与 OptimalWeights 最小化的量完全一致；eps2=0 时退化为可见棱能量。
// w 为长度 m+1 的采样权重。
func RidgeEnergy(w, deltaMid []float64, eps2, dx float64) float64 {
	e := 0.0
	for i := range deltaMid {
		wp := (w[i+1] - w[i]) / dx
		r2 := (deltaMid[i]*deltaMid[i] + eps2) * wp * wp
		e += r2
	}
	return e * dx
}

// VisibleRidge 只计可见棱能量 dx·Σ (w'_i·Δmid_i)²（不含正则项）。
func VisibleRidge(w, deltaMid []float64, dx float64) float64 {
	e := 0.0
	for i := range deltaMid {
		wp := (w[i+1] - w[i]) / dx
		r := wp * deltaMid[i]
		e += r * r
	}
	return e * dx
}

// CauchyBound 返回无正则时的理论最小能量 E*=1/(dx·Σ 1/Δmid_i²)，
// 由 Cauchy-Schwarz 下界取等得到；Δmid 含零时返回 NaN（下界发散）。
func CauchyBound(deltaMid []float64, dx float64) float64 {
	sum := 0.0
	for _, d := range deltaMid {
		if d == 0 {
			return math.NaN()
		}
		sum += 1 / (d * d)
	}
	return 1 / (dx * sum)
}

// RegularizedBound 返回正则化问题的理论最小能量
// E*_eps = 1/(dx·Σ 1/(Δmid_i²+eps²))，与 OptimalWeights 取等。
func RegularizedBound(deltaMid []float64, eps2, dx float64) float64 {
	sum := 0.0
	for _, d := range deltaMid {
		sum += 1 / (d*d + eps2)
	}
	return 1 / (dx * sum)
}

// SeamJumps 返回两个端点处的坡度跳变（外侧减内侧的一阶差分），
// 单位为高度/格。J[a] 用入口右侧减左侧，J[b] 同一定义，符号保留。
func SeamJumps(blended []float64, a, b int, dx float64) (ja, jb float64) {
	sOutL := (blended[a] - blended[a-1]) / dx
	sInL := (blended[a+1] - blended[a]) / dx
	ja = sInL - sOutL
	sInR := (blended[b] - blended[b-1]) / dx
	sOutR := (blended[b+1] - blended[b]) / dx
	jb = sOutR - sInR
	return ja, jb
}

// SampleWeight 把函数型权重按过渡带格点离散成长度 m+1 的切片。
func SampleWeight(wf Weight, a, b int) []float64 {
	m := b - a
	w := make([]float64, m+1)
	for i := range w {
		w[i] = wf(float64(i) / float64(m))
	}
	return w
}

// RMSWeightDistance 计算两条等长权重曲线的 RMS 距离。
func RMSWeightDistance(w1, w2 []float64) float64 {
	sum := 0.0
	for i := range w1 {
		d := w1[i] - w2[i]
		sum += d * d
	}
	return math.Sqrt(sum / float64(len(w1)))
}
