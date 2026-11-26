// Package slope 从谱与极值理论预测 fBm 高度图的坡度分布，并提供实测工具。
//
// 理论链条（详见 docs/slope-theory.md）：
//  1. 单层值噪声的梯度是有界独立变量的线性组合，渐近正态，
//     方差为谱的二阶矩，可由谱斜率积出；
//  2. 多层梯度在同一点几乎处处不共享角点，近似独立，合成仍正态，
//     模长服从 Rayleigh 分布；
//  3. N 个样本的最大值期望由重现格数 R 对应的上分位给出，
//     本包取 R = N/1000（总格数的千分之一）。
package slope

import (
	"math"

	"goproc/api"
	"goproc/internal/fbm"
	"goproc/internal/hmap"
	"goproc/internal/noise"
)

// cornerVar 是角点值 c = h/127.5 − 1 的方差，
// h 为 0..255 的均匀置换：Var(c) = (256²−1)/12 / 127.5²。
var cornerVar = (256*256 - 1) / 12.0 / (127.5 * 127.5)

// CornerVar 暴露角点方差供测试对照。
func CornerVar() float64 { return cornerVar }

func fade(t float64) float64 { return t * t * t * (t*(t*6-15) + 10) }

// phi 是值噪声的一维插值核：d∈[0,1] 时 1−fade(d)，d∈[−1,0] 时 fade(1+d)。
func phi(d float64) float64 {
	if d < -1 || d > 1 {
		return 0
	}
	if d < 0 {
		return fade(1 + d)
	}
	return 1 - fade(d)
}

// Phi2 是 ∫φ²du 的解析值 181/231（命题 1 的 m₂）。
const Phi2 = 181.0 / 231.0

// FDKernelEnergy 返回 D(f) = ∫ [(φ(u+f)−φ(u−f))/2]² du，
// 即频率 f（周/像素）处中心差分作用在插值核上的能量。
// f→0 时 D(f) → f²·∫φ'² = f²·20/7；f 超过 Nyquist 后自动包含混叠效应。
func FDKernelEnergy(f float64) float64 {
	if f == 0 {
		return 0
	}
	if f < 0 {
		f = -f
	}
	lo, hi := -1-f, 1+f
	n := int((hi-lo)/1e-3) + 1
	h := (hi - lo) / float64(n)
	sum := 0.0
	for i := 0; i <= n; i++ {
		u := lo + float64(i)*h
		d := (phi(u+f) - phi(u-f)) / 2
		w := 1.0
		if i == 0 || i == n {
			w = 0.5
		}
		sum += w * d * d
	}
	return sum * h
}

// Params 是 fBm 参数，与 api.Options 中的对应字段一致。
type Params struct {
	F0   float64 // 基础频率，周/像素
	Lac  float64 // 频率倍增
	Gain float64 // 振幅衰减
}

// DefaultParams 与 api.DefaultOptions 一致。
func DefaultParams() Params { return Params{F0: 0.02, Lac: 2, Gain: 0.5} }

func octaves(K int, p Params) (amps, freqs []float64, norm float64) {
	amps = make([]float64, K)
	freqs = make([]float64, K)
	a, f := 1.0, p.F0
	for i := 0; i < K; i++ {
		amps[i], freqs[i] = a, f
		norm += a
		a *= p.Gain
		f *= p.Lac
	}
	return amps, freqs, norm
}

// SigmaFD 预测原始（未归一化）fBm 在网格中心差分下单分量梯度的标准差：
// σ_K² = σ_c²·Φ₂·Σ a_i²D(f_i) / A_K²，A_K = Σa_i。
func SigmaFD(K int, p Params) float64 {
	amps, freqs, norm := octaves(K, p)
	v := 0.0
	for i := 0; i < K; i++ {
		v += amps[i] * amps[i] * FDKernelEnergy(freqs[i])
	}
	return math.Sqrt(cornerVar * Phi2 * v) / norm
}

// SigmaZ 预测原始 fBm 高度值的标准差：σ_z = σ_c·Φ₂·√(Σa_i²)/A_K。
func SigmaZ(K int, p Params) float64 {
	amps, _, norm := octaves(K, p)
	v := 0.0
	for _, a := range amps {
		v += a * a
	}
	return math.Sqrt(cornerVar) * Phi2 * math.Sqrt(v) / norm
}

// NEff 是高度值极值的有效独立样本数，取基频格数 (S·f₀)²。
func NEff(size int, f0 float64) float64 {
	c := float64(size) * f0
	return c * c
}

// PredRange 预测归一化极差 E[R] ≈ 2σ_z·√(2 ln N_eff)（对称高斯场极差近似）。
func PredRange(K int, p Params, nEff float64) float64 {
	return 2 * SigmaZ(K, p) * math.Sqrt(2*math.Log(nEff))
}

// PredS 预测归一化导出图的单分量梯度标准差 s_K = σ_K / E[R]。
func PredS(K int, p Params, nEff float64) float64 {
	return SigmaFD(K, p) / PredRange(K, p, nEff)
}

// RayleighQuantile 返回重现格数 r 对应的上分位 s·√(2 ln r)。
func RayleighQuantile(s, r float64) float64 {
	return s * math.Sqrt(2*math.Log(r))
}

// ExceedProb 是 Rayleigh(s) 超过 g 的概率 exp(−g²/2s²)。
func ExceedProb(s, g float64) float64 {
	return math.Exp(-g * g / (2 * s * s))
}

// SlopesXY 返回归一化高度图内部像素的中心差分梯度分量（每像素）。
func SlopesXY(data []float64, size int) (dx, dy []float64) {
	n := size - 2
	dx = make([]float64, 0, n*n)
	dy = make([]float64, 0, n*n)
	for y := 1; y < size-1; y++ {
		for x := 1; x < size-1; x++ {
			dx = append(dx, (data[y*size+x+1]-data[y*size+x-1])/2)
			dy = append(dy, (data[(y+1)*size+x]-data[(y-1)*size+x])/2)
		}
	}
	return dx, dy
}

// Slopes 返回坡度角正切 |∇h|（垂直单位 = 整个值域，水平单位 = 1 像素）。
func Slopes(data []float64, size int) []float64 {
	dx, dy := SlopesXY(data, size)
	out := make([]float64, len(dx))
	for i := range out {
		out[i] = math.Hypot(dx[i], dy[i])
	}
	return out
}

// Quantile 返回已排序样本的 p 分位（线性插值）。
func Quantile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	pos := p * float64(len(sorted)-1)
	i := int(pos)
	if i >= len(sorted)-1 {
		return sorted[len(sorted)-1]
	}
	frac := pos - float64(i)
	return sorted[i]*(1-frac) + sorted[i+1]*frac
}

// Generate 复刻 api.Generate 的流程，同时返回归一化前的极差，
// 供对照理论中的归一化修正。输出与 api.Generate 完全一致（有测试保证）。
func Generate(opt api.Options) ([]float64, float64) {
	src := noise.New(opt.Seed)
	out := make([]float64, opt.Size*opt.Size)
	for y := 0; y < opt.Size; y++ {
		for x := 0; x < opt.Size; x++ {
			out[y*opt.Size+x] = fbm.Eval(src, float64(x)*opt.Frequency, float64(y)*opt.Frequency,
				opt.Octaves, opt.Lacunarity, opt.Gain)
		}
	}
	lo, hi := out[0], out[0]
	for _, v := range out {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	rawRange := hi - lo
	hmap.Normalize(out)
	return out, rawRange
}
