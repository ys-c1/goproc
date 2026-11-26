// Package tilt 由谱斜率给出平均倾角的闭式预判，并统计已生成地形的倾角。
//
// 推导要点见各符号注释：值噪声的单倍频程梯度模为 Kappa 倍频率，二维各向
// 同性梯度模服从 Rayleigh 分布，均值是均方根的 sqrt(pi/2) 倍。于是全图
// 平均倾角只由基础频率、层数、频率倍增与振幅衰减决定，不必先生成地形。
//
// 约定：倾角一律按度返回；统计时排除边界一圈格子（该圈只有单侧邻居，
// 中心差分退化）。预判值对应「未归一化」的高度场；Generate 输出的归一化
// 图会按极差 R 整体缩放梯度，实际倾角以 atan(tan(预判)/R) 计。
package tilt

import "math"

// Kappa 是单位频率值噪声的连续梯度模均方根（每单位频率）。
//
// 角点取值 g = k/127.5-1（k=0..255）等概率，方差 sigma_c^2 = 21845/65025。
// 对三次 fade 多项式 u(t)=6t^5-15t^4+10t^3：
//
//	E[u'^2] = 10/7
//	E[u] = 1/2,  E[u^2] = 36/11-18+345/9-37.5+100/7 = 0.391774
//	Var_x(u) = (2 sigma_c^2)(1-2E[u]+2E[u^2])
//	Kappa^2 = 2 * E[u'^2] * Var_x(u) = 1.504178
//
// 即 Kappa ≈ 1.22645。对原始噪声用极小步长数值微分实测约 1.23，差在 1% 内。
const Kappa = 1.22645

// MeanOverRMS 是二维各向同性高斯梯度的 Rayleigh 模的 mean/RMS：
//
//	E|G| / sqrt(E|G|^2) = sqrt(pi/2)
const MeanOverRMS = 0.8862269254527580

// SlopePerFrequency 是「每单位频率、单位权重」贡献到平均斜率的比例，
// 等于 Kappa*MeanOverRMS ≈ 1.086913，单位是「每单位频率的平均斜率」。
//
// 于是一档的等效平均斜率 = SlopePerFrequency * weight * frequency，
// 全图平均倾角 = atan(合成斜率) 换算成度。
const SlopePerFrequency = Kappa * MeanOverRMS

// Band 是一个倍频程档位的谱斜率与倾角预判。
type Band struct {
	Index     int     // 档位，从 0 起
	Frequency float64 // 该档频率 f0*L^k，每像素周期数
	Weight    float64 // 该档振幅权重 g^k / sum_j g^j
	Slope     float64 // 该档等效平均斜率 SlopePerFrequency*weight*frequency
	TiltDeg   float64 // 该档等效倾角 atan(Slope)，度
}

// Describe 逐档给出谱斜率与倾角预判，全程不生成地形。
//
// 各档梯度互相独立，合成斜率按平方和叠加；单档倾角是该档单独存在时的
// 等效角度，角度本身不可相加，逐档值只用于与实测逐档比对。
func Describe(baseFrequency float64, octaves int, lacunarity, gain float64) []Band {
	if octaves <= 0 {
		return nil
	}
	norm := 0.0
	for k, a := 0, 1.0; k < octaves; k, a = k+1, a*gain {
		norm += a
	}
	out := make([]Band, 0, octaves)
	for k, a, f := 0, 1.0, baseFrequency; k < octaves; k, a, f = k+1, a*gain, f*lacunarity {
		w := a / norm
		slope := SlopePerFrequency * w * f
		out = append(out, Band{
			Index:     k,
			Frequency: f,
			Weight:    w,
			Slope:     slope,
			TiltDeg:   Degrees(math.Atan(slope)),
		})
	}
	return out
}

// Overview 返回合成谱斜率与平均倾角预判（度）。
//
//	slope = sqrt( sum_k (SlopePerFrequency * g^k/sum(g^j) * f0*L^k)^2 )
//	tilt  = atan(slope)
//
// 这是「斜率是基础频率的倍数，再取反正切」的闭式：所有 f_k 都正比于 f0，
// 故 slope 也正比于 f0。若需在归一化后的图上用，见 NormalizedTilt。
func Overview(baseFrequency float64, octaves int, lacunarity, gain float64) (slope, tiltDeg float64) {
	sum := 0.0
	for _, b := range Describe(baseFrequency, octaves, lacunarity, gain) {
		sum += b.Slope * b.Slope
	}
	slope = math.Sqrt(sum)
	return slope, Degrees(math.Atan(slope))
}

// NormalizedTilt 把未归一化预判斜率 slope 换算成归一化图（极差为 span）上的倾角。
//
// 归一化把每个梯度除以 span，斜率尺度随之为 1/span。
func NormalizedTilt(slope, span float64) float64 {
	if span <= 0 {
		return 0
	}
	return math.Atan(slope/span) * 180 / math.Pi
}

// Degrees 把弧度换算成度。
func Degrees(rad float64) float64 { return rad * 180 / math.Pi }

// Radians 把度换算成弧度。
func Radians(deg float64) float64 { return deg * math.Pi / 180 }
