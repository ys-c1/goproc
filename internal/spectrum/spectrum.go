// Package spectrum 由高度图的二维功率谱预判法线朝向分布。
//
// 思路（详见 artifacts/PROOF.md）：
//   - 平稳高斯高度场的梯度 (p,q) 是零均值联合高斯，其协方差等于谱矩：
//     E[p²]=Mxx, E[q²]=Myy, E[pq]=Mxy，谱矩直接由 FFT 功率谱加权求和；
//   - 给定协方差即可解析求出梯度方位角落在任意角度扇区的概率，
//     而法线方位角与梯度方位角仅相差 π，扇区概率集合不变；
//   - 径向平均功率谱 P(w)~w^(-β) 的谱斜率 β 决定梯度能量，
//     理想各向同性 fBm 下任意扇区概率恒为箱宽/2π（八方位即 1/8），
//     二维谱矩则进一步捕捉实际实现带来的各向异性。
package spectrum

import (
	"math"

	"goproc/internal/dft"
)

// Moments 是由功率谱得到的梯度二阶矩（即梯度协方差）。
type Moments struct {
	Mxx, Myy, Mxy float64
	Windowed      bool // 是否使用 Hann 窗
}

// SpectralMoments 计算高度场的谱矩。windowed=true 时先加二维 Hann 窗抑制泄漏。
func SpectralMoments(size int, h []float64, windowed bool) Moments {
	if len(h) != size*size {
		panic("spectrum: 高度图尺寸不匹配")
	}
	work := make([]float64, size*size)
	mean := 0.0
	for _, v := range h {
		mean += v
	}
	mean /= float64(size * size)
	if windowed {
		for y := 0; y < size; y++ {
			wy := 0.5 * (1 - math.Cos(2*math.Pi*float64(y)/float64(size-1)))
			for x := 0; x < size; x++ {
				wx := 0.5 * (1 - math.Cos(2*math.Pi*float64(x)/float64(size-1)))
				work[y*size+x] = (h[y*size+x] - mean) * wx * wy
			}
		}
	} else {
		for i, v := range h {
			work[i] = v - mean
		}
	}
	spec := dft.FFT2(work, size)
	w := dft.Freq(size)
	// PSD 归一化 1/N⁴：二维 DFT 正逆各带 1/N，使 ΣPSD=均方能量（Parseval）。
	// 谱矩用中心差分传递函数 i·sin(w)（与 normal.FromHeightmap 的差分核严格一致），
	// 而非理想导数 iw——后者在高频段会系统性高估梯度能量。
	scale := 1 / float64(size*size*size*size)
	var mxx, myy, mxy float64
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			if x == 0 && y == 0 {
				continue // 去均值后直流为零，显式跳过
			}
			power := math.Pow(math.Hypot(real(spec[y*size+x]), imag(spec[y*size+x])), 2) * scale
			sx, sy := math.Sin(w[x]), math.Sin(w[y])
			mxx += sx * sx * power
			myy += sy * sy * power
			mxy += sx * sy * power
		}
	}
	return Moments{Mxx: mxx, Myy: myy, Mxy: mxy, Windowed: windowed}
}

// SlopeFit 是径向功率谱的幂律拟合结果 P(w) ∝ w^(-β)。
type SlopeFit struct {
	Beta       float64   // 谱斜率（取正号表示 P∝w^-β）
	RSquared   float64   // log-log 线性拟合决定系数
	W, LogP    []float64 // 参与拟合的径向频率与对数功率（报告用）
	Wmin, Wmax float64   // 拟合的惯性区间
}

// RadialSlope 做径向平均并在惯性区间 [wMin,wMax] 内做 log-log 最小二乘。
func RadialSlope(size int, h []float64, wMin, wMax float64) SlopeFit {
	work := make([]float64, size*size)
	mean := 0.0
	for _, v := range h {
		mean += v
	}
	mean /= float64(size * size)
	for i, v := range h {
		work[i] = v - mean
	}
	spec := dft.FFT2(work, size)
	w := dft.Freq(size)
	scale := 1 / float64(size*size) // PSD 归一化：E[h²]=Σ|H|²/N²

	nbin := size / 2
	binPow := make([]float64, nbin)
	binCnt := make([]int, nbin)
	binW := make([]float64, nbin)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			if x == 0 && y == 0 {
				continue
			}
			r := math.Hypot(w[x], w[y])
			b := int(r / (math.Pi / float64(nbin)))
			if b >= nbin {
				b = nbin - 1
			}
			if b < 0 {
				b = 0
			}
			binPow[b] += math.Pow(math.Hypot(real(spec[y*size+x]), imag(spec[y*size+x])), 2) * scale
			binW[b] += r
			binCnt[b]++
		}
	}
	var xs, ys []float64
	for b := 0; b < nbin; b++ {
		if binCnt[b] == 0 {
			continue
		}
		rw := binW[b] / float64(binCnt[b])
		pp := binPow[b] / float64(binCnt[b])
		if rw < wMin || rw > wMax || pp <= 0 {
			continue
		}
		xs = append(xs, math.Log(rw))
		ys = append(ys, math.Log(pp))
	}
	n := float64(len(xs))
	var sx, sy, sxx, sxy float64
	for i := range xs {
		sx += xs[i]
		sy += ys[i]
		sxx += xs[i] * xs[i]
		sxy += xs[i] * ys[i]
	}
	slope := (n*sxy - sx*sy) / (n*sxx - sx*sx)
	intercept := (sy - slope*sx) / n
	var ssRes, ssTot float64
	ybar := sy / n
	for i := range xs {
		res := ys[i] - (slope*xs[i] + intercept)
		ssRes += res * res
		ssTot += (ys[i] - ybar) * (ys[i] - ybar)
	}
	fit := SlopeFit{Beta: -slope, RSquared: 1 - ssRes/ssTot, Wmin: wMin, Wmax: wMax}
	for i := range xs {
		fit.W = append(fit.W, math.Exp(xs[i]))
		fit.LogP = append(fit.LogP, ys[i])
	}
	return fit
}

// angularCDFRaw 是零均值二元高斯梯度方位角密度的闭式原函数在 φ 处的主值。
// 协方差矩阵 Σ=[[mxx,mxy],[mxy,myy]]，det 为其行列式。
//
// 角度密度（由联合高斯 pdf 经极坐标积分得到，推导见 PROOF.md 第 2 节）：
//
//	f(φ)= √det / [2π·(myy·cos²φ − 2mxy·sinφcosφ + mxx·sin²φ)]
//
// 其原函数为
//
//	F(φ)= 1/(2π)·atan2(√det·sinφ, myy·cosφ − mxy·sinφ)
//
// atan2 主值落在 (−π,π]，跨支时扇区差分补 2π 即可（见 SectorProbabilities）。
func angularCDFRaw(phi, mxx, myy, mxy, sqrtDet float64) float64 {
	sp, cp := math.Sin(phi), math.Cos(phi)
	return math.Atan2(sqrtDet*sp, myy*cp-mxy*sp) / (2 * math.Pi)
}

// SectorProbabilities 按 nb 个等宽扇区给出高斯近似下的各箱概率（和为 1）。
// 法线方位角 = 梯度方位角 + π，整箱平移不改变占比集合，
// 因此直接返回以梯度角分箱的概率，调用方按同一约定比对即可。
func SectorProbabilities(nb int, mxx, myy, mxy float64) []float64 {
	det := mxx*myy - mxy*mxy
	if det <= 0 {
		out := make([]float64, nb) // 退化保护：按各向同性处理
		for k := range out {
			out[k] = 1 / float64(nb)
		}
		return out
	}
	sqrtDet := math.Sqrt(det)
	width := 2 * math.Pi / float64(nb)
	out := make([]float64, nb)
	prev := angularCDFRaw(0, mxx, myy, mxy, sqrtDet)
	var sum float64
	for k := 0; k < nb; k++ {
		hi := float64(k+1) * width
		cur := angularCDFRaw(hi, mxx, myy, mxy, sqrtDet)
		p := cur - prev
		if p < 0 { // atan2 跨过 ±π 支切
			p += 1
		}
		out[k] = p
		sum += p
		prev = cur
	}
	for k := range out { // 归一化消除舍入误差
		out[k] /= sum
	}
	return out
}

// SectorProbabilitiesNumeric 用细网格数值积分实现同一计算，供交叉校验。
func SectorProbabilitiesNumeric(nb int, mxx, myy, mxy float64) []float64 {
	const stepsPerSector = 2048
	width := 2 * math.Pi / float64(nb)
	sqrtDet := math.Sqrt(mxx*myy - mxy*mxy)
	norm := 1 / (2 * math.Pi)
	out := make([]float64, nb)
	var total float64
	for k := 0; k < nb; k++ {
		var p float64
		for s := 0; s < stepsPerSector; s++ {
			phi := (float64(k) + (float64(s)+0.5)/float64(stepsPerSector)) * width
			cp, sp := math.Cos(phi), math.Sin(phi)
			den := myy*cp*cp - 2*mxy*sp*cp + mxx*sp*sp
			p += norm * sqrtDet / den * (width / float64(stepsPerSector))
		}
		out[k] = p
		total += p
	}
	for k := range out {
		out[k] /= total
	}
	return out
}
