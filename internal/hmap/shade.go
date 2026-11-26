// Package hmap 负责归一化、导出与地形光照。
package hmap

import "math"

// Gradient 用前向差分计算高度图梯度。内部像素 (x<size-1, y<size-1)：
//
//	Dx(x,y) = (h(x+1,y) - h(x,y)) / dx
//	Dy(x,y) = (h(x,y+1) - h(x,y)) / dx
//
// 最右一列与最下一行没有前向邻点，退回后向差分，保证每个像素都有定义。
// 返回两个等长切片，长度均为 size*size。
func Gradient(size int, h []float64, dx float64) (gx, gy []float64) {
	gx = make([]float64, size*size)
	gy = make([]float64, size*size)
	if dx <= 0 {
		return gx, gy
	}
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			i := y*size + x
			if x < size-1 {
				gx[i] = (h[i+1] - h[i]) / dx
			} else {
				gx[i] = (h[i] - h[i-1]) / dx
			}
			if y < size-1 {
				gy[i] = (h[i+size] - h[i]) / dx
			} else {
				gy[i] = (h[i] - h[i-size]) / dx
			}
		}
	}
	return gx, gy
}

// Light 是指向光源的单位向量。坐标系为图像坐标：X 向右、Y 向下、Z 向上。
// Azimuth 从 +X（右）起逆时针为正；Altitude 为仰角，取正值时光在地形上方。
type Light struct {
	Azimuth  float64 // 弧度，0=右，π/2=上
	Altitude float64 // 弧度，π/2=正上方
}

// Vector 返回光源单位向量 (lx, ly, lz)。
func (l Light) Vector() (lx, ly, lz float64) {
	cosAlt := math.Cos(l.Altitude)
	lx = cosAlt * math.Cos(l.Azimuth)
	ly = -cosAlt * math.Sin(l.Azimuth) // 图像 Y 轴向下
	lz = math.Sin(l.Altitude)
	return lx, ly, lz
}

// ShadeLinear 是点积光照模型：L = l·n，其中
//
//	n = (-gx, -gy, 1)/sqrt(1+gx²+gy²)
//
// 是归一化表面法向量。结果即“梯度与光向的点积”，未截断，可正可负。
// gx、gy 由 Gradient 以同一格距 dx 计算时，dx 同时缩放两个分量，
// 点积对 dx 不变（等价性证明见 REPORT.md）。
func ShadeLinear(gx, gy []float64, l Light) []float64 {
	lx, ly, lz := l.Vector()
	out := make([]float64, len(gx))
	for i := range gx {
		denom := math.Sqrt(1 + gx[i]*gx[i] + gy[i]*gy[i])
		nx, ny, nz := -gx[i]/denom, -gy[i]/denom, 1/denom
		out[i] = lx*nx + ly*ny + lz*nz
	}
	return out
}

// Shade 是完整朗伯模型：先截断负点积（背光面全黑），单位光向下结果落在零到一。
// 不再做二次拉伸，以保留与线性点积模型可直接比较的绝对亮度；
// zeroFrac 返回被截断（点积为负）像素的比例。
func Shade(gx, gy []float64, l Light) (b []float64, zeroFrac float64) {
	raw := ShadeLinear(gx, gy, l)
	b = make([]float64, len(raw))
	neg := 0
	for i, v := range raw {
		if v < 0 {
			neg++
			continue
		}
		b[i] = v
	}
	return b, float64(neg) / float64(len(raw))
}

// Moments 是亮度分布的样本矩统计。Skew 为总体（有偏）偏度 E[(X-μ)³]/σ³。
type Moments struct {
	N       int     `json:"n"`
	Mean    float64 `json:"mean"`
	Std     float64 `json:"std"`
	Min     float64 `json:"min"`
	Max     float64 `json:"max"`
	Skew    float64 `json:"skew"`
	NegFrac float64 `json:"neg_frac"` // 线性点积模型中负值像素占比
}

// Stats 计算亮度分布矩。v 为线性点积亮度（未截断）。
func Stats(v []float64) Moments {
	if len(v) == 0 {
		return Moments{}
	}
	var sum float64
	lo, hi := v[0], v[0]
	for _, x := range v {
		sum += x
		if x < lo {
			lo = x
		}
		if x > hi {
			hi = x
		}
	}
	mean := sum / float64(len(v))
	var m2, m3, neg float64
	for _, x := range v {
		d := x - mean
		d2 := d * d
		m2 += d2
		m3 += d2 * d
		if x < 0 {
			neg++
		}
	}
	n := float64(len(v))
	std := math.Sqrt(m2 / n)
	skew := 0.0
	if std > 0 {
		skew = (m3 / n) / (std * std * std)
	}
	return Moments{N: len(v), Mean: mean, Std: std, Min: lo, Max: hi, Skew: skew, NegFrac: neg / n}
}
