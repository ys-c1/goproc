package normalmap

import "math"

// GradientBin 是某个倾角区间内的误差统计，用来实证“误差随梯度变化”。
type GradientBin struct {
	TiltLoDeg     float64 `json:"tilt_lo_deg"`
	TiltHiDeg     float64 `json:"tilt_hi_deg"`
	Pixels        int     `json:"pixels"`
	MeanErrorDeg  float64 `json:"mean_error_deg"`
	MaxErrorDeg   float64 `json:"max_error_deg"`
	LocalBoundDeg float64 `json:"local_bound_deg_mid_gradient"`
	SafeBoundDeg  float64 `json:"safe_bound_deg_mid_gradient"`
}

// BinErrorsByGradient 按梯度（倾角）区间统计误差。bins 给出梯度模长的上下界。
func BinErrorsByGradient(nm *NormalMap, q *Quantized, mask []bool, bins [][2]float64) []GradientBin {
	out := make([]GradientBin, 0, len(bins))
	for _, bn := range bins {
		b := GradientBin{TiltLoDeg: SlopeTiltDeg(bn[0]), TiltHiDeg: SlopeTiltDeg(bn[1])}
		gmid := 0.5 * (bn[0] + bn[1])
		b.LocalBoundDeg = LocalErrorBoundDeg(q.Bits, gmid)
		b.SafeBoundDeg = BoundDeg(q.Bits, gmid)
		var sum float64
		for y := 0; y < nm.Height; y++ {
			for x := 0; x < nm.Width; x++ {
				i := nm.idx(x, y)
				if mask != nil && !mask[i] {
					continue
				}
				g := nm.Gradient[i]
				if g < bn[0] || g >= bn[1] {
					continue
				}
				e := AngleDeg(nm.Normals[i], q.Normals[i])
				b.Pixels++
				sum += e
				if e > b.MaxErrorDeg {
					b.MaxErrorDeg = e
				}
			}
		}
		if b.Pixels > 0 {
			b.MeanErrorDeg = sum / float64(b.Pixels)
		}
		out = append(out, b)
	}
	return out
}

// DefaultGradientBins 是报告使用的四档倾角区间（约 3–15/15–30/30–50/50–60°）。
func DefaultGradientBins() [][2]float64 {
	d := math.Pi / 180
	g := func(deg float64) float64 { return math.Tan(deg * d) }
	return [][2]float64{{g(3), g(15)}, {g(15), g(30)}, {g(30), g(50)}, {g(50), g(60.5)}}
}

// QuantizeXY 只量化 nx、ny 两个通道，着色器端由
// nz = sqrt(1-nx²-ny²) 重构造 z（BC5/XY 型法线贴图的标准做法）。
// 该模型下角度误差对梯度的依赖最直接：沿梯度方向
// δθ ≈ δnx/cos θ = δnx·√(1+g²)。
func QuantizeXY(nm *NormalMap, bits int) *Quantized {
	m := MaxCode(bits)
	q := &Quantized{
		Bits: bits, MaxCode: m,
		Codes:   make([][3]int, len(nm.Normals)),
		Decoded: make([]Vec3, len(nm.Normals)),
		Normals: make([]Vec3, len(nm.Normals)),
	}
	for i, n := range nm.Normals {
		enc := func(v float64) (int, float64) {
			c := int(math.Round((v*0.5 + 0.5) * float64(m)))
			if c < 0 {
				c = 0
			}
			if c > m {
				c = m
			}
			return c, 2*float64(c)/float64(m) - 1
		}
		cx, dx := enc(n.X)
		cy, dy := enc(n.Y)
		dz := math.Sqrt(math.Max(0, 1-dx*dx-dy*dy))
		v := Vec3{dx, dy, dz}.Normalize()
		q.Codes[i] = [3]int{cx, cy, -1}
		q.Decoded[i] = Vec3{dx, dy, dz}
		q.Normals[i] = v
	}
	return q
}
