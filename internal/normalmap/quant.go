package normalmap

import "math"

// 位深与量化误差界。
//
// 设位深为 b，M = 2^b-1 是 unorm 码值上限，单分量最大解码误差 h = 1/M。
//
// 梯度 g = |∇h| 与法线倾角 θ 满足 g = tan θ，水平分量 u = sin θ = g/√(1+g²)。
// 倾角对水平分量的敏感度为
//
//	dθ/du = 1/cos θ = √(1+g²)。
//
// 所以一个码级 Δu = 2/M 对应的法线角度量化步长（一阶）为
//
//	Δθ(b,g) ≈ 2√(1+g²)/M（弧度），
//
// 半级舍入误差约为 √(1+g²)/M。
//
// 严格界：三个分量的解码误差向量满足 |e| ≤ √3/M。量化后法线 q=n+e
// 再归一化，几何上 q 落在以 n 为球心、√3/M 为半径的球内；原点对该球的
// 最大张角为 arcsin(√3/M)，故
//
//	ε ≤ arcsin(√3/M)。
//
// 当 b≥3 时 M≥7，数值上有 arcsin(√3/M) ≤ 2/M ≤ 2√(1+g²)/M，于是得到
// 带梯度的严格保守界
//
//	ε ≤ BoundRad(b,g) = 2√(1+g²)/M。
const (
	// Deg2Rad 度转弧度乘数。
	Deg2Rad = math.Pi / 180
	// Rad2Deg 弧度转度乘数。
	Rad2Deg = 180 / math.Pi
)

// MaxCode 返回 b 位 unorm 的最大码值 M = 2^b-1。
func MaxCode(bits int) int {
	if bits <= 0 {
		return 0
	}
	return 1<<bits - 1
}

// ComponentHalfStep 是单分量最大解码误差 h = 1/M。
func ComponentHalfStep(bits int) float64 {
	return 1 / float64(MaxCode(bits))
}

// AngularStepRad 是一阶法线角度量化步长 Δθ ≈ 2√(1+g²)/M（弧度）。
func AngularStepRad(bits int, gradient float64) float64 {
	m := float64(MaxCode(bits))
	return 2 * math.Sqrt(1+gradient*gradient) / m
}

// AngularStepDeg 同 AngularStepRad，单位度。
func AngularStepDeg(bits int, gradient float64) float64 {
	return AngularStepRad(bits, gradient) * Rad2Deg
}

// LocalErrorBoundRad 是一阶半级误差估计 √(1+g²)/M（弧度）。
// 它刻画误差随位深与梯度变化的比例，但不是全 RGB 三通道下的严格上界；
// 验收严格界请用 BoundRad。
func LocalErrorBoundRad(bits int, gradient float64) float64 {
	m := float64(MaxCode(bits))
	return math.Sqrt(1+gradient*gradient) / m
}

// LocalErrorBoundDeg 同 LocalErrorBoundRad，单位度。
func LocalErrorBoundDeg(bits int, gradient float64) float64 {
	return LocalErrorBoundRad(bits, gradient) * Rad2Deg
}

// RigorousBoundRad 是不含梯度的严格球界 arcsin(√3/M)（弧度）。
func RigorousBoundRad(bits int) float64 {
	r := math.Sqrt(3) * ComponentHalfStep(bits)
	if r >= 1 {
		return math.Pi / 2
	}
	return math.Asin(r)
}

// RigorousBoundDeg 同 RigorousBoundRad，单位度。
func RigorousBoundDeg(bits int) float64 { return RigorousBoundRad(bits) * Rad2Deg }

// BoundRad 是带梯度的严格保守误差界 2√(1+g²)/M（弧度）。
// 仅对 bits≥3 给出严格保证（见包注释的不等式）。
func BoundRad(bits int, gradient float64) float64 {
	m := float64(MaxCode(bits))
	return 2 * math.Sqrt(1+gradient*gradient) / m
}

// BoundDeg 同 BoundRad，单位度。
func BoundDeg(bits int, gradient float64) float64 {
	return BoundRad(bits, gradient) * Rad2Deg
}

// MinBits 按梯度误差界反解满足 angleErrorDeg 所需的最小位深：
//
//	2√(1+g²)/(2^b-1) ≤ ε_rad
//	⇒ b_min = ceil(log2(1 + 2√(1+g²)/ε_rad))
//
// 与 BoundRad 配套，要求返回值 b≥3 时界是严格的。
func MinBits(gradient, angleErrorDeg float64) int {
	eps := angleErrorDeg * Deg2Rad
	if eps <= 0 {
		return 0
	}
	b := int(math.Ceil(math.Log2(1 + 2*math.Sqrt(1+gradient*gradient)/eps)))
	if b < 1 {
		b = 1
	}
	return b
}

// MinBitsRigorous 按无梯度严格球界反解：
//
//	arcsin(√3/(2^b-1)) ≤ ε
//	⇒ b_min = ceil(log2(1 + √3/sin ε))。
func MinBitsRigorous(angleErrorDeg float64) int {
	eps := angleErrorDeg * Deg2Rad
	if eps <= 0 {
		return 0
	}
	s := math.Sin(eps)
	if s <= 0 {
		return 0
	}
	b := int(math.Ceil(math.Log2(1 + math.Sqrt(3)/s)))
	if b < 1 {
		b = 1
	}
	return b
}

// Quantized 是一次 b 位量化的结果。
type Quantized struct {
	Bits    int
	MaxCode int
	Codes   [][3]int // 每通道整数码
	Decoded []Vec3   // 直接解码（未重归一化）
	Normals []Vec3   // 解码后重归一化的法线，供着色/测误差
}

// Quantize 对法线逐通道做 b 位 unorm 量化与解码。
func Quantize(nm *NormalMap, bits int) *Quantized {
	m := MaxCode(bits)
	q := &Quantized{
		Bits:    bits,
		MaxCode: m,
		Codes:   make([][3]int, len(nm.Normals)),
		Decoded: make([]Vec3, len(nm.Normals)),
		Normals: make([]Vec3, len(nm.Normals)),
	}
	for i, n := range nm.Normals {
		var code [3]int
		var dec Vec3
		vals := [3]float64{n.X, n.Y, n.Z}
		for k, v := range vals {
			c := int(math.Round((v*0.5 + 0.5) * float64(m)))
			if c < 0 {
				c = 0
			}
			if c > m {
				c = m
			}
			code[k] = c
			d := 2*float64(c)/float64(m) - 1
			switch k {
			case 0:
				dec.X = d
			case 1:
				dec.Y = d
			default:
				dec.Z = d
			}
		}
		q.Codes[i] = code
		q.Decoded[i] = dec
		q.Normals[i] = dec.Normalize()
	}
	return q
}

// ErrorStats 是一组像素上的法线夹角误差（度）统计。
type ErrorStats struct {
	Bits          int     `json:"bits"`
	Pixels        int     `json:"pixels"`
	MaxDeg        float64 `json:"max_angle_error_deg"`
	MeanDeg       float64 `json:"mean_angle_error_deg"`
	RMSDeg        float64 `json:"rms_angle_error_deg"`
	AtX           int     `json:"at_x"`
	AtY           int     `json:"at_y"`
	GradientAtMax float64 `json:"gradient_at_max"`
	BoundDeg      float64 `json:"bound_deg_at_max_gradient"`
	BoundMaxDeg   float64 `json:"max_gradient_bound_deg"`
	Violations    int     `json:"bound_violations"`
}

// MeasureErrors 在 mask（nil 表示全图）上比较原始法线与量化重归一化法线，
// 并逐像素核对 BoundDeg 严格保守界。
func MeasureErrors(nm *NormalMap, q *Quantized, mask []bool) ErrorStats {
	st := ErrorStats{Bits: q.Bits, AtX: -1, AtY: -1}
	var sum, sum2 float64
	for y := 0; y < nm.Height; y++ {
		for x := 0; x < nm.Width; x++ {
			i := nm.idx(x, y)
			if mask != nil && !mask[i] {
				continue
			}
			e := AngleDeg(nm.Normals[i], q.Normals[i])
			g := nm.Gradient[i]
			st.Pixels++
			sum += e
			sum2 += e * e
			if e > st.MaxDeg || st.AtX < 0 {
				st.MaxDeg = e
				st.AtX, st.AtY = x, y
				st.GradientAtMax = g
			}
			b := BoundDeg(q.Bits, g)
			if b > st.BoundMaxDeg {
				st.BoundMaxDeg = b
			}
			if e > b+1e-9 {
				st.Violations++
			}
		}
	}
	if st.Pixels > 0 {
		st.MeanDeg = sum / float64(st.Pixels)
		st.RMSDeg = math.Sqrt(sum2 / float64(st.Pixels))
	}
	st.BoundDeg = BoundDeg(q.Bits, st.GradientAtMax)
	return st
}
