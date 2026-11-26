package api

import (
	"goproc/internal/normalmap"
)

// NormalInfo 是一张法线贴图的对外数据。
type NormalInfo struct {
	Size     int          // 方形边长（Heightmap 为正方形）
	Normals  [][3]float64 `json:"-"` // 单位法线 (nx,ny,nz)
	Gradient []float64    `json:"-"` // 梯度模长 g=hypot(dh/dx,dh/dy)
}

// HeightmapNormals 由已生成的高度图构造切线空间法线。
// heightScale 是归一化高度 1.0 对应的世界高度；相邻像素间距取 1。
func HeightmapNormals(h *Heightmap, heightScale float64) *NormalInfo {
	m := normalmap.FromHeight(h.Size, h.Size, h.Data, heightScale)
	info := &NormalInfo{Size: h.Size, Normals: make([][3]float64, len(m.Normals)), Gradient: m.Gradient}
	for i, n := range m.Normals {
		info.Normals[i] = [3]float64{n.X, n.Y, n.Z}
	}
	return info
}

// QuantizedNormals 是 b 位 unorm 量化结果。
type QuantizedNormals struct {
	Bits    int
	MaxCode int
	Codes   [][3]int
	Normals [][3]float64 // 解码并重归一化后的法线
}

// QuantizeNormals 对法线做 b 位（6/8/10…）unorm 量化。
func QuantizeNormals(n *NormalInfo, bits int) *QuantizedNormals {
	nm := &normalmap.NormalMap{Width: n.Size, Height: n.Size, Gradient: n.Gradient}
	nm.Normals = make([]normalmap.Vec3, len(n.Normals))
	for i, v := range n.Normals {
		nm.Normals[i] = normalmap.Vec3{X: v[0], Y: v[1], Z: v[2]}
	}
	q := normalmap.Quantize(nm, bits)
	out := &QuantizedNormals{Bits: bits, MaxCode: q.MaxCode, Codes: q.Codes, Normals: make([][3]float64, len(q.Normals))}
	for i, v := range q.Normals {
		out.Normals[i] = [3]float64{v.X, v.Y, v.Z}
	}
	return out
}

// NormalAngleStepDeg 返回位深 b、梯度 g 下的一阶法线角度量化步长（度）：
// Δθ ≈ 2√(1+g²)/(2^b−1)。
func NormalAngleStepDeg(bits int, gradient float64) float64 {
	return normalmap.AngularStepDeg(bits, gradient)
}

// NormalAngleBoundDeg 返回带梯度的严格保守角度误差界（度）：
// ε ≤ 2√(1+g²)/(2^b−1)，对 bits≥3 严格成立。
func NormalAngleBoundDeg(bits int, gradient float64) float64 {
	return normalmap.BoundDeg(bits, gradient)
}

// NormalAngleRigorousBoundDeg 返回不含梯度的严格球界 arcsin(√3/(2^b−1))（度）。
func NormalAngleRigorousBoundDeg(bits int) float64 {
	return normalmap.RigorousBoundDeg(bits)
}

// MinNormalBits 按梯度误差界反解满足给定角度误差（度）的最小位深：
// b_min = ceil(log2(1 + 2√(1+g²)/ε))。
func MinNormalBits(gradient, maxAngleErrorDeg float64) int {
	return normalmap.MinBits(gradient, maxAngleErrorDeg)
}

// MinNormalBitsRigorous 按严格球界反解最小位深：
// b_min = ceil(log2(1 + √3/sin ε))。
func MinNormalBitsRigorous(maxAngleErrorDeg float64) int {
	return normalmap.MinBitsRigorous(maxAngleErrorDeg)
}

// DepthGroup 是一个位深下“误差 + 台阶数”的同组结果。
type DepthGroup = normalmap.StepReport

// SlopeStudy 是三档位深在标准斜坡靶上的整组实验结果。
type SlopeStudy struct {
	Width       int                    `json:"width"`
	Height      int                    `json:"height"`
	Theta0Deg   float64                `json:"theta0_deg"`
	Theta1Deg   float64                `json:"theta1_deg"`
	LightDeg    float64                `json:"light_tilt_deg"`
	GradientMax float64                `json:"gradient_max"`
	Groups      []normalmap.StepReport `json:"groups"`
}

// RunSlopeStudy 在同一张带平台的斜坡靶上依次评估给定位深（通常 6/8/10）。
func RunSlopeStudy(bitsList []int) *SlopeStudy {
	cfg := normalmap.DefaultSlopeConfig()
	target := normalmap.BuildSlopeTarget(cfg)
	nm := target.NormalMap()
	gmax := 0.0
	for i, ok := range target.Mask {
		if ok && nm.Gradient[i] > gmax {
			gmax = nm.Gradient[i]
		}
	}
	s := &SlopeStudy{
		Width: target.Width, Height: target.Height,
		Theta0Deg: cfg.Theta0Deg, Theta1Deg: cfg.Theta1Deg, LightDeg: cfg.LightDeg,
		GradientMax: gmax,
	}
	for _, b := range bitsList {
		s.Groups = append(s.Groups, normalmap.EvaluateBits(target, b))
	}
	return s
}
