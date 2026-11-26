// Package water 在高度图之上定义水面法线、光照与镜面反射着色。
//
// 反射系数（镜面项）采用对齐全反射方向取峰的模型：
//
//	F(v)   = F0 + (1-F0)*v^q          （对齐增益，v=cos gamma，越对齐越强）
//	I(p)   = K * F(v) * v^p
//
// 其中 gamma 是法线与半向量 H 的夹角，H 由光照方向 L 与视线方向 V 合成。
// gamma 为零时反射系数取峰 K，随夹角增大单调衰减。I 经八位灰度量化后
// 顶格（灰度 255）的像素即白像素，也就是高光像素。
package water

import (
	"errors"
	"math"
)

// ErrEmptyWhite 表示该光照角下没有任何白像素，无法据此标定阈值。
var ErrEmptyWhite = errors.New("该光照角下没有白像素，无法标定反射系数阈值")

// Shader 是水面镜面反射着色器的全部参数。
type Shader struct {
	K          float64 // 镜面强度
	P          float64 // 镜面幂次
	Q          float64 // 对齐增益幂次
	F0         float64 // 正入射反射率
	LightAz    float64 // 光照方位角（弧度），固定为 pi
	ViewElev   float64 // 视线仰角（弧度），固定为 pi/3
	ViewAz     float64 // 视线方位角（弧度），固定为零
	GridStep   float64 // 水平网格步长，与高度差同量纲
}

// DefaultShader 返回默认着色参数。幂次取高值使顶格只覆盖与半向量最对齐
// 的少数像素，高光呈局部斑块而不是整片齐亮。
func DefaultShader() Shader {
	return Shader{
		K:        1.05,
		P:        100,
		Q:        50,
		F0:       0.02,
		LightAz:  math.Pi,
		ViewElev: math.Pi / 3,
		ViewAz:   0,
		GridStep: 0.15,
	}
}

// Vec3 是最小三维向量工具，全部按单位向量语义使用。
type Vec3 struct{ X, Y, Z float64 }

func (a Vec3) Add(b Vec3) Vec3      { return Vec3{a.X + b.X, a.Y + b.Y, a.Z + b.Z} }
func (a Vec3) Dot(b Vec3) float64   { return a.X*b.X + a.Y*b.Y + a.Z*b.Z }
func (a Vec3) Scale(s float64) Vec3 { return Vec3{a.X * s, a.Y * s, a.Z * s} }

func (a Vec3) Norm() Vec3 {
	l := math.Sqrt(a.Dot(a))
	if l == 0 {
		return Vec3{}
	}
	return a.Scale(1 / l)
}

// NormalField 是逐像素单位法线，采用中心差分、边界复制。
type NormalField struct {
	Size int
	N    []Vec3
}

// At 取某点法线。
func (f *NormalField) At(x, y int) Vec3 { return f.N[y*f.Size+x] }

// NewNormalField 由归一化高度图（边长 size）计算单位法线场。
func NewNormalField(size int, height []float64, gridStep float64) *NormalField {
	n := make([]Vec3, size*size)
	at := func(x, y int) float64 {
		if x < 0 {
			x = 0
		}
		if x >= size {
			x = size - 1
		}
		if y < 0 {
			y = 0
		}
		if y >= size {
			y = size - 1
		}
		return height[y*size+x]
	}
	inv := 1 / (2 * gridStep)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			dx := (at(x+1, y) - at(x-1, y)) * inv
			dy := (at(x, y+1) - at(x, y-1)) * inv
			// 高度沿 +z，法线与梯度方向相反。
			n[y*size+x] = Vec3{-dx, -dy, 1}.Norm()
		}
	}
	return &NormalField{Size: size, N: n}
}

// LightDir 由光照仰角（弧度，零为水平）与固定方位角给出单位光照方向。
func LightDir(elev, az float64) Vec3 {
	c := math.Cos(elev)
	return Vec3{c * math.Cos(az), c * math.Sin(az), math.Sin(elev)}
}

// ViewDir 返回固定俯视相机的单位视线方向（从像素指向相机）。
func (s Shader) ViewDir() Vec3 {
	c := math.Cos(s.ViewElev)
	return Vec3{c * math.Cos(s.ViewAz), c * math.Sin(s.ViewAz), math.Sin(s.ViewElev)}
}

// HalfDir 返回给定光照仰角下光照方向与视线方向的单位半向量，纯几何量。
func (s Shader) HalfDir(lightElev float64) Vec3 {
	return HalfVector(LightDir(lightElev, s.LightAz), s.ViewDir())
}

// HalfVector 返回两个单位向量的单位半向量。
func HalfVector(l, v Vec3) Vec3 { return l.Add(v).Norm() }

// Fresnel 是对齐增益项 F(v)，v 即法线与半向量的夹角余弦，越对齐（v 越大）越强。
func (s Shader) Fresnel(cosGamma float64) float64 {
	return s.F0 + (1-s.F0)*math.Pow(cosGamma, s.Q)
}

// ReflectanceAt 返回法线与半向量夹角余弦为 cosGamma 时的反射系数，
// 供阈值曲线反解临界夹角使用，与 Specular 同一公式。
func (s Shader) ReflectanceAt(cosGamma float64) float64 {
	if cosGamma < 0 {
		return 0
	}
	return s.K * s.Fresnel(cosGamma) * math.Pow(cosGamma, s.P)
}

// Specular 返回单个像素的镜面反射系数（反射系数）。
func (s Shader) Specular(nn, h Vec3) float64 {
	return s.ReflectanceAt(nn.Dot(h))
}

// WhiteLevel 是白像素的最小强度：灰度 255 对应 I >= 254.5/255。
const WhiteLevel = 254.5 / 255.0

// IsWhite 判断强度量化为八位灰度后是否顶格。
func IsWhite(intensity float64) bool { return intensity >= WhiteLevel }

// Gray 把强度量化为零到二五五的灰度值。
func Gray(intensity float64) int {
	g := int(intensity*255 + 0.5)
	if g < 0 {
		return 0
	}
	if g > 255 {
		return 255
	}
	return g
}

// Frame 是一个光照角下的逐像素着色结果。
type Frame struct {
	Size       int
	Intensity  []float64 // 镜面强度 I
	Specular   []float64 // 反射系数（与强度同一模型，留作逐像素留档）
	White      []bool    // 白像素（灰度顶格）掩码
	WhiteCount int
}

// Render 在给定光照仰角（弧度）下对法线场做完整着色并统计白像素。
func (s Shader) Render(f *NormalField, lightElev float64) *Frame {
	size := f.Size
	h := s.HalfDir(lightElev)
	frame := &Frame{
		Size:      size,
		Intensity: make([]float64, size*size),
		Specular:  make([]float64, size*size),
		White:     make([]bool, size*size),
	}
	for i, nn := range f.N {
		rho := s.Specular(nn, h)
		frame.Specular[i] = rho
		frame.Intensity[i] = rho
		if IsWhite(rho) {
			frame.White[i] = true
			frame.WhiteCount++
		}
	}
	return frame
}

// CalibrationPoint 是一个标定点的离线测量结果。
type CalibrationPoint struct {
	LightElevDeg float64 // 光照仰角（度）
	Threshold     float64 // 反射系数阈值：白像素中的最小反射系数
	WhiteCount    int     // 该角度白像素数
}

// MeasureThreshold 在一个已知光照角下渲染并测出反射系数阈值。
// 没有白像素时返回 ErrEmptyWhite。
func (s Shader) MeasureThreshold(f *NormalField, lightElev float64) (CalibrationPoint, error) {
	frame := s.Render(f, lightElev)
	if frame.WhiteCount == 0 {
		return CalibrationPoint{LightElevDeg: lightElev * 180 / math.Pi}, ErrEmptyWhite
	}
	min := math.Inf(1)
	for i, w := range frame.White {
		if w && frame.Specular[i] < min {
			min = frame.Specular[i]
		}
	}
	return CalibrationPoint{
		LightElevDeg: lightElev * 180 / math.Pi,
		Threshold:     min,
		WhiteCount:    frame.WhiteCount,
	}, nil
}
