// Package normalmap 把高度图转成切线空间法线，并对“显存位深量化法线”
// 的角度误差、误差界反解与斜坡台阶计数给出可复现的实现。
//
// 约定（与显卡 unorm 纹理一致）：
//
//	b 位通道的最大码值 M = 2^b - 1；
//	[-1,1] 的法线分量编码 code = round((v*0.5+0.5)*M)，
//	解码 v' = 2*code/M - 1，因此单分量最大解码误差 h = 1/M。
package normalmap

import "math"

// Vec3 是三维向量。
type Vec3 struct{ X, Y, Z float64 }

// Add 返回 v+o。
func (v Vec3) Add(o Vec3) Vec3 { return Vec3{v.X + o.X, v.Y + o.Y, v.Z + o.Z} }

// Scale 返回 sv。
func (v Vec3) Scale(s float64) Vec3 { return Vec3{v.X * s, v.Y * s, v.Z * s} }

// Dot 点积。
func Dot(a, b Vec3) float64 { return a.X*b.X + a.Y*b.Y + a.Z*b.Z }

// Norm 模长。
func (v Vec3) Norm() float64 { return math.Sqrt(Dot(v, v)) }

// Normalize 归一化；零向量返回朝上的单位法线，避免 NaN。
func (v Vec3) Normalize() Vec3 {
	n := v.Norm()
	if n == 0 {
		return Vec3{0, 0, 1}
	}
	return v.Scale(1 / n)
}

// NormalMap 是逐像素单位法线与梯度模长。
type NormalMap struct {
	Width    int
	Height   int
	Normals  []Vec3
	Gradient []float64 // g = hypot(dh/dx, dh/dy)，世界单位/纹素
}

func (m *NormalMap) idx(x, y int) int { return y*m.Width + x }

// At 取像素法线。
func (m *NormalMap) At(x, y int) Vec3 { return m.Normals[m.idx(x, y)] }

// GradAt 取像素梯度模长。
func (m *NormalMap) GradAt(x, y int) float64 { return m.Gradient[m.idx(x, y)] }

// FromHeight 由高度场构造法线贴图。
//
// height 长度必须为 width*height；heightScale 是高度值 1.0 对应的世界高度，
// 相邻纹素间距取 1。中心差分求梯度，边界用单侧差分：
//
//	gx = heightScale * dh/dx, gy = heightScale * dh/dy
//	n  = normalize(-gx, -gy, 1)
func FromHeight(width, height int, h []float64, heightScale float64) *NormalMap {
	m := &NormalMap{
		Width:    width,
		Height:   height,
		Normals:  make([]Vec3, width*height),
		Gradient: make([]float64, width*height),
	}
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			var dx float64
			switch {
			case width <= 1:
				dx = 0
			case x == 0:
				dx = h[m.idx(1, y)] - h[m.idx(0, y)]
			case x == width-1:
				dx = h[m.idx(width-1, y)] - h[m.idx(width-2, y)]
			default:
				dx = (h[m.idx(x+1, y)] - h[m.idx(x-1, y)]) * 0.5
			}
			var dy float64
			switch {
			case height <= 1:
				dy = 0
			case y == 0:
				dy = h[m.idx(x, 1)] - h[m.idx(x, 0)]
			case y == height-1:
				dy = h[m.idx(x, height-1)] - h[m.idx(x, height-2)]
			default:
				dy = (h[m.idx(x, y+1)] - h[m.idx(x, y-1)]) * 0.5
			}
			gx := heightScale * dx
			gy := heightScale * dy
			n := Vec3{-gx, -gy, 1}.Normalize()
			i := m.idx(x, y)
			m.Normals[i] = n
			m.Gradient[i] = math.Hypot(gx, gy)
		}
	}
	return m
}

// SlopeTiltDeg 返回法线相对竖直向上的倾角 θ = atan(g)，单位度。
func SlopeTiltDeg(gradient float64) float64 {
	return math.Atan(gradient) * 180 / math.Pi
}

// AngleDeg 两法线夹角（度）。
func AngleDeg(a, b Vec3) float64 {
	c := Dot(a, b)
	if c > 1 {
		c = 1
	}
	if c < -1 {
		c = -1
	}
	return math.Acos(c) * 180 / math.Pi
}
