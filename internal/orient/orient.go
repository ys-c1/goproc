// Package orient 用结构张量（structure tensor）提取地形块的主方向角，
// 并由两块主方向角之差反解旋转角。
//
// 角度约定：图像坐标，x 向右、y 向下；0° 指向正东，角度顺时针为正；
// 主方向是无向轴，结果统一卷绕到 (-90°,90°]。
package orient

import "math"

// Tensor 是二阶结构张量 S=[[Sxx,Sxy],[Sxy,Syy]]。
type Tensor struct {
	Sxx, Sxy, Syy float64
}

// GradientTensor 在一个矩形块（闭区间 [x0,x1]×[y0,y1]）内累计中心差分梯度。
// 边界行/列的一阶导数用前向/后向差分补齐，避免漏掉块边。
func GradientTensor(h []float64, size, x0, y0, x1, y1 int) Tensor {
	var t Tensor
	for y := y0; y <= y1; y++ {
		for x := x0; x <= x1; x++ {
			gx := deriv(h, size, x, y, true, x0, x1)
			gy := deriv(h, size, x, y, false, y0, y1)
			t.Sxx += gx * gx
			t.Sxy += gx * gy
			t.Syy += gy * gy
		}
	}
	return t
}

func deriv(h []float64, size, x, y int, horizontal bool, lo, hi int) float64 {
	if horizontal {
		switch {
		case x == lo && x == hi:
			return 0
		case x == lo:
			return h[y*size+x+1] - h[y*size+x]
		case x == hi:
			return h[y*size+x] - h[y*size+x-1]
		default:
			return (h[y*size+x+1] - h[y*size+x-1]) * 0.5
		}
	}
	switch {
	case y == lo && y == hi:
		return 0
	case y == lo:
		return h[(y+1)*size+x] - h[y*size+x]
	case y == hi:
		return h[y*size+x] - h[(y-1)*size+x]
	default:
		return (h[(y+1)*size+x] - h[(y-1)*size+x]) * 0.5
	}
}

// GradientAngle 返回最大特征值特征向量（梯度主方向，即等高线法向）的角度，单位度，范围 (-90,90]。
func (t Tensor) GradientAngle() float64 {
	// 二倍角公式：2θ=atan2(2Sxy, Sxx-Syy)
	theta := 0.5 * math.Atan2(2*t.Sxy, t.Sxx-t.Syy) * 180 / math.Pi
	return wrap90(theta)
}

// RidgeAngle 返回最小特征值特征向量（山脊/沟谷走向，与梯度方向正交）的角度。
func (t Tensor) RidgeAngle() float64 { return wrap90(t.GradientAngle() + 90) }

// SplitVertical 把 size×size 方阵切成左右两个等宽块，返回各自像素范围。
func SplitVertical(size int) (left [4]int, right [4]int) {
	mid := size / 2
	left = [4]int{0, 0, mid - 1, size - 1}
	right = [4]int{mid, 0, size - 1, size - 1}
	return left, right
}

// Rotation 由右块主方向角减左块主方向角反解旋转角，并卷绕到 (-90,90]。
// 取法：θ = wrap90(θR - θL)，即把左块主方向旋到与右块主方向重合所需的有向转角。
func Rotation(thetaL, thetaR float64) float64 { return wrap90(thetaR - thetaL) }

func wrap90(a float64) float64 {
	for a <= -90 {
		a += 180
	}
	for a > 90 {
		a -= 180
	}
	return a
}
