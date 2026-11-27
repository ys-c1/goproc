// Package orient 用结构张量估计纹理主方向，并给出接缝处的旋转对齐。
//
// 约定：图像坐标 x 向右、y 向下，角度单位为度，主方向角按结构张量最大特征
// 值对应主轴的反正切一半计算，方向场是无向的，角度按模 180° 回卷到
// (-90, 90]。
package orient

import "math"

// Tensor 是窗口内平均后的结构张量 J = E[∇u ∇uᵀ] 的三个独立分量。
type Tensor struct {
	Jxx, Jyy, Jxy float64
}

// Diff 选择梯度的有限差分格式。
type Diff int

const (
	Central  Diff = iota // 中心差分（默认，二阶截断误差）
	Forward              // 前向差分（一阶）
	Backward             // 后向差分（一阶）
)

// wrap180 把任意角度回卷到 (-90, 90]，用于无向方向场的模 180° 比较。
func wrap180(deg float64) float64 {
	deg = math.Mod(deg, 180.0)
	if deg > 90.0 {
		deg -= 180.0
	}
	if deg <= -90.0 {
		deg += 180.0
	}
	return deg
}

// Wrap180 导出角度回卷，供实验与外部调用复用。
func Wrap180(deg float64) float64 { return wrap180(deg) }

// StructureTensor 在高度图的矩形窗口 [x0,x1)×[y0,y1) 内平均结构张量。
// 只采用窗口内部、两侧邻居都落在图内的像素，因此边界不引入单边差分污染。
// 返回张量与参与平均的有效像素数；空窗口返回零张量与 0。
func StructureTensor(size int, data []float64, x0, y0, x1, y1 int, diff Diff) (Tensor, int) {
	if x0 < 0 {
		x0 = 0
	}
	if y0 < 0 {
		y0 = 0
	}
	if x1 > size {
		x1 = size
	}
	if y1 > size {
		y1 = size
	}
	at := func(x, y int) float64 { return data[y*size+x] }

	var sx2, sy2, sxy, n float64
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			var gx, gy float64
			switch diff {
			case Forward:
				if x+1 >= size || y+1 >= size {
					continue
				}
				gx = at(x+1, y) - at(x, y)
				gy = at(x, y+1) - at(x, y)
			case Backward:
				if x-1 < 0 || y-1 < 0 {
					continue
				}
				gx = at(x, y) - at(x-1, y)
				gy = at(x, y) - at(x, y-1)
			default: // Central
				if x-1 < 0 || x+1 >= size || y-1 < 0 || y+1 >= size {
					continue
				}
				gx = 0.5 * (at(x+1, y) - at(x-1, y))
				gy = 0.5 * (at(x, y+1) - at(x, y-1))
			}
			sx2 += gx * gx
			sy2 += gy * gy
			sxy += gx * gy
			n++
		}
	}
	if n == 0 {
		return Tensor{}, 0
	}
	return Tensor{Jxx: sx2 / n, Jyy: sy2 / n, Jxy: sxy / n}, int(n)
}

// PrincipalAngle 返回结构张量主轴（最大特征值对应特征向量）的方向角，
// 单位度，范围 (-90, 90]。
//
// 特征值 λ± = (Jxx+Jyy ± r)/2，r = sqrt((Jxx-Jyy)²+4Jxy²)，
// 最大特征值对应特征向量的方向满足 tan(2θ) = 2Jxy/(Jxx-Jyy)，故
//
//	θ = ½ atan2(2 Jxy, Jxx - Jyy).
func PrincipalAngle(t Tensor) float64 {
	return wrap180(math.Atan2(2*t.Jxy, t.Jxx-t.Jyy) * 180.0 / math.Pi / 2.0)
}

// Seam 描述两块地形的拼接方向。
type Seam int

const (
	Horizontal Seam = iota // 左右拼接，接缝为竖直线
	Vertical               // 上下拼接，接缝为水平线
)

// SeamStrips 返回 A、B 两块各自贴接缝的条带窗口，width 为条带沿接缝法向
// 的像素宽度，margin 为沿接缝方向两端各内缩的像素数（用于避开旋转重采样
// 的边界污染区，传 0 表示不内缩）。水平拼接取 A 的右条带与 B 的左条带；
// 竖直拼接取 A 的下条带与 B 的上条带。
func SeamStrips(size, width, margin int, seam Seam) (ax0, ay0, ax1, ay1, bx0, by0, bx1, by1 int) {
	switch seam {
	case Horizontal:
		return size - width, margin, size, size - margin,
			0, margin, width, size - margin
	default:
		return margin, size - width, size - margin, size,
			margin, 0, size - margin, width
	}
}

// WindowArea 返回窗口的像素总数。
func WindowArea(x0, y0, x1, y1 int) int {
	if x1 <= x0 || y1 <= y0 {
		return 0
	}
	return (x1 - x0) * (y1 - y0)
}

// RotateTensor 返回纹理场主动旋转 phi 度（梯度向量 g' = R(phi) g）后的
// 结构张量，按 J' = R(phi) J R(phi)ᵀ 变换，其主轴角为原角度加 phi（模 180°）。
func RotateTensor(t Tensor, phiDeg float64) Tensor {
	phi := phiDeg * math.Pi / 180.0
	c, s := math.Cos(phi), math.Sin(phi)
	c2, s2, cs := c*c, s*s, c*s
	return Tensor{
		Jxx: c2*t.Jxx + s2*t.Jyy - 2*cs*t.Jxy,
		Jyy: s2*t.Jxx + c2*t.Jyy + 2*cs*t.Jxy,
		Jxy: cs*(t.Jxx-t.Jyy) + (c2-s2)*t.Jxy,
	}
}

// RotationAngle 返回把 B 的主方向旋转到 A 所需的角度（度，(-90,90]）。
func RotationAngle(thetaA, thetaB float64) float64 {
	return wrap180(thetaA - thetaB)
}

// AngularDifference 返回两个无向方向角的模 180° 差，范围 (-90, 90]。
func AngularDifference(thetaA, thetaB float64) float64 {
	return wrap180(thetaA - thetaB)
}

// RotateField 以图中心为原点把整幅高度图正向旋转 phi 度，双线性采样，
// 越界采用边界值延展。正向旋转的定义与 RotationAngle 配套：对旋转后的
// 图重新估计主方向，主方向角增加 phi（模 180°）。
func RotateField(size int, data []float64, phiDeg float64) []float64 {
	phi := phiDeg * math.Pi / 180.0
	cos, sin := math.Cos(phi), math.Sin(phi)
	c := float64(size-1) / 2.0
	out := make([]float64, len(data))

	sample := func(x, y float64) float64 {
		// 双线性采样，坐标夹取到合法范围，等价于边界延展。
		if x < 0 {
			x = 0
		}
		if y < 0 {
			y = 0
		}
		if x > float64(size-1) {
			x = float64(size - 1)
		}
		if y > float64(size-1) {
			y = float64(size - 1)
		}
		x0 := int(math.Floor(x))
		y0 := int(math.Floor(y))
		x1 := x0 + 1
		y1 := y0 + 1
		if x1 > size-1 {
			x1 = size - 1
		}
		if y1 > size-1 {
			y1 = size - 1
		}
		fx := x - float64(x0)
		fy := y - float64(y0)
		v00 := data[y0*size+x0]
		v10 := data[y0*size+x1]
		v01 := data[y1*size+x0]
		v11 := data[y1*size+x1]
		return (1-fx)*(1-fy)*v00 + fx*(1-fy)*v10 + (1-fx)*fy*v01 + fx*fy*v11
	}

	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			dx := float64(x) - c
			dy := float64(y) - c
			// 输出像素 p 处的值取自原图 R(phi)^-1 p。
			sx := c + cos*dx + sin*dy
			sy := c - sin*dx + cos*dy
			out[y*size+x] = sample(sx, sy)
		}
	}
	return out
}
