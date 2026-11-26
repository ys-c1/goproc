// Package shade 在高度图上做双方向光源的 Lambert 着色。
//
// 坐标系：x 向右、y 向下。方位角 phi 从 +y 轴顺时针为正，
// 光源水平投影为 (sin phi, cos phi)；仰角 elev 为光源高出水平面的角度。
package shade

import "math"

// Light 是一个方向光源，两灯强度相等。
type Light struct {
	Azimuth float64 // 方位角，弧度
	Elev    float64 // 仰角，弧度
}

func vec(l Light) (x, y, z float64) {
	ce := math.Cos(l.Elev)
	return ce * math.Sin(l.Azimuth), ce * math.Cos(l.Azimuth), math.Sin(l.Elev)
}

func clamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// Render 对 size×size 归一化高度图 h 做双光源着色，返回零到一灰度。
// 梯度用中心差分（边界单侧差分），每盏灯各算一次 max(0, n·l) 后相加除二。
func Render(size int, h []float64, a, b Light) []float64 {
	ax, ay, az := vec(a)
	bx, by, bz := vec(b)
	out := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			var gx, gy float64
			switch {
			case x > 0 && x < size-1:
				gx = h[y*size+x+1] - h[y*size+x-1]
			case x == 0:
				gx = h[y*size+x+1] - h[y*size+x]
			default:
				gx = h[y*size+x] - h[y*size+x-1]
			}
			switch {
			case y > 0 && y < size-1:
				gy = h[(y+1)*size+x] - h[(y-1)*size+x]
			case y == 0:
				gy = h[(y+1)*size+x] - h[y*size+x]
			default:
				gy = h[y*size+x] - h[(y-1)*size+x]
			}
			m := math.Sqrt(1 + gx*gx + gy*gy)
			nx, ny, nz := -gx/m, -gy/m, 1/m
			la := math.Max(0, nx*ax+ny*ay+nz*az)
			lb := math.Max(0, nx*bx+ny*by+nz*bz)
			out[y*size+x] = clamp01((la + lb) / 2)
		}
	}
	return out
}
