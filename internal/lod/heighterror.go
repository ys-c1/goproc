package lod

import "math"

// at 返回高度图 (gx,gy) 处值，越界按边缘钳制。
func at(h []float64, n, gx, gy int) float64 {
	if gx < 0 {
		gx = 0
	} else if gx >= n {
		gx = n - 1
	}
	if gy < 0 {
		gy = 0
	} else if gy >= n {
		gy = n - 1
	}
	return h[gy*n+gx]
}

// Bilinear 用步长 d 的对齐粗网格对连续像素坐标 (x,y) 做双线性重建。
// 粗网格点位于 (i*d, j*d)，高程直接取全分辨率参考图在该点的真值。
func Bilinear(h []float64, n, d int, x, y float64) float64 {
	gx := x / float64(d)
	gy := y / float64(d)
	x0 := int(math.Floor(gx))
	y0 := int(math.Floor(gy))
	fx := gx - float64(x0)
	fy := gy - float64(y0)
	px0, py0 := x0*d, y0*d
	c00 := at(h, n, px0, py0)
	c10 := at(h, n, px0+d, py0)
	c01 := at(h, n, px0, py0+d)
	c11 := at(h, n, px0+d, py0+d)
	top := c00 + (c10-c00)*fx
	bot := c01 + (c11-c01)*fx
	return top + (bot-top)*fy
}

// MaxError 计算步长 d 的 LOD 双线性重建相对全分辨率参考图的最大绝对高程误差。
// 返回最大误差及其像素位置。高程域与输入一致（api.Generate 已归一化到 [0,1]）。
func MaxError(ref []float64, n, d int) (worst float64, atX, atY int) {
	worst = 0.0
	wx, wy := 0, 0
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			e := math.Abs(Bilinear(ref, n, d, float64(x), float64(y)) - ref[y*n+x])
			if e > worst {
				worst, wx, wy = e, x, y
			}
		}
	}
	return worst, wx, wy
}

// MaxErrorBand 只在水平接缝 y=Y 上下 half 像素的条带内统计最大重建误差。
func MaxErrorBand(ref []float64, n, d, Y, half int) float64 {
	worst := 0.0
	y0 := Y - half
	y1 := Y + half
	for y := y0; y <= y1; y++ {
		if y < 0 || y >= n {
			continue
		}
		for x := 0; x < n; x++ {
			e := math.Abs(Bilinear(ref, n, d, float64(x), float64(y)) - ref[y*n+x])
			if e > worst {
				worst = e
			}
		}
	}
	return worst
}

// GrayLevels 把 [0,1] 高程误差换算成 8 位灰度级误差（×255）。
func GrayLevels(e float64) float64 { return e * 255.0 }
