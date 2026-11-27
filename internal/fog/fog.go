// Package fog 定义雾化程度度量：缩略图与原图缩放到同尺寸后的局部方差比。
//
// 度量流程：
//  1. 缩略图经最近邻放大回原图尺寸（块复制不引入额外平滑，仅做网格对齐）；
//  2. 用同一个连续盒形局部窗在两张图上算加权局部方差。原图窗半宽固定为
//     W/2，放大缩略图窗半宽为 W*s/2——两个窗覆盖原图上的同一物理区域。
//     窗半宽随 s 连续变化，边界像素按覆盖面积加权，避免整数窗边长取奇数
//     造成的台阶伪迹（整数匹配窗会让 R 只在八个离散窗宽之间跳变）；
//  3. 取全部内部点均值之比：
//
//	R = mean(加权局部方差_放大缩略图) / mean(加权局部方差_原图)
//	F = 1 - R          // 雾化程度，零为无雾，趋向一为完全抹平
package fog

import "math"

// cover1 返回像素中心 p 相对 1D 窗口 [c-r, c+r] 的覆盖长度（0..1）。
func cover1(c, r, p float64) float64 {
	lo := math.Max(p-0.5, c-r)
	hi := math.Min(p+0.5, c+r)
	if hi <= lo {
		return 0
	}
	return hi - lo
}

// WeightedLocalVariance 在边长 n 的图上以 (cx,cy) 为中心、半宽 r 的连续
// 盒窗计算加权总体方差。边界像素按窗口覆盖面积计权。
func WeightedLocalVariance(img []float64, n int, cx, cy, r float64) float64 {
	x0 := int(math.Floor(cx - r))
	x1 := int(math.Ceil(cx + r))
	y0 := int(math.Floor(cy - r))
	y1 := int(math.Ceil(cy + r))
	sw, swv, swv2 := 0.0, 0.0, 0.0
	for y := y0; y <= y1; y++ {
		if y < 0 || y >= n {
			continue
		}
		wy := cover1(cy, r, float64(y))
		if wy == 0 {
			continue
		}
		for x := x0; x <= x1; x++ {
			if x < 0 || x >= n {
				continue
			}
			wx := cover1(cx, r, float64(x))
			if wx == 0 {
				continue
			}
			w := wx * wy
			v := img[y*n+x]
			sw += w
			swv += w * v
			swv2 += w * v * v
		}
	}
	mean := swv / sw
	v := swv2/sw - mean*mean
	if v < 0 {
		return 0 // 浮点误差兜底
	}
	return v
}

// MeanWeightedVariance 对全部完整落在图内的内部点求加权局部方差均值。
func MeanWeightedVariance(img []float64, n int, r float64) float64 {
	ri := int(math.Ceil(r))
	sum := 0.0
	count := 0
	for y := ri; y < n-ri; y++ {
		for x := ri; x < n-ri; x++ {
			sum += WeightedLocalVariance(img, n, float64(x), float64(y), r)
			count++
		}
	}
	return sum / float64(count)
}

// Measure 返回局部方差比 R 与雾化程度 F=1-R。
// orig 为边长 n 的原图；up 为缩略图最近邻放大回 n 的图；
// W 为原图参考窗边长，s 为该缩略图实际缩放比。
func Measure(orig, up []float64, n, W int, s float64) (r, f float64) {
	r0 := float64(W) / 2
	rUp := r0 * s
	vOrig := MeanWeightedVariance(orig, n, r0)
	vUp := MeanWeightedVariance(up, n, rUp)
	r = vUp / vOrig
	return r, 1 - r
}
