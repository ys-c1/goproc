// Package sampling 依据拉伸后的局部频率给出各向异性奈奎斯特采样步长。
package sampling

import "math"

// Steps 返回物理坐标下两个主轴允许的最大采样步长。
//
// fBase 是各向同性基础噪声的带边频率（周/单位距离），rX、rY 是两轴相对
// 基础场的坐标缩放比（场点 q = (rX*x, rY*y)）。缩放后该轴的局部频率为
// f = r*fBase，奈奎斯特要求采样步长 h = 1/(2*f) = 1/(2*r*fBase)，
// 即采样数刚好压在奈奎斯特边上。
func Steps(fBase, rX, rY float64) (hX, hY float64) {
	return 1 / (2 * fBase * rX), 1 / (2 * fBase * rY)
}

// Counts 给边长 L 的物理域（半开区间 [0,L)），返回两轴所需采样数。
func Counts(L, hX, hY float64) (nX, nY int) {
	return int(math.Ceil(L / hX)), int(math.Ceil(L / hY))
}
