package tilt

import (
	"math"
	"sort"
)

// Grid 是倾角图的一层，Index 与 api.Heightmap 的 Data 一一对应。
type Grid struct {
	Size int
	Tilt []float64 // 每格的倾角，度；边界一圈置 NaN，表示不参与统计
}

// Stats 是倾角统计结果，全部按度。
type Stats struct {
	Count     int     // 参与统计的格数（已排除边界一圈）
	MeanTilt  float64 // 平均倾角，度
	Median    float64 // 中位倾角，度
	P90       float64 // 九十分位倾角，度
	MaxTilt   float64 // 最大倾角，度
	MeanSlope float64 // 平均斜率，tan 前
	RMSSlope  float64 // 斜率均方根
}

// FromField 由高度场计算逐格倾角与统计量，边界一圈不计入统计。
//
// 梯度用中心差分（步长一格像素），倾角 = atan(|grad|)，再换算成度。
// atan 的值域是 (-pi/2, pi/2)，斜率恒非负，故倾角落在 [0, 90) 度，不涉及
// 象限选择；这里对 |grad| 取 atan，等价于第一象限。
func FromField(size int, data []float64) (Grid, Stats) {
	g := Grid{Size: size, Tilt: make([]float64, size*size)}
	for i := range g.Tilt {
		g.Tilt[i] = math.NaN()
	}
	var inner []float64
	var sumT, sumS, sumS2 float64
	for y := 1; y < size-1; y++ {
		for x := 1; x < size-1; x++ {
			gx := (data[y*size+x+1] - data[y*size+x-1]) / 2
			gy := (data[(y+1)*size+x] - data[(y-1)*size+x]) / 2
			s := math.Hypot(gx, gy)
			t := Degrees(math.Atan(s))
			g.Tilt[y*size+x] = t
			inner = append(inner, t)
			sumT += t
			sumS += s
			sumS2 += s * s
		}
	}
	n := len(inner)
	if n == 0 {
		return g, Stats{}
	}
	sort.Float64s(inner)
	st := Stats{
		Count:     n,
		MeanTilt:  sumT / float64(n),
		Median:    inner[n/2],
		P90:       inner[n*9/10],
		MaxTilt:   inner[n-1],
		MeanSlope: sumS / float64(n),
		RMSSlope:  math.Sqrt(sumS2 / float64(n)),
	}
	return g, st
}
