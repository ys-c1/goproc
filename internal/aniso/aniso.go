// Package aniso 在 Perlin 梯度噪声/FBM 之上构造多分量各向异性地形。
//
// 每个分量沿走向 Strike 方向拉长：局部坐标 u 沿走向、v 垂直走向，
// 采样时 v 方向频率乘轴比 Ratio，谱支集随之在垂直走向方向拉长 Ratio 倍，
// 地形脊线沿走向延伸，梯度主方向垂直于走向。
//
// 两种合成方式：
//   - Field：相干线性叠加，对应同源同时期构造；梯度矢量相加产生交叉项，
//     会把多个脊系方向合并，实测见报告的相干性实验。
//   - FieldRMS：非相干能量合成 z=sqrt(Σwᵢzᵢ²)，对应不同期次、独立成因的
//     脊系；保持 C¹ 光滑且无符号抵消，各脊系方向在 HOG 中独立成峰。
package aniso

import (
	"math"

	"goproc/internal/hmap"
	"goproc/internal/perlin"
)

// Component 描述一个定向地形分量。
type Component struct {
	Seed      uint64  // 独立噪声种子，保证分量间相互独立
	Strike    float64 // 走向角（弧度），地形脊线延伸方向
	Ratio     float64 // 轴比，>=1，垂直走向相对沿走向的频率倍数
	Frequency float64 // 沿走向的基础采样频率
	Weight    float64 // 叠加权重
}

func evalComponent(src *perlin.Source, c Component, x, y float64,
	octaves int, lacunarity, gain float64) float64 {
	cosT, sinT := math.Cos(c.Strike), math.Sin(c.Strike)
	u := x*cosT + y*sinT
	v := -x*sinT + y*cosT
	return perlin.Eval(src, u*c.Frequency, v*c.Frequency*c.Ratio,
		octaves, lacunarity, gain)
}

// Field 相干线性叠加多个定向 Perlin-FBM 分量，结果归一化到 [0,1]。
func Field(size, octaves int, lacunarity, gain float64, comps []Component) []float64 {
	out := make([]float64, size*size)
	srcs := make([]*perlin.Source, len(comps))
	for i, c := range comps {
		srcs[i] = perlin.New(c.Seed)
	}
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			fx, fy := float64(x), float64(y)
			var sum float64
			for i, c := range comps {
				sum += c.Weight * evalComponent(srcs[i], c, fx, fy, octaves, lacunarity, gain)
			}
			out[y*size+x] = sum
		}
	}
	hmap.Normalize(out)
	return out
}

// FieldRMS 非相干能量合成多个独立脊系，结果归一化到 [0,1]。
// 每个分量独立生成后按 z=sqrt(Σwᵢzᵢ²) 合成，C¹ 光滑、无符号抵消。
func FieldRMS(size, octaves int, lacunarity, gain float64, comps []Component) []float64 {
	acc := make([]float64, size*size)
	for _, c := range comps {
		src := perlin.New(c.Seed)
		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				v := evalComponent(src, c, float64(x), float64(y), octaves, lacunarity, gain)
				acc[y*size+x] += c.Weight * v * v
			}
		}
	}
	for i := range acc {
		acc[i] = math.Sqrt(acc[i])
	}
	hmap.Normalize(acc)
	return acc
}
