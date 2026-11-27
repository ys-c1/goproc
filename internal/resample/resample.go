// Package resample 提供缩略图重采样：两个归一化采样核的低通下采样与最近邻放大。
//
// 约定 s = thumbSize / origSize（缩略图边长 / 原图边长），0 < s <= 1。
// 采样核在原图坐标上定义，宽度随 1/s 增长：
//   - box（盒形核，支撑半径 r = 0.5/s）：平均滤波，每维一阶
//   - triangle（三角核/线性核，支撑半径 r = 1/s）：每维二阶，支撑是盒核的两倍
//
// 两个核都经过每点权重归一化，s=1 时均为恒等映射。
package resample

import "math"

// Kind 标识采样核。
type Kind int

const (
	Box Kind = iota
	Triangle
)

// String 返回核名称。
func (k Kind) String() string {
	switch k {
	case Box:
		return "box"
	case Triangle:
		return "triangle"
	default:
		return "unknown"
	}
}

// radius 返回该核在缩放比 s 下的支撑半径（原图坐标单位）。
func (k Kind) radius(s float64) float64 {
	switch k {
	case Box:
		return 0.5 / s
	case Triangle:
		return 1.0 / s
	default:
		panic("unknown kernel")
	}
}

// weight 给出距采样中心距离 d 处的核值（未归一化）。
func (k Kind) weight(d float64) float64 {
	switch k {
	case Box:
		return 1
	case Triangle:
		return 1 - d
	default:
		panic("unknown kernel")
	}
}

// Downsample 把边长 n 的正方形图按缩放比 s 用核 k 低通下采样，返回边长 m 的图。
// 可分离实现，每行/列各自归一化，边界外按零填充处理。
func Downsample(src []float64, n int, s float64, k Kind) []float64 {
	m := int(math.Round(float64(n) * s))
	if m <= 0 || m > n {
		panic("resample: 缩放比越界")
	}
	r := k.radius(s)
	// 第一遍：水平方向，n*n -> n*m。
	tmp := make([]float64, n*m)
	for y := 0; y < n; y++ {
		for j := 0; j < m; j++ {
			cx := (float64(j) + 0.5) / s - 0.5
			lo := int(math.Ceil(cx - r))
			hi := int(math.Floor(cx + r))
			sum, wsum := 0.0, 0.0
			for x := lo; x <= hi; x++ {
				if x < 0 || x >= n {
					continue
				}
				w := k.weight(math.Abs(float64(x)-cx) * s)
				sum += src[y*n+x] * w
				wsum += w
			}
			tmp[y*m+j] = sum / wsum
		}
	}
	// 第二遍：垂直方向，n*m -> m*m。
	out := make([]float64, m*m)
	for j := 0; j < m; j++ {
		for i := 0; i < m; i++ {
			cy := (float64(i) + 0.5) / s - 0.5
			lo := int(math.Ceil(cy - r))
			hi := int(math.Floor(cy + r))
			sum, wsum := 0.0, 0.0
			for y := lo; y <= hi; y++ {
				if y < 0 || y >= n {
					continue
				}
				w := k.weight(math.Abs(float64(y)-cy) * s)
				sum += tmp[y*m+j] * w
				wsum += w
			}
			out[i*m+j] = sum / wsum
		}
	}
	return out
}

// UpscaleNearest 用最近邻把边长 m 的缩略图放大回边长 n，块复制不引入额外平滑。
func UpscaleNearest(thumb []float64, m, n int) []float64 {
	out := make([]float64, n*n)
	for y := 0; y < n; y++ {
		sy := y * m / n
		for x := 0; x < n; x++ {
			sx := x * m / n
			out[y*n+x] = thumb[sy*m+sx]
		}
	}
	return out
}
