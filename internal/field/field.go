// Package field 生成纹理主方向已知的定向地形。
//
// 定向纹理在局部坐标系内构造为沿固定方向传播的平稳多频正弦场叠加弱
// 伪随机扰动，再把采样坐标旋转 phi 度注入主方向。平稳性保证整幅旋转
// 后任意同尺寸窗口的结构张量统计量不变，从而可以干净地验证“旋转对齐”
// 本身，而不混入窗口内容漂移。方向约定与 orient 包一致（x 向右、y 向下，
// 角度单位度，模 180° 无向）。
package field

import (
	"math"
)

// splitmix64 生成确定性的伪随机扰动，种子派生不依赖全局状态。
func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// Options 描述一块定向地形。
type Options struct {
	Size      int
	Seed      uint64
	PhiDeg    float64 // 注入的纹理主方向角（度）
	Lambda    float64 // 基频波长（像素）
	Harmonics int     // 谐波数量
	Noise     float64 // 空间相关低频扰动的相对振幅
}

// valueNoise 返回双线性插值的平滑值噪声，保证扰动在空间上相关，
// 与真实 fbm 地形的纹理统计一致，而不是逐像素白噪声。
func valueNoise(perm [512]uint8, x, y float64) float64 {
	xi := int(math.Floor(x))
	yi := int(math.Floor(y))
	fx := x - float64(xi)
	fy := y - float64(yi)
	sm := func(t float64) float64 { return t * t * (3 - 2*t) }
	u, v := sm(fx), sm(fy)
	g := func(ix, iy int) float64 {
		h := perm[(int(perm[ix&255])+iy)&511]
		return float64(h)/127.5 - 1
	}
	a := g(xi, yi) + (g(xi+1, yi)-g(xi, yi))*u
	b := g(xi, yi+1) + (g(xi+1, yi+1)-g(xi, yi+1))*u
	return a + (b-a)*v
}

// Generate 生成零到一归一化的定向高度图，返回行主序数据。
func Generate(o Options) []float64 {
	phi := o.PhiDeg * math.Pi / 180.0
	cos, sin := math.Cos(phi), math.Sin(phi)

	// 由种子派生置换表（splitmix64），与 internal/noise 同思路但自包含。
	var perm [512]uint8
	var p [256]uint8
	for i := range p {
		p[i] = uint8(i)
	}
	state := o.Seed
	for i := 255; i > 0; i-- {
		state = splitmix64(state)
		j := int(state % uint64(i+1))
		p[i], p[j] = p[j], p[i]
	}
	for i := range perm {
		perm[i] = p[i&255]
	}

	d := make([]float64, o.Size*o.Size)
	const noiseScale = 0.06 // 扰动空间频率，明显低于纹理基频
	for y := 0; y < o.Size; y++ {
		for x := 0; x < o.Size; x++ {
			// 图像坐标旋转 -phi 取回纹理局部坐标。
			lx := float64(x)*cos + float64(y)*sin
			v := 0.0
			amp := 1.0
			norm := 0.0
			for k := 1; k <= o.Harmonics; k++ {
				v += amp * math.Sin(2*math.Pi*float64(k)*lx/o.Lambda)
				norm += amp
				amp *= 0.6
			}
			v = v/norm + o.Noise*valueNoise(perm, float64(x)*noiseScale, float64(y)*noiseScale)
			d[y*o.Size+x] = v
		}
	}

	lo, hi := d[0], d[0]
	for _, v := range d {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	if hi > lo {
		for i := range d {
			d[i] = (d[i] - lo) / (hi - lo)
		}
	}
	return d
}
