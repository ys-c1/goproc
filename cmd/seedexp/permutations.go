// 种子扩散实验: 置换表派生方案与值噪声评估。
//
// 本文件给出三类置换表派生:
//   - production: 与 internal/noise 完全一致的 splitmix64 链式 Fisher-Yates;
//   - weak:       固定基础置换 + 仅顶部 k 次交换由种子决定(弱派生基线);
//   - hash:       每个条目位置独立用 h(seed,i) 决定交换(散列扰动)。
package main

import "math"

func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// basePerm 返回与种子无关的固定基础置换。
func basePerm() [256]uint8 {
	var p [256]uint8
	for i := range p {
		p[i] = uint8(i)
	}
	state := uint64(0x12345678)
	for i := 255; i > 0; i-- {
		state = splitmix64(state)
		j := int(state % uint64(i+1))
		p[i], p[j] = p[j], p[i]
	}
	return p
}

// productionPerm 复现 internal/noise.New 的置换表派生。
func productionPerm(seed uint64) [256]uint8 {
	var p [256]uint8
	for i := range p {
		p[i] = uint8(i)
	}
	state := seed
	for i := 255; i > 0; i-- {
		state = splitmix64(state)
		j := int(state % uint64(i+1))
		p[i], p[j] = p[j], p[i]
	}
	return p
}

// weakPerm 弱派生: 基础置换之上仅做顶部 k 次由种子散列决定的交换。
// 相邻种子的散列序列彼此独立, 但只有 k 个洗牌决策参与, 故改动被限制在约 2k 个条目。
func weakPerm(seed uint64, k int) [256]uint8 {
	p := basePerm()
	h := seed ^ 0xabcdef
	for t := 0; t < k; t++ {
		h = splitmix64(h)
		i := 255 - t
		j := int(h % uint64(i+1))
		p[i], p[j] = p[j], p[i]
	}
	return p
}

// hashPerm 散列扰动: 每个位置 i 独立计算 h(seed,i), 以概率 q 执行一次交换。
// q 即"每个条目被扰动的概率"; q=1 时每个条目都参与扰动。
func hashPerm(seed uint64, q float64) [256]uint8 {
	p := basePerm()
	const scale = float64(uint64(1) << 53)
	for i := 255; i > 0; i-- {
		h := splitmix64(seed ^ uint64(i)*0x9e3779b97f4a7c15)
		if float64(h>>11)/scale > q {
			continue // 该条目未被扰动
		}
		j := int(h % uint64(i+1))
		p[i], p[j] = p[j], p[i]
	}
	return p
}

// 三档种子间隔, 以及弱基线每档对应的顶部交换次数 k。
var gaps = []int{1, 16, 256}
var weakK = map[int]int{1: 1, 16: 2, 256: 3}

// 散列扰动扫描的每条目扰动概率 q。
var qLevels = []float64{0.10, 0.50, 0.90, 255.0 / 256.0, 1.0}

const (
	mapSize   = 128
	frequency = 0.02
	tau       = 0.10 // 相关面积判据的逐点高度差阈值
	trials    = 400  // 每档蒙特卡洛对数
)

// hammingDistance 统计两张置换表取值不同的条目数。
func hammingDistance(a, b [256]uint8) int {
	n := 0
	for i := range a {
		if a[i] != b[i] {
			n++
		}
	}
	return n
}

// ---- 值噪声(与 internal/noise 同构的单层值噪声, 便于隔离置换表效应) ----

type noiseSrc struct {
	perm [512]uint8
}

func newNoiseSrc(p [256]uint8) *noiseSrc {
	var s noiseSrc
	for i := 0; i < 512; i++ {
		s.perm[i] = p[i&255]
	}
	return &s
}

func (s *noiseSrc) grad(ix, iy int) float64 {
	h := s.perm[(int(s.perm[ix&255])+iy)&511]
	return float64(h)/127.5 - 1
}

func fade(t float64) float64 { return t * t * t * (t*(t*6-15) + 10) }

func lerp(a, b, t float64) float64 { return a + (b-a)*t }

func (s *noiseSrc) value(x, y float64) float64 {
	x0 := int(math.Floor(x))
	y0 := int(math.Floor(y))
	fx := x - float64(x0)
	fy := y - float64(y0)
	u, v := fade(fx), fade(fy)
	return lerp(
		lerp(s.grad(x0, y0), s.grad(x0+1, y0), u),
		lerp(s.grad(x0, y0+1), s.grad(x0+1, y0+1), u),
		v,
	)
}

// terrain 渲染归一化到零到一的单层值噪声高度图。
func terrain(p [256]uint8, size int, frequency float64) []float64 {
	s := newNoiseSrc(p)
	out := make([]float64, size*size)
	lo, hi := math.Inf(1), math.Inf(-1)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			v := s.value(float64(x)*frequency, float64(y)*frequency)
			out[y*size+x] = v
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
	}
	scale := 1 / (hi - lo)
	for i := range out {
		out[i] = (out[i] - lo) * scale
	}
	return out
}

// correlatedArea 返回两幅高度图逐点差小于 tau 的像素占比。
func correlatedArea(a, b []float64, tau float64) float64 {
	n := 0
	for i := range a {
		if math.Abs(a[i]-b[i]) < tau {
			n++
		}
	}
	return float64(n) / float64(len(a))
}

// pearson 返回两幅高度图的皮尔逊相关系数。
func pearson(a, b []float64) float64 {
	var sa, sb, sab, sa2, sb2 float64
	n := float64(len(a))
	for i := range a {
		sa += a[i]
		sb += b[i]
		sab += a[i] * b[i]
		sa2 += a[i] * a[i]
		sb2 += b[i] * b[i]
	}
	num := sab - sa*sb/n
	den := math.Sqrt((sa2 - sa*sa/n) * (sb2 - sb*sb/n))
	if den == 0 {
		return 0
	}
	return num / den
}
