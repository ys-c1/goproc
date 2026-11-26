// Package noise 提供确定性的值噪声，种子派生用 splitmix64。
package noise

// Source 是带种子的噪声源。
type Source struct {
	seed uint64
	perm [512]uint8
}

func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// New 按种子构造噪声源，置换表由种子派生，不依赖全局状态。
func New(seed uint64) *Source {
	s := &Source{seed: seed}
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
	for i := 0; i < 512; i++ {
		s.perm[i] = p[i&255]
	}
	return s
}

// LayerSeed 从主种子与层号派生第 layer 层专属子表的种子。
//
// 派生用 splitmix64 链加固定域分隔常数：同一主种子不同层得到互不相关的
// 子种子，不同主种子也不相关，且完全确定。它是“按层派生子表”的唯一派生
// 规则，分析脚本 analysis/goproc_port.py 逐式镜像本函数。
func LayerSeed(master uint64, layer int) uint64 {
	h := splitmix64(master + 0x9E3779B97F4A7C15)
	h ^= splitmix64(uint64(layer) + 0x243F6A8885A308D3)
	return splitmix64(h)
}

// NewLayer 构造第 layer 层专属的噪声源，其置换子表由 (master, layer) 派生。
func NewLayer(master uint64, layer int) *Source {
	return New(LayerSeed(master, layer))
}

// Seed 返回构造时的种子。
func (s *Source) Seed() uint64 { return s.seed }

func fade(t float64) float64 { return t * t * t * (t*(t*6-15) + 10) }

func lerp(a, b, t float64) float64 { return a + (b-a)*t }

func (s *Source) grad(ix, iy int) float64 {
	h := s.perm[(int(s.perm[ix&255])+iy)&511]
	return float64(h)/127.5 - 1
}

// Value 返回一个格点插值后的噪声值，值域约在负一到一之间。
func (s *Source) Value(x, y float64) float64 {
	x0 := int(floor(x))
	y0 := int(floor(y))
	fx := x - float64(x0)
	fy := y - float64(y0)
	u := fade(fx)
	v := fade(fy)
	c00 := s.grad(x0, y0)
	c10 := s.grad(x0+1, y0)
	c01 := s.grad(x0, y0+1)
	c11 := s.grad(x0+1, y0+1)
	return lerp(lerp(c00, c10, u), lerp(c01, c11, u), v)
}

func floor(v float64) float64 {
	i := float64(int64(v))
	if v < i {
		return i - 1
	}
	return i
}
