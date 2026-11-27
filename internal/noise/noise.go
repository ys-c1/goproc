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

// TraceValue 记录 Value(x,y) 求值时四个角点实际访问的全部置换表槽位。
// grad(ix,iy) 先读槽 ix&255，再读槽 (perm[ix&255]+iy)&511，两处都记录。
func (s *Source) TraceValue(x, y float64, slots map[int]struct{}) {
	x0 := int(floor(x))
	y0 := int(floor(y))
	for _, ix := range []int{x0, x0 + 1} {
		for _, iy := range []int{y0, y0 + 1} {
			slots[ix&255] = struct{}{}
			slots[(int(s.perm[ix&255])+iy)&511] = struct{}{}
		}
	}
}

// Snapshot 读出给定槽位上的条目，槽位统一折算到零到二百五十五。
func (s *Source) Snapshot(slots map[int]struct{}) map[int]uint8 {
	out := make(map[int]uint8, len(slots))
	for k := range slots {
		j := k & 255
		out[j] = s.perm[j]
	}
	return out
}

// ApplyEntries 把条目写入指定槽位，并自动同步置换表双倍长度的镜像段。
func (s *Source) ApplyEntries(entries map[int]uint8) {
	for k, v := range entries {
		j := k & 255
		s.perm[j] = v
		s.perm[j+256] = v
	}
}

// Entry 读取零到二百五十五号槽位上的置换条目。
func (s *Source) Entry(slot int) uint8 { return s.perm[slot&255] }

func floor(v float64) float64 {
	i := float64(int64(v))
	if v < i {
		return i - 1
	}
	return i
}
