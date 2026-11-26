// Package noise 提供确定性的值噪声，种子派生用 splitmix64。
package noise

// DefaultPeriod 是默认的置换表周期，单位是格点。
const DefaultPeriod = 256

// Source 是带种子的噪声源。置换表长度就是周期：
// Value 沿 x、y 两轴都以 Period 个格点重复。
type Source struct {
	seed   uint64
	period int
	perm   []int
}

func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

func mod(a, n int) int {
	r := a % n
	if r < 0 {
		r += n
	}
	return r
}

// New 按种子构造默认周期的噪声源。
func New(seed uint64) *Source { return NewPeriod(seed, DefaultPeriod) }

// NewPeriod 按种子与置换表周期构造噪声源。period 是格点数，
// 取值必须不小于二；表长就是 period，所以 Value(x+period, y) == Value(x, y)。
func NewPeriod(seed uint64, period int) *Source {
	if period < 2 {
		period = DefaultPeriod
	}
	p := make([]int, period)
	for i := range p {
		p[i] = i
	}
	state := seed
	for i := period - 1; i > 0; i-- {
		state = splitmix64(state)
		j := int(state % uint64(i+1))
		p[i], p[j] = p[j], p[i]
	}
	perm := make([]int, 2*period)
	for i := range perm {
		perm[i] = p[i%period]
	}
	return &Source{seed: seed, period: period, perm: perm}
}

// Seed 返回构造时的种子。
func (s *Source) Seed() uint64 { return s.seed }

// Period 返回置换表周期，单位是格点。
func (s *Source) Period() int { return s.period }

func fade(t float64) float64 { return t * t * t * (t*(t*6-15) + 10) }

func lerp(a, b, t float64) float64 { return a + (b-a)*t }

// grad 取格点上的伪随机值。索引全部对 period 或两倍 period 取模，
// 因此 grad(ix+period, iy) == grad(ix, iy)、grad(ix, iy+period) == grad(ix, iy)。
func (s *Source) grad(ix, iy int) float64 {
	base := s.perm[mod(ix, s.period)]
	h := s.perm[mod(base+iy, 2*s.period)]
	return float64(h)/(float64(s.period-1)/2) - 1
}

// Value 返回一个格点插值后的噪声值，值域约在负一到一之间。
// 它沿两轴都以 Period 为周期。
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
