// Package noise 提供确定性的值噪声，种子派生用 splitmix64。
package noise

// Kernel 是值噪声在每个网格单元内使用的淡入淡出插值核 w:[0,1]→[0,1]。
// 同一张噪声图的光滑度完全由核在端点 t=0、t=1 的导数行为决定。
type Kernel interface {
	// Name 返回核的标识名。
	Name() string
	// Eval 计算 w(t)。
	Eval(t float64) float64
	// D1 是一阶导 w'(t)，D2 是二阶导 w''(t)。解析形式，供连续性证明的数值复核。
	D1(t float64) float64
	D2(t float64) float64
}

// CubicKernel 是经典 Hermite smoothstep：w3(t)=3t²−2t³。
// 它只保证格点处函数与一阶导连续（C¹），二阶导在格点处有 −12 的跳变。
type CubicKernel struct{}

func (CubicKernel) Name() string { return "cubic" }

func (CubicKernel) Eval(t float64) float64 { return t * t * (3 - 2*t) }
func (CubicKernel) D1(t float64) float64   { return 6 * t * (1 - t) }
func (CubicKernel) D2(t float64) float64   { return 6 - 12*t }

// QuinticKernel 是 Perlin 改进噪声的五次核：w5(t)=6t⁵−15t⁴+10t³。
// 端点处一二阶导全为零，格点处二阶导也连续（C²）。
type QuinticKernel struct{}

func (QuinticKernel) Name() string { return "quintic" }

func (QuinticKernel) Eval(t float64) float64 { return t * t * t * (t*(t*6-15) + 10) }
func (QuinticKernel) D1(t float64) float64   { return 30 * t * t * (1 - 2*t + t*t) }
func (QuinticKernel) D2(t float64) float64   { return 60 * t * (1 - 3*t + 2*t*t) }

// Source 是带种子的噪声源。
type Source struct {
	seed   uint64
	perm   [512]uint8
	kernel Kernel
}

func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// New 按种子构造噪声源，置换表由种子派生，不依赖全局状态。
// 插值核默认使用五次核（C² 连续）；历史输出保持不变。
func New(seed uint64) *Source {
	return NewWithKernel(seed, QuinticKernel{})
}

// NewWithKernel 按指定插值核构造噪声源，用于核之间的对照实验。
func NewWithKernel(seed uint64, k Kernel) *Source {
	s := &Source{seed: seed, kernel: k}
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

// Kernel 返回构造时选用的插值核。
func (s *Source) Kernel() Kernel { return s.kernel }

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
	u := s.kernel.Eval(fx)
	v := s.kernel.Eval(fy)
	c00 := s.grad(x0, y0)
	c10 := s.grad(x0+1, y0)
	c01 := s.grad(x0, y0+1)
	c11 := s.grad(x0+1, y0+1)
	return lerp(lerp(c00, c10, u), lerp(c01, c11, u), v)
}

// GradForTest 暴露格点值供实验程序核对（生产代码不应依赖）。
func (s *Source) GradForTest(ix, iy int) float64 { return s.grad(ix, iy) }

// Value1D 返回一维值噪声，专门用于格点连续性的一维解析验证。
func (s *Source) Value1D(x float64) float64 {
	x0 := int(floor(x))
	t := x - float64(x0)
	return lerp(s.grad(x0, 0), s.grad(x0+1, 0), s.kernel.Eval(t))
}

func floor(v float64) float64 {
	i := float64(int64(v))
	if v < i {
		return i - 1
	}
	return i
}
