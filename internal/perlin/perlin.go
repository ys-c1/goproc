// Package perlin 提供经典 Perlin 梯度噪声，置换表同样由种子 splitmix64 派生。
//
// 相比值噪声的可分离双线性/三次插值，梯度噪声在格点处为零、
// 谱能不沿网格轴泄漏，各向同性场的梯度方向接近均匀，避免 HOG 假峰。
package perlin

// Source 是带种子的 Perlin 噪声源。
type Source struct {
	seed uint64
	perm [512]uint8
}

var gradX = [8]float64{1, 0.7071067811865476, 0, -0.7071067811865476, -1, -0.7071067811865476, 0, 0.7071067811865476}
var gradY = [8]float64{0, 0.7071067811865476, 1, 0.7071067811865476, 0, -0.7071067811865476, -1, -0.7071067811865476}

func splitmix64(x uint64) uint64 {
	x += 0x9e3779b97f4a7c15
	x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9
	x = (x ^ (x >> 27)) * 0x94d049bb133111eb
	return x ^ (x >> 31)
}

// New 按种子构造噪声源，Fisher–Yates 洗牌与值噪声实现保持一致。
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

func fade(t float64) float64       { return t * t * t * (t*(t*6-15) + 10) }
func lerp(a, b, t float64) float64 { return a + (b-a)*t }

func (s *Source) dot(ix, iy int, dx, dy float64) float64 {
	h := s.perm[(int(s.perm[ix&255])+iy)&511] & 7
	return gradX[h]*dx + gradY[h]*dy
}

// Value 返回 Perlin 梯度噪声，值域约在负一到一之间。
func (s *Source) Value(x, y float64) float64 {
	x0 := int(floor(x))
	y0 := int(floor(y))
	fx := x - float64(x0)
	fy := y - float64(y0)
	u := fade(fx)
	v := fade(fy)
	n00 := s.dot(x0, y0, fx, fy)
	n10 := s.dot(x0+1, y0, fx-1, fy)
	n01 := s.dot(x0, y0+1, fx, fy-1)
	n11 := s.dot(x0+1, y0+1, fx-1, fy-1)
	return lerp(lerp(n00, n10, u), lerp(n01, n11, u), v)
}

// Eval 做分形叠加，把多层 Perlin 噪声按频率与振幅合成。
func Eval(s *Source, x, y float64, octaves int, lacunarity, gain float64) float64 {
	amp := 1.0
	freq := 1.0
	sum := 0.0
	norm := 0.0
	for i := 0; i < octaves; i++ {
		sum += s.Value(x*freq, y*freq) * amp
		norm += amp
		amp *= gain
		freq *= lacunarity
	}
	if norm == 0 {
		return 0
	}
	return sum / norm
}

func floor(v float64) float64 {
	i := float64(int64(v))
	if v < i {
		return i - 1
	}
	return i
}
