// Package fbm 做分形叠加，把多层噪声按频率与振幅合成。
package fbm

import "goproc/internal/noise"

// Eval 返回分形噪声，结果约在负一到一之间。
func Eval(s *noise.Source, x, y float64, octaves int, lacunarity, gain float64) float64 {
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

// Trace 记录点 (x,y) 在全部八度求值中访问到的置换表槽位。
func Trace(s *noise.Source, x, y float64, octaves int, lacunarity, gain float64, slots map[int]struct{}) {
	freq := 1.0
	for i := 0; i < octaves; i++ {
		s.TraceValue(x*freq, y*freq, slots)
		freq *= lacunarity
	}
}
