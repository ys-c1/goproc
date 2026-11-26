// Package fbm 做分形叠加，把多层噪声按频率与振幅合成。
package fbm

import (
	"math"

	"goproc/internal/noise"
)

// Norm 返回 octaves 层在给定 gain 下的振幅和，用于归一化。
func Norm(octaves int, gain float64) float64 {
	norm := 0.0
	amp := 1.0
	for i := 0; i < octaves; i++ {
		norm += amp
		amp *= gain
	}
	return norm
}

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

// Octave 返回单独一层的加权贡献，已按总层数的振幅和归一化。
// 逐层贡献加起来就是 Eval，因此接缝跳变可以逐层拆开看。
func Octave(s *noise.Source, x, y float64, octave, octaves int, lacunarity, gain float64) float64 {
	if octaves <= 0 || octave < 0 || octave >= octaves {
		return 0
	}
	norm := Norm(octaves, gain)
	if norm == 0 {
		return 0
	}
	amp := math.Pow(gain, float64(octave))
	freq := math.Pow(lacunarity, float64(octave))
	return s.Value(x*freq, y*freq) * amp / norm
}
