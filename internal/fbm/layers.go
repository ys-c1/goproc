package fbm

import "goproc/internal/noise"

// EvalLayers 返回每一倍频层在归一化后的贡献；各层之和与 Eval 完全相同。
func EvalLayers(s *noise.Source, x, y float64, octaves int, lacunarity, gain float64) []float64 {
	layers := make([]float64, octaves)
	amp := 1.0
	freq := 1.0
	norm := 0.0
	for i := 0; i < octaves; i++ {
		layers[i] = s.Value(x*freq, y*freq) * amp
		norm += amp
		amp *= gain
		freq *= lacunarity
	}
	if norm == 0 {
		return layers
	}
	for i := range layers {
		layers[i] /= norm
	}
	return layers
}

// Eval 返回分形噪声，结果约在负一到一之间。
func Eval(s *noise.Source, x, y float64, octaves int, lacunarity, gain float64) float64 {
	layers := EvalLayers(s, x, y, octaves, lacunarity, gain)
	sum := 0.0
	for _, v := range layers {
		sum += v
	}
	return sum
}
