// Package fbm 做分形叠加，把多层噪声按频率与振幅合成。
package fbm

import "goproc/internal/noise"

// Eval 返回分形噪声，结果约在负一到一之间。
//
// 所有层共用同一个 *noise.Source（同一张置换表）。这是产生跨层尺度同源
// 重复山脊的旧路径，保留用于对照与复现；新代码应使用 EvalLayered。
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

// EvalLayered 与 Eval 的合成规则完全相同，但第 i 层使用 layers[i] 这张
// 按层派生的独立置换子表，从而切断共用置换表导致的跨层恒等复制。
// 调用方通常用 noise.NewLayer(masterSeed, i) 预先构造好各层 Source 并在
// 逐像素循环外复用，避免每像素重建置换表。
func EvalLayered(layers []*noise.Source, x, y float64, octaves int, lacunarity, gain float64) float64 {
	amp := 1.0
	freq := 1.0
	sum := 0.0
	norm := 0.0
	for i := 0; i < octaves; i++ {
		sum += layers[i].Value(x*freq, y*freq) * amp
		norm += amp
		amp *= gain
		freq *= lacunarity
	}
	if norm == 0 {
		return 0
	}
	return sum / norm
}
