package hmap

import "sort"

// Mapping 是由经验累积分布反解出的单调映射，把输入分布拉向均匀分布。
// 原理是概率积分变换：若 X 的累积分布函数是 F，则 F(X) 服从零到一的均匀分布，
// 所以取 T = F 即可把分布拉平，某点的拉伸量就是该处密度，可直接从样本秩次算出。
type Mapping struct {
	sorted []float64
}

// BuildEqualizer 根据样本构造拉平映射，样本不会被修改。
func BuildEqualizer(samples []float64) Mapping {
	s := make([]float64, len(samples))
	copy(s, samples)
	sort.Float64s(s)
	return Mapping{sorted: s}
}

// Apply 把 x 映到零到一，取值是 x 在样本中的经验累积分布函数值。
func (m Mapping) Apply(x float64) float64 {
	n := len(m.sorted)
	if n == 0 {
		return 0
	}
	r := sort.SearchFloat64s(m.sorted, x)
	return (float64(r) + 0.5) / float64(n)
}

// Equalize 原地拉平分布，返回所用的单调映射。
func Equalize(v []float64) Mapping {
	m := BuildEqualizer(v)
	for i := range v {
		v[i] = m.Apply(v[i])
	}
	return m
}

// HistCounts 把零到一的值分进 bins 个等宽箱子，返回各箱计数。
func HistCounts(v []float64, bins int) []int {
	counts := make([]int, bins)
	if bins <= 0 {
		return counts
	}
	for _, x := range v {
		b := int(x * float64(bins))
		if b < 0 {
			b = 0
		}
		if b >= bins {
			b = bins - 1
		}
		counts[b]++
	}
	return counts
}

// CountVariance 返回分箱计数的总体方差，越小说明分布越平。
func CountVariance(counts []int) float64 {
	n := len(counts)
	if n == 0 {
		return 0
	}
	sum := 0.0
	for _, c := range counts {
		sum += float64(c)
	}
	mean := sum / float64(n)
	acc := 0.0
	for _, c := range counts {
		d := float64(c) - mean
		acc += d * d
	}
	return acc / float64(n)
}
