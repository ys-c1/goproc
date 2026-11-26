package hmap

import (
	"math"
	"testing"
)

// 构造一个中间多两头少的样本，模拟多倍频叠加后的钟形分布。
func bellSamples(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		u := float64(i+1) / float64(n+1)
		// 用反正切型的反函数造一批中间密集的确定性样本。
		v[i] = 0.5 + 0.5*math.Tanh(4*(u-0.5))/math.Tanh(2)
	}
	return v
}

func TestEqualizeFlattens(t *testing.T) {
	v := bellSamples(4096)
	before := CountVariance(HistCounts(v, 16))
	Equalize(v)
	after := CountVariance(HistCounts(v, 16))
	if after >= before {
		t.Fatalf("拉平后方差未下降: before=%v after=%v", before, after)
	}
}

func TestMappingMonotone(t *testing.T) {
	v := bellSamples(1024)
	m := BuildEqualizer(v)
	prev := m.Apply(0)
	for i := 1; i <= 100; i++ {
		cur := m.Apply(float64(i) / 100)
		if cur < prev {
			t.Fatalf("映射非单调: %v -> %v", prev, cur)
		}
		prev = cur
	}
}

func TestEqualizeKeepsRange(t *testing.T) {
	v := bellSamples(256)
	Equalize(v)
	for _, x := range v {
		if x < 0 || x > 1 {
			t.Fatalf("值域越界: %v", x)
		}
	}
}
