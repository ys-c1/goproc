package hmap

import (
	"strings"
	"testing"
)

func TestEncodePGMHeader(t *testing.T) {
	v := []float64{0, 0.5, 1, 0.25}
	b := EncodePGM(2, v)
	// P5 头必须含真正的换行，而不是字面量 "\n"。
	if strings.Contains(string(b), `\n`) {
		t.Fatalf("PGM 头含字面量反斜杠 n: %q", b)
	}
	if !strings.HasPrefix(string(b), "P5\n2 2\n255\n") {
		t.Fatalf("PGM 头不正确: %q", b[:16])
	}
	if len(b) != len("P5\n2 2\n255\n")+4 {
		t.Fatalf("PGM 数据长度错误: %d", len(b))
	}
}

func TestNormalize(t *testing.T) {
	v := []float64{2, 4, 6}
	Normalize(v)
	if v[0] != 0 || v[2] != 1 || v[1] != 0.5 {
		t.Fatalf("归一化错误: %v", v)
	}
}
