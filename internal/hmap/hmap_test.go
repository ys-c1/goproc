package hmap

import (
	"strings"
	"testing"
)

func TestEncodePGMHeader(t *testing.T) {
	v := []float64{0, 0.5, 1, 0.25}
	b := EncodePGM(2, v)
	s := string(b)
	if !strings.HasPrefix(s, "P5\n2 2\n255\n") {
		t.Fatalf("PGM 头不是合法的真实换行头: %q", s[:20])
	}
	if strings.Contains(s[:len("P5\n2 2\n255\n")], `\n`) {
		t.Fatal("PGM 头里出现了字面量 \\n 而非换行")
	}
	head := len("P5\n2 2\n255\n")
	if len(b) != head+4 {
		t.Fatalf("PGM 数据长度错误: got %d want %d", len(b), head+4)
	}
}

func TestNormalizeFlat(t *testing.T) {
	v := []float64{7, 7, 7}
	Normalize(v)
	for _, x := range v {
		if x != 0 {
			t.Fatalf("平坦场应归一为零, got %v", x)
		}
	}
}
