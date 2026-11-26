package hmap

import (
	"bytes"
	"testing"
)

func TestPGMHeader(t *testing.T) {
	data := make([]float64, 4)
	out := EncodePGM(2, data)
	want := []byte("P5\n2 2\n255\n")
	if !bytes.HasPrefix(out, want) {
		t.Fatalf("PGM 头非法: %q", string(out[:len(want)]))
	}
	if len(out) != len(want)+4 {
		t.Fatalf("PGM 长度异常: got %d want %d", len(out), len(want)+4)
	}
}

func TestNormalize(t *testing.T) {
	v := []float64{2, 4, 6}
	Normalize(v)
	if v[0] != 0 || v[1] != 0.5 || v[2] != 1 {
		t.Fatalf("归一化错误: %v", v)
	}
}
