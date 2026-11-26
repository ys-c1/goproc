package hmap

import (
	"bytes"
	"testing"
)

func TestEncodePGMHeaderIsValid(t *testing.T) {
	v := []float64{0, 0.5, 1, 0.25}
	b := EncodePGM(2, v)
	wantPrefix := []byte("P5\n2 2\n255\n")
	if !bytes.HasPrefix(b, wantPrefix) {
		t.Fatalf("PGM 头不是合法换行头: %q", b[:16])
	}
	if len(b) != len(wantPrefix)+4 {
		t.Fatalf("PGM 数据长度错误: %d", len(b))
	}
}

func TestNormalizeConstant(t *testing.T) {
	v := []float64{0.3, 0.3}
	Normalize(v)
	if v[0] != 0 || v[1] != 0 {
		t.Fatalf("常值场应归一为 0: %v", v)
	}
}
