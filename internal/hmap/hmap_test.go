package hmap

import (
	"bytes"
	"testing"
)

func TestEncodePGMHeader(t *testing.T) {
	out := EncodePGM(4, make([]float64, 16))
	// 头必须是真实换行符，不能是字面 \n。
	want := []byte("P5\n4 4\n255\n")
	if !bytes.HasPrefix(out, want) {
		t.Fatalf("PGM 头错误: %q", string(out[:len(want)]))
	}
	if len(out) != len(want)+16 {
		t.Fatalf("PGM 长度错误: %d", len(out))
	}
}

func TestNormalize(t *testing.T) {
	v := []float64{2, 4, 6}
	Normalize(v)
	if v[0] != 0 || v[1] != 0.5 || v[2] != 1 {
		t.Fatalf("归一化错误: %v", v)
	}
	flat := []float64{3, 3, 3}
	Normalize(flat)
	for _, x := range flat {
		if x != 0 {
			t.Fatalf("常数列应归零: %v", flat)
		}
	}
}
