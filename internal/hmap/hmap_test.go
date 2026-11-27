package hmap

import (
	"bytes"
	"testing"
)

func TestNormalize(t *testing.T) {
	v := []float64{2, 4, 6}
	Normalize(v)
	want := []float64{0, 0.5, 1}
	for i := range v {
		if v[i] != want[i] {
			t.Fatalf("归一化错误: %v", v)
		}
	}
}

func TestEncodePGMHeader(t *testing.T) {
	out := EncodePGM(2, []float64{0, 1, 0.5, 0})
	// 必须是真正的 P5 二进制头：换行符是 0x0A，不是字面 "\n"
	wantHead := []byte("P5\n2 2\n255\n")
	if !bytes.HasPrefix(out, wantHead) {
		t.Fatalf("PGM 头非法: %q", out[:len(wantHead)])
	}
	if len(out) != len(wantHead)+4 {
		t.Fatalf("PGM 长度错误: %d", len(out))
	}
}
