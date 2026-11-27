package hmap

import (
	"bytes"
	"testing"
)

func TestEncodePGMHeader(t *testing.T) {
	out := EncodePGM(2, []float64{0, 0.5, 0.5, 1})
	want := []byte("P5\n2 2\n255\n")
	if !bytes.HasPrefix(out, want) {
		t.Fatalf("PGM 头部不合规，got %q", string(out[:len(want)]))
	}
	// 头部之后恰为 size*size 个灰度字节。
	if len(out) != len(want)+4 {
		t.Fatalf("PGM 长度异常: %d", len(out))
	}
}

func TestNormalize(t *testing.T) {
	v := []float64{2, 4, 6}
	Normalize(v)
	if v[0] != 0 || v[1] != 0.5 || v[2] != 1 {
		t.Fatalf("归一化错误: %v", v)
	}
}
