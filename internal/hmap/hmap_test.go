package hmap

import (
	"bytes"
	"testing"
)

func TestEncodePGMHeader(t *testing.T) {
	v := []float64{0, 0.5, 1, 1} // 2x2
	out := EncodePGM(2, v)
	// PGM 头必须是真实换行符，不能是字面量 \n。
	if !bytes.HasPrefix(out, []byte("P5\n2 2\n255\n")) {
		t.Fatalf("PGM 头非法: %q", out[:12])
	}
	// 头之后恰为 size*size 个灰度字节。
	body := out[len("P5\n2 2\n255\n"):]
	if len(body) != 4 {
		t.Fatalf("灰度字节数=%d，期望 4", len(body))
	}
}

func TestNormalize(t *testing.T) {
	v := []float64{2, 4, 6}
	Normalize(v)
	if v[0] != 0 || v[1] != 0.5 || v[2] != 1 {
		t.Fatalf("归一化错误: %v", v)
	}
}
