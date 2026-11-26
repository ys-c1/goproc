package hmap

import (
	"bytes"
	"testing"
)

func TestEncodePGMHeader(t *testing.T) {
	size := 4
	v := make([]float64, size*size)
	out := EncodePGM(size, v)
	wantHead := []byte("P5\n4 4\n255\n")
	if !bytes.HasPrefix(out, wantHead) {
		t.Fatalf("PGM 头非法: 前缀=%q", out[:len(wantHead)])
	}
	if len(out) != len(wantHead)+size*size {
		t.Fatalf("PGM 长度错误: got=%d want=%d", len(out), len(wantHead)+size*size)
	}
}

func TestEncodePGMRounding(t *testing.T) {
	out := EncodePGM(1, []float64{0.5})
	if out[len(out)-1] != 128 {
		t.Fatalf("四舍五入错误: got=%d want=128", out[len(out)-1])
	}
	clampHi := EncodePGM(1, []float64{2})
	if clampHi[len(clampHi)-1] != 255 {
		t.Fatalf("上界钳制失败")
	}
	clampLo := EncodePGM(1, []float64{-1})
	if clampLo[len(clampLo)-1] != 0 {
		t.Fatalf("下界钳制失败")
	}
}
