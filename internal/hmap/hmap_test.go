package hmap

import (
	"bytes"
	"testing"
)

func TestEncodePGMHeader(t *testing.T) {
	got := EncodePGM(2, []float64{0, 0.5, 0.5, 1})
	want := []byte("P5\n2 2\n255\n")
	if !bytes.HasPrefix(got, want) {
		t.Fatalf("illegal PGM header: %q", string(got[:len(want)+2]))
	}
	// 头部之后恰好是 size*size 个像素字节。
	if len(got)-len(want) != 4 {
		t.Fatalf("payload size = %d, want 4", len(got)-len(want))
	}
}

func TestNormalize(t *testing.T) {
	v := []float64{2, 4, 6}
	Normalize(v)
	if v[0] != 0 || v[1] != 0.5 || v[2] != 1 {
		t.Fatalf("normalize wrong: %v", v)
	}
}
