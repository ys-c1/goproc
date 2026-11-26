package hmap

import (
	"bytes"
	"testing"
)

func TestEncodePGMHeader(t *testing.T) {
	got := EncodePGM(3, []float64{0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1})
	// 头部必须使用真实换行，尺寸与 maxval 是可解析的裸 token，不能被注释掉。
	want := []byte("P5\n3 3\n255\n")
	if !bytes.HasPrefix(got, want) {
		t.Fatalf("header=%q want prefix %q", string(got[:min(len(got), 16)]), string(want))
	}
	if len(got) != len(want)+9 {
		t.Fatalf("length=%d want %d", len(got), len(want)+9)
	}
	// 0/0.5/1 分别量化为 0/128/255（四舍五入）。
	pixels := got[len(want):]
	wantPx := []byte{0, 128, 255, 0, 128, 255, 0, 128, 255}
	for i, b := range pixels {
		if b != wantPx[i] {
			t.Fatalf("pixel %d = %d want %d", i, b, wantPx[i])
		}
	}
}

func TestNormalize(t *testing.T) {
	v := []float64{2, 4, 6}
	Normalize(v)
	if v[0] != 0 || v[1] != 0.5 || v[2] != 1 {
		t.Fatalf("normalized=%v want [0 0.5 1]", v)
	}
	flat := []float64{3, 3, 3}
	Normalize(flat)
	for _, x := range flat {
		if x != 0 {
			t.Fatalf("flat normalize=%v want all zero", flat)
		}
	}
}
