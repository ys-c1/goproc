package hmap

import (
	"bytes"
	"strings"
	"testing"
)

func TestEncodePGMHeaderHasRealNewlines(t *testing.T) {
	got := EncodePGM(2, []float64{0, 0.25, 0.75, 1})
	header, _, ok := bytes.Cut(got, []byte("\n255\n"))
	preview := got
	if len(preview) > 32 {
		preview = preview[:32]
	}
	if !ok {
		t.Fatalf("PGM 头缺少真实换行: %q", preview)
	}
	if !strings.HasPrefix(string(header), "P5\n2 2") {
		t.Fatalf("PGM 头非法: %q", header)
	}
}
