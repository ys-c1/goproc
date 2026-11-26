package hmap

import (
	"bytes"
	"testing"
)

func TestEncodePGMHeaderUsesRealNewlines(t *testing.T) {
	got := EncodePGM(2, []float64{0, 0.25, 0.75, 1})
	wantPrefix := []byte("P5\n2 2\n255\n")
	if !bytes.HasPrefix(got, wantPrefix) {
		t.Fatalf("PGM prefix = %q, want %q", got[:min(len(got), 16)], wantPrefix)
	}
	if len(got) != len(wantPrefix)+4 {
		t.Fatalf("PGM length = %d, want %d raster bytes", len(got), len(wantPrefix)+4)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
