package hmap

import (
	"bytes"
	"testing"
)

func TestEncodePGMHeaderUsesRealNewlines(t *testing.T) {
	out := EncodePGM(2, []float64{0, 0.25, 0.75, 1})
	wantPrefix := []byte("P5\n2 2\n255\n")
	if !bytes.HasPrefix(out, wantPrefix) {
		t.Fatalf("PGM prefix = %q, want %q", out[:12], wantPrefix)
	}
	if len(out) != len(wantPrefix)+4 {
		t.Fatalf("PGM length = %d, want %d", len(out), len(wantPrefix)+4)
	}
}
