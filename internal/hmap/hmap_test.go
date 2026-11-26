package hmap

import (
	"bytes"
	"testing"
)

func TestEncodePGMHeaderUsesRealNewlines(t *testing.T) {
	got := EncodePGM(2, []float64{0, 0.25, 0.75, 1})
	wantPrefix := []byte("P5\n2 2\n255\n")
	if !bytes.HasPrefix(got, wantPrefix) {
		t.Fatalf("PGM header = %q, want prefix %q", got[:16], wantPrefix)
	}
}
