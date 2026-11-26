package hmap

import (
	"bytes"
	"testing"
)

func TestEncodePGMHeaderUsesRealNewlines(t *testing.T) {
	got := EncodePGM(1, []float64{1})
	wantPrefix := []byte("P5\n1 1\n255\n")
	if !bytes.HasPrefix(got, wantPrefix) {
		t.Fatalf("PGM output = %q, want prefix %q", got, wantPrefix)
	}
}
