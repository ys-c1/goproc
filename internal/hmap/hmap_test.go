package hmap

import (
	"bytes"
	"testing"
)

func TestEncodePGMHeader(t *testing.T) {
	got := EncodePGM(1, []float64{0.5})
	wantPrefix := []byte("P5\n1 1\n255\n")
	if !bytes.HasPrefix(got, wantPrefix) {
		t.Fatalf("PGM header = %q, want prefix %q", got, wantPrefix)
	}
}
