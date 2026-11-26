package hmap

import (
	"strings"
	"testing"
)

func TestEncodePGMHeaderUsesRealNewlines(t *testing.T) {
	got := EncodePGM(2, []float64{0, 0.25, 0.75, 1})
	if !strings.HasPrefix(string(got), "P5\n2 2\n255\n") {
		t.Fatalf("invalid PGM header: %q", got[:16])
	}
	if got[len("P5\n2 2\n255\n")] != 0 || got[len(got)-1] != 255 {
		t.Fatalf("unexpected raster bytes: %v", got)
	}
}
