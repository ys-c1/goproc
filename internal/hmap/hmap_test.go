package hmap

import (
	"strings"
	"testing"
)

func TestEncodePGMHeaderIsBinaryP5(t *testing.T) {
	pgm := string(EncodePGM(1, []float64{0.5}))
	if !strings.HasPrefix(pgm, "P5\n1 1\n255\n") {
		t.Fatalf("invalid PGM header: %q", pgm[:min(len(pgm), 16)])
	}
	if strings.HasPrefix(pgm, `P5\n`) {
		t.Fatal("PGM header contains a literal backslash-n instead of a newline")
	}
}
