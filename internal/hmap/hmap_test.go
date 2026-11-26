package hmap

import (
	"strings"
	"testing"
)

func TestEncodePGMHeader(t *testing.T) {
	got := EncodePGM(2, []float64{0, 1, 0.5, 0.25})
	header := string(got[:15])
	if !strings.HasPrefix(header, "P5\n2 2\n255\n") {
		t.Fatalf("bad PGM header: %q", header)
	}
	if len(got) != len("P5\n2 2\n255\n")+4 {
		t.Fatalf("PGM length = %d, want header + 4 pixels", len(got))
	}
}
