package noise

import (
	"math"
	"testing"
)

func TestValueHasPermutationPeriod256(t *testing.T) {
	s := New(7)
	points := [][2]float64{
		{0.3, 0.7},
		{17.25, -4.5},
		{-123.625, 89.125},
		{300.1, 700.9},
	}
	for _, p := range points {
		x, y := p[0], p[1]
		v := s.Value(x, y)
		for _, q := range [][2]float64{{x + 256, y}, {x, y + 256}, {x + 512, y + 256}} {
			got := s.Value(q[0], q[1])
			if math.Abs(got-v) > 1e-12 {
				t.Fatalf("period mismatch at (%v,%v)->(%v,%v): %.17g vs %.17g", x, y, q[0], q[1], v, got)
			}
		}
	}
}
