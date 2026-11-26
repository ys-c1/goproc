package axis

import (
	"math"
	"testing"
)

func TestPredictAxisRecoversKnownColumn(t *testing.T) {
	for _, width := range []int{96, 128} {
		for _, a := range []int{width / 3, width / 2, 2 * width / 3} {
			c := Generate("test", width, width, a, 4, 42)
			p := PredictAxis(c.Layers)
			if math.Abs(p.AxisFloat-float64(a)) > 1e-9 {
				t.Fatalf("width=%d axis=%d phase axis=%.12f", width, a, p.AxisFloat)
			}
			if p.RoundedAxis != a {
				t.Fatalf("rounded prediction %d != %d", p.RoundedAxis, a)
			}
		}
	}
}

func TestGeneratedTerrainIsPixelSymmetric(t *testing.T) {
	c := Generate("sym", 128, 96, 61, 4, 7)
	got := ConfirmAxis(c, 61)
	if got.Mismatches != 0 {
		t.Fatalf("expected exact symmetry, got %d float mismatches (max %g)", got.Mismatches, got.MaxAbsDiff)
	}
	if got.QuantizedBad != 0 {
		t.Fatalf("expected quantized symmetry, got %d mismatches", got.QuantizedBad)
	}
	if got.ExactEqual != got.Pairs || got.Pairs == 0 {
		t.Fatalf("bad pair count: %+v", got)
	}
}

func TestCorrelationMaximumAtPhaseAxis(t *testing.T) {
	for _, width := range []int{96, 128} {
		c := Generate("scan", width, width, int(math.Round(0.47*float64(width-1))), 4, 99)
		p := PredictAxis(c.Layers)
		scores := ScanColumns(c, width/4)
		best, tied := BestColumn(scores, 1e-12)
		if best.Column != p.RoundedAxis {
			t.Fatalf("width=%d phase=%d scan=%d tied=%v r=%.12f", width, p.RoundedAxis, best.Column, tied, best.R)
		}
		atAxis := MirrorCorrelation(c.Data, c.Width, c.Height, c.Axis)
		if math.Abs(atAxis.R-1) > 1e-12 {
			t.Fatalf("axis correlation = %.15f, want 1", atAxis.R)
		}
	}
}

func TestWrongColumnIsNotConfirmed(t *testing.T) {
	c := Generate("wrong", 128, 128, 64, 4, 5)
	wrong := ConfirmAxis(c, 58)
	if wrong.Mismatches == 0 {
		t.Fatal("six-column displaced candidate should not be exactly symmetric")
	}
}

func TestEdgeSelfPairColumnIsDegenerate(t *testing.T) {
	c := Generate("edge", 96, 96, 47, 4, 8)
	for _, col := range []int{0, c.Width - 1} {
		got := ConfirmAxis(c, col)
		if got.Viable {
			t.Fatalf("column %d should be self-pair degenerate: %+v", col, got)
		}
		if got.NonSelfObservations != 0 {
			t.Fatalf("column %d has %d non-self observations, want 0", col, got.NonSelfObservations)
		}
	}

	scores := ScanColumns(c, 3)
	for _, s := range scores {
		if s.Column == 0 || s.Column == c.Width-1 {
			t.Fatalf("degenerate edge column %d must be excluded from scan", s.Column)
		}
	}
	best, _ := BestColumn(scores, 1e-12)
	if best.Column != 47 {
		t.Fatalf("nondegenerate scan best=%d, want 47", best.Column)
	}
}
