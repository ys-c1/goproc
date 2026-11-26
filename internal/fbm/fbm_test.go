package fbm

import (
	"math"
	"testing"

	"goproc/internal/noise"
)

func TestEvalLayersSumEqualsEval(t *testing.T) {
	src := noise.New(11)
	const x, y = 1.37, -2.91
	layers := EvalLayers(src, x, y, 4, 2, 0.5)
	sum := 0.0
	for _, v := range layers {
		sum += v
	}
	got := Eval(src, x, y, 4, 2, 0.5)
	if math.Abs(sum-got) > 1e-15 {
		t.Fatalf("分层和 %.17g 与 Eval %.17g 不一致", sum, got)
	}
}
