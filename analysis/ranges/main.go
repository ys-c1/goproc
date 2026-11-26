package main

import (
	"fmt"
	"math"

	"goproc/internal/fbm"
	"goproc/internal/noise"
)

func main() {
	for _, g := range []float64{0.3535533905932738, 0.5, 0.7071067811865476} {
		for k := 1; k <= 12; k++ {
			s := noise.New(7)
			lo, hi := math.Inf(1), math.Inf(-1)
			for y := 0; y < 128; y++ {
				for x := 0; x < 128; x++ {
					v := fbm.Eval(s, float64(x)*.02, float64(y)*.02, k, 2, g)
					lo = math.Min(lo, v)
					hi = math.Max(hi, v)
				}
			}
			fmt.Printf("g=%.6f k=%02d lo=%.6f hi=%.6f range=%.6f\n", g, k, lo, hi, hi-lo)
		}
	}
}
