package api_test

import (
	"testing"

	"goproc/api"
)

// TestBlockPeriodCandidates verifies the theorem-derived candidate offsets on
// production-generated heightmaps. Coordinates are block coordinates; each
// block is 32x32 cells.
func TestBlockPeriodCandidates(t *testing.T) {
	const (
		size  = 512
		block = 32
		grid  = size / block
	)

	cases := []struct {
		name      string
		frequency float64
		q         int // expected period in blocks
		wantCount int
	}{
		{name: "P*s=128px", frequency: 2, q: 4, wantCount: 48},
		{name: "P*s=256px", frequency: 1, q: 8, wantCount: 8},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			opt := api.DefaultOptions()
			opt.Size = size
			opt.Seed = 7
			opt.Frequency = tc.frequency
			h, err := api.Generate(opt)
			if err != nil {
				t.Fatal(err)
			}

			m := (grid - 1) / tc.q
			count := 0
			for a := -m; a <= m; a++ {
				for b := -m; b <= m; b++ {
					if a == 0 && b == 0 {
						continue
					}
					count++
					dr, dc := a*tc.q, b*tc.q
					for r := 0; r < grid; r++ {
						for c := 0; c < grid; c++ {
							tr, tc2 := r+dr, c+dc
							if tr < 0 || tr >= grid || tc2 < 0 || tc2 >= grid {
								continue
							}
							for y := 0; y < block; y++ {
								for x := 0; x < block; x++ {
									v1 := h.At(c*block+x, r*block+y)
									v2 := h.At(tc2*block+x, tr*block+y)
									if v1 != v2 {
										t.Fatalf("candidate (%+d,%+d) mismatch block (%d,%d)->(%d,%d) at %d,%d: %v != %v",
											dr, dc, r, c, tr, tc2, x, y, v1, v2)
									}
								}
							}
						}
					}
				}
			}
			if count != tc.wantCount {
				t.Fatalf("candidate count = %d, want %d", count, tc.wantCount)
			}
		})
	}
}
