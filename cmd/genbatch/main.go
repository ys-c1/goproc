// genbatch 按固定两组拉伸参数批量生成 PGM，参数表写死以便复现。
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"goproc/api"
)

type caseSpec struct {
	Group string  `json:"group"`
	Name  string  `json:"name"`
	Theta float64 `json:"thetaDeg"` // 度
	Ratio float64 `json:"ratio"`
	File  string  `json:"file"`
}

func main() {
	outDir := "artifacts/pgm"
	if len(os.Args) > 1 {
		outDir = os.Args[1]
	}
	const size, seed, octaves = 512, uint64(20261127), 4
	groups := []struct {
		group  string
		theta  float64 // 度
		ratios []float64
	}{
		{"A_theta00", 0, []float64{0.5, 1.0, 1.5, 2.0, 3.0}},
		{"B_theta35", 35, []float64{0.5, 1.0, 1.5, 2.0, 3.0}},
	}
	var specs []caseSpec
	for _, g := range groups {
		for _, r := range g.ratios {
			name := fmt.Sprintf("%s_r%04.1f", g.group, r)
			file := name + ".pgm"
			opt := api.Options{
				Size: size, Octaves: octaves, Lacunarity: 2, Gain: 0.5,
				Seed: seed, Frequency: 0.02,
				StretchTheta: g.theta * math.Pi / 180,
				StretchRatio: r,
			}
			h, err := api.Generate(opt)
			if err != nil {
				fmt.Fprintln(os.Stderr, "generate failed:", err)
				os.Exit(1)
			}
			if err := os.WriteFile(filepath.Join(outDir, file), h.ToPGM(), 0o644); err != nil {
				fmt.Fprintln(os.Stderr, "write failed:", err)
				os.Exit(1)
			}
			specs = append(specs, caseSpec{g.group, name, g.theta, r, file})
		}
	}
	manifest, _ := json.MarshalIndent(specs, "", "  ")
	if err := os.WriteFile(filepath.Join(outDir, "manifest.json"), append(manifest, '\n'), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "manifest failed:", err)
		os.Exit(1)
	}
	fmt.Printf("generated %d maps in %s\n", len(specs), outDir)
}
