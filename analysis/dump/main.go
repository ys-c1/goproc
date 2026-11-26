package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"goproc/internal/fbm"
	"goproc/internal/hmap"
	"goproc/internal/noise"
)

type rangeRow struct {
	seed   uint64
	gain   float64
	octave int
	lo     float64
	hi     float64
}

func main() {
	outDir := flag.String("out", "artifacts/boxcount/maps", "PGM output directory")
	size := flag.Int("size", 512, "heightmap side length")
	seedsFlag := flag.String("seeds", "7", "comma-separated deterministic seeds")
	flag.Parse()

	gains := []float64{0.40, 0.50, 0.60}
	var seeds []uint64
	for _, part := range strings.Split(*seedsFlag, ",") {
		v, err := strconv.ParseUint(strings.TrimSpace(part), 10, 64)
		if err != nil {
			fmt.Fprintln(os.Stderr, "bad seed:", part)
			os.Exit(2)
		}
		seeds = append(seeds, v)
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	var rows []rangeRow
	for _, seed := range seeds {
		src := noise.New(seed)
		for gi, gain := range gains {
			for octaves := 1; octaves <= 12; octaves++ {
				data := make([]float64, (*size)*(*size))
				lo, hi := 1.0, -1.0
				for y := 0; y < *size; y++ {
					for x := 0; x < *size; x++ {
						v := fbm.Eval(src, float64(x)*0.02, float64(y)*0.02, octaves, 2.0, gain)
						data[y*(*size)+x] = v
						if v < lo {
							lo = v
						}
						if v > hi {
							hi = v
						}
					}
				}
				hmap.Normalize(data)
				pgm := hmap.EncodePGM(*size, data)
				name := fmt.Sprintf("s%03d_g%d_o%02d.pgm", seed, gi, octaves)
				if err := os.WriteFile(filepath.Join(*outDir, name), pgm, 0o644); err != nil {
					fmt.Fprintln(os.Stderr, err)
					os.Exit(1)
				}
				rows = append(rows, rangeRow{seed, gain, octaves, lo, hi})
			}
		}
	}

	f, err := os.Create(filepath.Join(*outDir, "ranges.csv"))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	_ = w.Write([]string{"seed", "gain", "octaves", "raw_lo", "raw_hi", "raw_range"})
	for _, r := range rows {
		_ = w.Write([]string{
			strconv.FormatUint(r.seed, 10),
			strconv.FormatFloat(r.gain, 'f', 2, 64),
			strconv.Itoa(r.octave),
			strconv.FormatFloat(r.lo, 'g', 17, 64),
			strconv.FormatFloat(r.hi, 'g', 17, 64),
			strconv.FormatFloat(r.hi-r.lo, 'g', 17, 64),
		})
	}
	w.Flush()
	if err := w.Error(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
