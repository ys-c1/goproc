// Command dumpfloats writes normalized float64 heightmaps for a seed list.
// It is an analysis helper; the terrain values come directly from api.Generate.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"goproc/api"
)

func parseSeeds(spec string) ([]uint64, error) {
	var seeds []uint64
	for _, part := range strings.Split(spec, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if strings.Contains(part, "-") {
			ends := strings.SplitN(part, "-", 2)
			lo, err := strconv.ParseUint(strings.TrimSpace(ends[0]), 10, 64)
			if err != nil {
				return nil, err
			}
			hi, err := strconv.ParseUint(strings.TrimSpace(ends[1]), 10, 64)
			if err != nil {
				return nil, err
			}
			if hi < lo {
				return nil, fmt.Errorf("bad seed range %s", part)
			}
			for s := lo; s <= hi; s++ {
				seeds = append(seeds, s)
			}
		} else {
			s, err := strconv.ParseUint(part, 10, 64)
			if err != nil {
				return nil, err
			}
			seeds = append(seeds, s)
		}
	}
	return seeds, nil
}

func main() {
	size := flag.Int("size", 128, "边长")
	octaves := flag.Int("octaves", 4, "层数")
	frequency := flag.Float64("frequency", 0.02, "基础频率")
	gain := flag.Float64("gain", 0.5, "振幅衰减")
	lacunarity := flag.Float64("lacunarity", 2, "频率倍增")
	seedSpec := flag.String("seeds", "1-60", "种子列表，可写 1-30,41,42")
	outDir := flag.String("out", "work/go_floats", "输出目录")
	flag.Parse()

	seeds, err := parseSeeds(*seedSpec)
	if err != nil {
		fmt.Fprintln(os.Stderr, "bad seeds:", err)
		os.Exit(2)
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir failed:", err)
		os.Exit(1)
	}

	var manifest strings.Builder
	manifest.WriteString("seed,size,octaves,frequency,gain,lacunarity,file\n")
	for _, seed := range seeds {
		h, err := api.Generate(api.Options{
			Size:       *size,
			Octaves:    *octaves,
			Lacunarity: *lacunarity,
			Gain:       *gain,
			Seed:       seed,
			Frequency:  *frequency,
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, "generate failed:", err)
			os.Exit(1)
		}
		name := fmt.Sprintf("seed_%04d.f64", seed)
		path := filepath.Join(*outDir, name)
		buf := make([]byte, 8*len(h.Data))
		for i, v := range h.Data {
			binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
		}
		if err := os.WriteFile(path, buf, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write failed:", err)
			os.Exit(1)
		}
		manifest.WriteString(fmt.Sprintf("%d,%d,%d,%g,%g,%g,%s\n", seed, h.Size, *octaves, *frequency, *gain, *lacunarity, name))
	}
	if err := os.WriteFile(filepath.Join(*outDir, "manifest.csv"), []byte(manifest.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "manifest failed:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %d float64 heightmaps to %s\n", len(seeds), *outDir)
}
