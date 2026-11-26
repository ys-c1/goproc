package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"goproc/api"
	"goproc/internal/hmap"
	"goproc/internal/shade"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: goproc gen ... | goproc shade ...")
		os.Exit(2)
	}
	switch os.Args[1] {
	case "gen":
		runGen(os.Args[2:])
	case "shade":
		runShade(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "未知子命令 %q\n", os.Args[1])
		os.Exit(2)
	}
}

func runGen(args []string) {
	fs := flag.NewFlagSet("gen", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 1, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	out := fs.String("out", "map.pgm", "输出文件")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	h, err := api.Generate(api.Options{Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5, Seed: *seed, Frequency: 0.02})
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*out, h.ToPGM(), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write failed:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s size=%d\n", *out, h.Size)
}

func runShade(args []string) {
	fs := flag.NewFlagSet("shade", flag.ContinueOnError)
	in := fs.String("in", "", "输入高度图 PGM(P5)")
	out := fs.String("out", "shade.pgm", "输出着色 PGM")
	phi1 := fs.Float64("phi1", 0, "光源一方位角，度")
	phi2 := fs.Float64("phi2", 90, "光源二方位角，度")
	elev := fs.Float64("elev", 15, "两灯仰角，度")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	if *in == "" {
		fmt.Fprintln(os.Stderr, "shade 需要 --in")
		os.Exit(2)
	}
	raw, err := os.ReadFile(*in)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read failed:", err)
		os.Exit(1)
	}
	size, data, err := hmap.DecodePGM(raw)
	if err != nil {
		fmt.Fprintln(os.Stderr, "decode failed:", err)
		os.Exit(1)
	}
	rad := func(d float64) float64 { return d * math.Pi / 180 }
	l1 := shade.Light{Azimuth: rad(*phi1), Elev: rad(*elev)}
	l2 := shade.Light{Azimuth: rad(*phi2), Elev: rad(*elev)}
	gray := shade.Render(size, data, l1, l2)
	if err := os.WriteFile(*out, hmap.EncodePGM(size, gray), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write failed:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s size=%d phi1=%.1f phi2=%.1f elev=%.1f\n", *out, size, *phi1, *phi2, *elev)
}
