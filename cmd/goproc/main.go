package main

import (
	"flag"
	"fmt"
	"math"
	"os"

	"goproc/api"
)

func main() {
	if len(os.Args) < 2 || os.Args[1] != "gen" {
		fmt.Fprintln(os.Stderr, "usage: goproc gen --size N --seed S --octaves K --out F")
		os.Exit(2)
	}
	fs := flag.NewFlagSet("gen", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 1, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	theta := fs.Float64("theta", 0, "拉伸方向角（度），正东为零逆时针为正")
	ratio := fs.Float64("ratio", 0, "拉伸轴比（剪切参数），零为不拉伸")
	out := fs.String("out", "map.pgm", "输出文件")
	if err := fs.Parse(os.Args[2:]); err != nil {
		os.Exit(2)
	}
	h, err := api.Generate(api.Options{
		Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5, Seed: *seed,
		Frequency:    0.02,
		StretchTheta: (*theta) * math.Pi / 180,
		StretchRatio: *ratio,
	})
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
