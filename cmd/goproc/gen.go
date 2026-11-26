package main

import (
	"flag"
	"fmt"
	"os"

	"goproc/api"
)

func runGen(args []string) int {
	fs := flag.NewFlagSet("gen", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 1, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	out := fs.String("out", "map.pgm", "输出文件")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	h, err := api.Generate(api.Options{
		Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5,
		Seed: *seed, Frequency: 0.02,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		return 1
	}
	if err := os.WriteFile(*out, h.ToPGM(), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write failed:", err)
		return 1
	}
	fmt.Printf("wrote %s size=%d\n", *out, h.Size)
	return 0
}
