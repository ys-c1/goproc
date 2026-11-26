package main

import (
	"flag"
	"fmt"
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
	out := fs.String("out", "map.pgm", "输出文件")
	shared := fs.Bool("shared-perm", false, "所有层共用同一张置换表（复现重复山脊缺陷）")
	if err := fs.Parse(os.Args[2:]); err != nil {
		os.Exit(2)
	}
	h, err := api.Generate(api.Options{
		Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5,
		Seed: *seed, Frequency: 0.02, SharedPerm: *shared,
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
