package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"

	"goproc/api"
)

func main() {
	if len(os.Args) < 2 || os.Args[1] != "gen" {
		fmt.Fprintln(os.Stderr, "usage: goproc gen --size N --seed S --octaves K [--frequency F] [--format pgm|f64] --out FILE")
		os.Exit(2)
	}
	fs := flag.NewFlagSet("gen", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 1, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	frequency := fs.Float64("frequency", 0.02, "基础频率（每像素噪声坐标步长）")
	format := fs.String("format", "pgm", "输出格式：pgm（8 位灰度）或 f64（小端 float64 原始高度）")
	out := fs.String("out", "map.pgm", "输出文件")
	if err := fs.Parse(os.Args[2:]); err != nil {
		os.Exit(2)
	}
	h, err := api.Generate(api.Options{
		Size:       *size,
		Octaves:    *octaves,
		Lacunarity: 2,
		Gain:       0.5,
		Seed:       *seed,
		Frequency:  *frequency,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		os.Exit(1)
	}

	var payload []byte
	switch *format {
	case "pgm":
		payload = h.ToPGM()
	case "f64":
		payload = make([]byte, 8*len(h.Data))
		for i, v := range h.Data {
			binary.LittleEndian.PutUint64(payload[8*i:], math.Float64bits(v))
		}
	default:
		fmt.Fprintln(os.Stderr, "unknown format:", *format)
		os.Exit(2)
	}

	if err := os.WriteFile(*out, payload, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write failed:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s size=%d format=%s\n", *out, h.Size, *format)
}
