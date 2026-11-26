package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"

	"goproc/api"
)

func usage() {
	fmt.Fprintln(os.Stderr, "usage:")
	fmt.Fprintln(os.Stderr, "  goproc gen  --size N --seed S --octaves K --out F")
	fmt.Fprintln(os.Stderr, "  goproc raw  --size N --seed S --octaves K --gain G --lacunarity L --freq F0 --out F   (F=二进制 float64 裸数据，另写 F.json 参数侧车)")
	os.Exit(2)
}

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	switch os.Args[1] {
	case "gen":
		runGen(os.Args[2:])
	case "raw":
		runRaw(os.Args[2:])
	default:
		usage()
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

func runRaw(args []string) {
	fs := flag.NewFlagSet("raw", flag.ContinueOnError)
	size := fs.Int("size", 1024, "边长")
	seed := fs.Uint64("seed", 1, "种子")
	octaves := fs.Int("octaves", 6, "层数")
	gain := fs.Float64("gain", 0.5, "振幅衰减")
	lac := fs.Float64("lacunarity", 2.0, "频率倍增")
	freq := fs.Float64("freq", 0.005, "基础频率（每格）")
	out := fs.String("out", "map.f64", "输出文件（小端 float64，行优先）")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	opt := api.Options{Size: *size, Octaves: *octaves, Lacunarity: *lac, Gain: *gain, Seed: *seed, Frequency: *freq}
	h, err := api.Generate(opt)
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		os.Exit(1)
	}
	buf := make([]byte, 0, len(h.Data)*8)
	var tmp [8]byte
	for _, v := range h.Data {
		binary.LittleEndian.PutUint64(tmp[:], math.Float64bits(v))
		buf = append(buf, tmp[:]...)
	}
	if err := os.WriteFile(*out, buf, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write failed:", err)
		os.Exit(1)
	}
	meta := map[string]any{
		"file":        *out,
		"size":        h.Size,
		"seed":        *seed,
		"octaves":     *octaves,
		"gain":        *gain,
		"lacunarity":  *lac,
		"frequency":   *freq,
		"dtype":       "float64le",
		"layout":      "row-major",
		"normalized":  "[0,1]",
	}
	mb, _ := json.MarshalIndent(meta, "", "  ")
	if err := os.WriteFile(*out+".json", append(mb, '\n'), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write meta failed:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s size=%d octaves=%d gain=%g freq=%g\n", *out, h.Size, *octaves, *gain, *freq)
}
