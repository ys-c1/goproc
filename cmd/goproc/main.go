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
		fmt.Fprintln(os.Stderr, "usage: goproc gen --size N --seed S --octaves K [--frequency f] [--gain g] [--lacunarity l] [--format pgm|f64] [--raw] --out F")
		os.Exit(2)
	}
	fs := flag.NewFlagSet("gen", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 1, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	frequency := fs.Float64("frequency", api.DefaultOptions().Frequency, "基础频率（周/像素口径前的噪声坐标系数）")
	gain := fs.Float64("gain", api.DefaultOptions().Gain, "振幅衰减/增益")
	lacunarity := fs.Float64("lacunarity", api.DefaultOptions().Lacunarity, "频率倍增")
	format := fs.String("format", "pgm", "输出格式：pgm（8 位灰度）或 f64（小端 float64）")
	raw := fs.Bool("raw", false, "保留 fBm 原值，跳过 min-max 归一化（f64 推荐）")
	out := fs.String("out", "map.pgm", "输出文件")
	if err := fs.Parse(os.Args[2:]); err != nil {
		os.Exit(2)
	}
	h, err := api.Generate(api.Options{
		Size:       *size,
		Octaves:    *octaves,
		Lacunarity: *lacunarity,
		Gain:       *gain,
		Seed:       *seed,
		Frequency:  *frequency,
		Raw:        *raw,
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
		payload = encodeF64(h.Data)
	default:
		fmt.Fprintln(os.Stderr, "unknown --format:", *format)
		os.Exit(2)
	}
	if err := os.WriteFile(*out, payload, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write failed:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s size=%d format=%s raw=%v\n", *out, h.Size, *format, *raw)
}

// encodeF64 把高度样本编码为小端 IEEE-754 float64 连续数组（行优先）。
func encodeF64(v []float64) []byte {
	buf := make([]byte, 8*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint64(buf[8*i:], math.Float64bits(x))
	}
	return buf
}
