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
		fmt.Fprintln(os.Stderr, "usage: goproc gen --size N --seed S --octaves K --out F [--raw R]")
		os.Exit(2)
	}
	fs := flag.NewFlagSet("gen", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 1, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	out := fs.String("out", "map.pgm", "输出文件")
	raw := fs.String("raw", "", "可选：浮点高度导出路径（16 字节头 + 小端 float64 高度数组）")
	if err := fs.Parse(os.Args[2:]); err != nil {
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
	if *raw != "" {
		// 头：魔数 "GORAW001"（8 字节）+ 边长 int64 + 边长 int64，随后为行优先小端 float64 高度。
		buf := make([]byte, 24+len(h.Data)*8)
		copy(buf[:8], "GORAW001")
		binary.LittleEndian.PutUint64(buf[8:16], uint64(h.Size))
		binary.LittleEndian.PutUint64(buf[16:24], uint64(h.Size))
		for i, v := range h.Data {
			binary.LittleEndian.PutUint64(buf[24+i*8:], math.Float64bits(v))
		}
		if err := os.WriteFile(*raw, buf, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "raw write failed:", err)
			os.Exit(1)
		}
		fmt.Printf("wrote %s size=%d (raw float64)\n", *raw, h.Size)
	}
}
