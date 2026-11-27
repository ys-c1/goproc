// hmdump 复用 api.Generate 生成高度图，导出原始 float64 与元数据，供分析管线读取。
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"goproc/api"
)

type meta struct {
	Size       int     `json:"size"`
	Octaves    int     `json:"octaves"`
	Lacunarity float64 `json:"lacunarity"`
	Gain       float64 `json:"gain"`
	Seed       uint64  `json:"seed"`
	Frequency  float64 `json:"frequency"`
	Encoding   string  `json:"encoding"`
	Endian     string  `json:"endian"`
}

func main() {
	fs := flag.NewFlagSet("hmdump", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 7, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	out := fs.String("out", "heightmap.bin", "float64 原始数据输出路径")
	if err := fs.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	opt := api.Options{Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5, Seed: *seed, Frequency: 0.02}
	h, err := api.Generate(opt)
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		os.Exit(1)
	}
	buf := make([]byte, 0, len(h.Data)*8)
	for _, v := range h.Data {
		b := make([]byte, 8)
		binary.LittleEndian.PutUint64(b, math.Float64bits(v))
		buf = append(buf, b...)
	}
	if err := os.WriteFile(*out, buf, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write failed:", err)
		os.Exit(1)
	}
	m := meta{
		Size: opt.Size, Octaves: opt.Octaves, Lacunarity: opt.Lacunarity, Gain: opt.Gain,
		Seed: opt.Seed, Frequency: opt.Frequency, Encoding: "raw-float64-le", Endian: "little",
	}
	mb, _ := json.MarshalIndent(m, "", "  ")
	metaPath := filepath.Join(filepath.Dir(*out), "heightmap.meta.json")
	if err := os.WriteFile(metaPath, append(mb, '\n'), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write meta failed:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s (%d bytes) and %s size=%d seed=%d\n", *out, len(buf), metaPath, h.Size, *seed)
}
