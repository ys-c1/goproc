// dumpraw 导出归一化后的原始浮点高度图（小端 float64 连续存储），
// 与 PGM 的八位取整区分开，供量化/锯齿分析使用。
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
	fs := flag.NewFlagSet("dumpraw", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 1, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	out := fs.String("out", "raw.bin", "输出文件")
	if err := fs.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	h, err := api.Generate(api.Options{Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5, Seed: *seed, Frequency: 0.02})
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		os.Exit(1)
	}
	buf := make([]byte, 8*len(h.Data))
	for i, v := range h.Data {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(v))
	}
	if err := os.WriteFile(*out, buf, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write failed:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s size=%d floats=%d\n", *out, h.Size, len(h.Data))
}
