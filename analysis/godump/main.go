// godump 复用 api.Generate 的同一条生成链路，导出：
//  1. 归一化后的浮点高度（二进制 float64 小端，长度 size*size）
//  2. 与主程序 ToPGM 完全一致的量化字节（原始 P5 负载，不含文件头）
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"goproc/api"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Fprintln(os.Stderr, "usage: godump <size> <outdir>")
		os.Exit(2)
	}
	size := 0
	if _, err := fmt.Sscanf(os.Args[1], "%d", &size); err != nil {
		fmt.Fprintln(os.Stderr, "bad size:", err)
		os.Exit(2)
	}
	outdir := os.Args[2]
	if err := os.MkdirAll(outdir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir:", err)
		os.Exit(1)
	}

	// 与 README 示例 / cmd/goproc 默认参数保持一致：
	// octaves=4 lacunarity=2 gain=0.5 frequency=0.02 seed=7
	opt := api.Options{Size: size, Octaves: 4, Lacunarity: 2, Gain: 0.5, Seed: 7, Frequency: 0.02}
	h, err := api.Generate(opt)
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate:", err)
		os.Exit(1)
	}

	floats := make([]byte, len(h.Data)*8)
	for i, v := range h.Data {
		binary.LittleEndian.PutUint64(floats[i*8:], math.Float64bits(v))
	}
	fp := filepath.Join(outdir, fmt.Sprintf("floats_%d.bin", size))
	if err := os.WriteFile(fp, floats, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write floats:", err)
		os.Exit(1)
	}

	pgm := h.ToPGM()
	pp := filepath.Join(outdir, fmt.Sprintf("map_%d.pgm", size))
	if err := os.WriteFile(pp, pgm, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write pgm:", err)
		os.Exit(1)
	}
	fmt.Printf("size=%d n=%d wrote %s %s\n", size, len(h.Data), fp, pp)
}
