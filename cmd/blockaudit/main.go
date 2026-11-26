// Command blockaudit 导出“置换表周期 × 采样步长”实验所需的原始高度图。
// 它只负责通过生产 api.Generate 生成数据；候选偏移比对由 analysis/block_audit.py 独立复算。
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"goproc/api"
	"goproc/internal/hmap"
)

type specimen struct {
	Name      string
	Frequency float64 // 代码中的 Frequency：每像素对应的噪声坐标步长 δ
	StepPx    float64 // 采样步长 s=1/δ：一个噪声格点对应的像素距离
	PeriodPx  int     // P*s
	QBlocks   int     // PeriodPx / blockSize
}

const (
	size      = 512
	blockSize = 32
	seed      = 7
	octaves   = 4
)

func main() {
	dir := filepath.Join("artifacts", "blockdup")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir:", err)
		os.Exit(1)
	}

	specs := []specimen{
		{Name: "period4", Frequency: 2.0, StepPx: 0.5, PeriodPx: 128, QBlocks: 4},
		{Name: "period8", Frequency: 1.0, StepPx: 1.0, PeriodPx: 256, QBlocks: 8},
	}

	for _, sp := range specs {
		h, err := api.Generate(api.Options{
			Size:       size,
			Octaves:    octaves,
			Lacunarity: 2,
			Gain:       0.5,
			Seed:       seed,
			Frequency:  sp.Frequency,
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, "generate:", err)
			os.Exit(1)
		}

		raw := make([]byte, 8*len(h.Data))
		for i, v := range h.Data {
			binary.LittleEndian.PutUint64(raw[8*i:], math.Float64bits(v))
		}
		rawPath := filepath.Join(dir, sp.Name+".f64")
		if err := os.WriteFile(rawPath, raw, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write raw:", err)
			os.Exit(1)
		}

		pgmPath := filepath.Join(dir, sp.Name+".pgm")
		if err := os.WriteFile(pgmPath, hmap.EncodePGM(size, h.Data), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write pgm:", err)
			os.Exit(1)
		}
		fmt.Printf("wrote %s and %s (q=%d blocks)\n", rawPath, pgmPath, sp.QBlocks)
	}
}
