// gobatch 对一组种子批量生成高度图并输出每个画幅的统计：
// 量化直方图计数、内部空隙档（1..254 中计数为零者）。
// 用法: gobatch <size> <seed0> <seed1> ...  -> 每行一个 JSON 记录到 stdout
package main

import (
	"encoding/json"
	"fmt"
	"os"

	"goproc/api"
)

type rec struct {
	Size   int   `json:"size"`
	Seed   uint64 `json:"seed"`
	N      int   `json:"n"`
	Counts []int `json:"counts"`
	Gaps   []int `json:"gaps"`
}

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: gobatch <size> <seed> [seed...]")
		os.Exit(2)
	}
	size := 0
	if _, err := fmt.Sscanf(os.Args[1], "%d", &size); err != nil {
		fmt.Fprintln(os.Stderr, "bad size:", err)
		os.Exit(2)
	}
	enc := json.NewEncoder(os.Stdout)
	for _, a := range os.Args[2:] {
		var seed uint64
		if _, err := fmt.Sscanf(a, "%d", &seed); err != nil {
			fmt.Fprintln(os.Stderr, "bad seed:", err)
			os.Exit(2)
		}
		opt := api.Options{Size: size, Octaves: 4, Lacunarity: 2, Gain: 0.5, Seed: seed, Frequency: 0.02}
		h, err := api.Generate(opt)
		if err != nil {
			fmt.Fprintln(os.Stderr, "generate:", err)
			os.Exit(1)
		}
		counts := make([]int, 256)
		for _, v := range h.Data {
			g := int(v*255 + 0.5)
			if g < 0 {
				g = 0
			}
			if g > 255 {
				g = 255
			}
			counts[g]++
		}
		var gaps []int
		for k := 1; k < 255; k++ {
			if counts[k] == 0 {
				gaps = append(gaps, k)
			}
		}
		if err := enc.Encode(rec{Size: size, Seed: seed, N: len(h.Data), Counts: counts, Gaps: gaps}); err != nil {
			fmt.Fprintln(os.Stderr, "encode:", err)
			os.Exit(1)
		}
	}
}
