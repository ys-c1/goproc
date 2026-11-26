// bandscan 逐像素扫描量化高度图，统计等高带（同灰度四连通段），
// 与按跨度/步长预判的带数逐档比对，并给出差出的带所在行号。
//
// 用法：
//
//	go run ./cmd/bandscan --size 256 --seed 1 --octaves 6 --img-octaves 4 --outdir artifacts
//
// 输出（每个位深一份）：
//
//	bands_8bit.csv / bands_16bit.csv   逐层数比对表
//	terrain_8bit.pgm / terrain_16bit.pgm  留档量化图
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"goproc/api"
	"goproc/internal/band"
	"goproc/internal/hmap"
)

func generate(size, octaves int, seed uint64) []float64 {
	h, err := api.Generate(api.Options{
		Size: size, Octaves: octaves, Lacunarity: 2, Gain: 0.5,
		Seed: seed, Frequency: 0.02,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		os.Exit(1)
	}
	return h.Data
}

func main() {
	size := flag.Int("size", 256, "边长像素数")
	seed := flag.Uint64("seed", 1, "种子")
	maxOctaves := flag.Int("octaves", 6, "比对的最大分形层数")
	imgOctaves := flag.Int("img-octaves", 4, "留档量化图所用层数")
	outdir := flag.String("outdir", "artifacts", "输出目录")
	flag.Parse()

	if err := os.MkdirAll(*outdir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir failed:", err)
		os.Exit(1)
	}

	for _, bits := range []uint{8, 16} {
		tablePath := filepath.Join(*outdir, fmt.Sprintf("bands_%dbit.csv", bits))
		f, err := os.Create(tablePath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "create failed:", err)
			os.Exit(1)
		}
		fmt.Fprintln(f, "octaves,predicted,codes_used,segments,split_excess,net_diff,extra_rows")
		fmt.Printf("== %d bit ==\n", bits)
		for k := 1; k <= *maxOctaves; k++ {
			data := generate(*size, k, *seed)
			grid := band.Quantize(data, bits)
			total, segs := band.Scan(grid, *size)

			used := make(map[uint32]int, total)
			for _, s := range segs {
				used[s.Code]++
			}
			// 同一码值最大的段视为该码的本带，其余为差出的带，收集其行号。
			extraRows := make(map[int]bool)
			split := 0
			for code, n := range used {
				if n < 2 {
					continue
				}
				split += n - 1
				best := -1
				for i, s := range segs {
					if s.Code == code && (best < 0 || s.Pixels > segs[best].Pixels) {
						best = i
					}
				}
				for i, s := range segs {
					if s.Code == code && i != best {
						for r := s.Row0; r <= s.Row1; r++ {
							extraRows[r] = true
						}
					}
				}
			}
			rows := make([]int, 0, len(extraRows))
			for r := range extraRows {
				rows = append(rows, r)
			}
			sort.Ints(rows)
			rowStrs := make([]string, len(rows))
			for i, r := range rows {
				rowStrs[i] = strconv.Itoa(r)
			}

			pred := band.Predict(1.0, bits) // 归一化后跨度恒为 1
			fmt.Fprintf(f, "%d,%d,%d,%d,%d,%d,%s\n",
				k, pred, len(used), total, split, total-pred, strings.Join(rowStrs, ";"))
			fmt.Printf("octaves=%d predicted=%d used=%d segments=%d split=+%d net=%+d extra_rows=%d\n",
				k, pred, len(used), total, split, total-pred, len(rows))
		}
		if err := f.Close(); err != nil {
			fmt.Fprintln(os.Stderr, "close failed:", err)
			os.Exit(1)
		}

		// 每个位深留一张量化图。
		data := generate(*size, *imgOctaves, *seed)
		imgPath := filepath.Join(*outdir, fmt.Sprintf("terrain_%dbit.pgm", bits))
		var img []byte
		if bits == 8 {
			img = hmap.EncodePGM(*size, data)
		} else {
			img = hmap.EncodePGM16(*size, data)
		}
		if err := os.WriteFile(imgPath, img, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write failed:", err)
			os.Exit(1)
		}
		fmt.Printf("wrote %s and %s\n", tablePath, imgPath)
	}
}
