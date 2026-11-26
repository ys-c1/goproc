package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"

	"goproc/api"
	"goproc/internal/fill"
	"goproc/internal/hmap"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: goproc <gen|fill> ...")
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "gen":
		err = genCmd(os.Args[2:])
	case "fill":
		err = fillCmd(os.Args[2:])
	default:
		err = fmt.Errorf("未知子命令 %q", os.Args[1])
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func genCmd(args []string) error {
	fs := flag.NewFlagSet("gen", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 1, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	out := fs.String("out", "map.pgm", "输出文件")
	if err := fs.Parse(args); err != nil {
		return err
	}
	h, err := api.Generate(api.Options{Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5, Seed: *seed, Frequency: 0.02})
	if err != nil {
		return fmt.Errorf("generate failed: %w", err)
	}
	if err := os.WriteFile(*out, h.ToPGM(), 0o644); err != nil {
		return fmt.Errorf("write failed: %w", err)
	}
	fmt.Printf("wrote %s size=%d\n", *out, h.Size)
	return nil
}

func parseThresholds(s string) ([]int, error) {
	parts := strings.Split(s, ",")
	if len(parts) != 2 {
		return nil, fmt.Errorf("需要两个阈值，逗号分隔: %q", s)
	}
	var ts []int
	for _, p := range parts {
		t, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil || t < 1 || t > 255 {
			return nil, fmt.Errorf("阈值越界(1..255): %q", p)
		}
		ts = append(ts, t)
	}
	sort.Ints(ts)
	return ts, nil
}

func fillCmd(args []string) error {
	fs := flag.NewFlagSet("fill", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 1, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	thresh := fs.String("thresh", "8,32", "两个水深阈值（灰度），只统计水深不低于阈值的像素")
	before := fs.String("before", "fill-before.pgm", "填洼前图")
	after := fs.String("after", "fill-after.pgm", "填洼后图")
	if err := fs.Parse(args); err != nil {
		return err
	}
	thresholds, err := parseThresholds(*thresh)
	if err != nil {
		return err
	}
	h, err := api.Generate(api.Options{Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5, Seed: *seed, Frequency: 0.02})
	if err != nil {
		return fmt.Errorf("generate failed: %w", err)
	}
	grid := &fill.Grid{Size: h.Size, G: hmap.ToGray8(h.Data)}

	// 预判侧：单遍优先泛滥求水位（每格水位即其洼地溢流高度），再按命题2 由 S 重建水面。
	wPri := fill.PriorityFlood(grid)
	lakes, err := fill.AnalyzeLakes(grid, wPri)
	if err != nil {
		return err
	}
	predMask := make([]bool, h.Size*h.Size)
	predPixels := 0
	for _, lk := range lakes {
		cells, _ := fill.PredictLake(grid, lk.Cells[0], lk.Level)
		for _, i := range cells {
			if !predMask[i] {
				predMask[i] = true
				predPixels++
			}
		}
	}

	// 对照侧：迭代松弛填洼。
	wIter := fill.IterativeFill(grid)
	iterMask := fill.WaterMask(grid, wIter)
	iterPixels := 0
	var mismatches []int
	for i := range iterMask {
		if iterMask[i] {
			iterPixels++
		}
		if iterMask[i] != predMask[i] {
			mismatches = append(mismatches, i)
		}
	}

	if err := os.WriteFile(*before, hmap.EncodePGM8(grid.Size, grid.G), 0o644); err != nil {
		return fmt.Errorf("write failed: %w", err)
	}
	if err := os.WriteFile(*after, hmap.EncodePGM8(grid.Size, wPri), 0o644); err != nil {
		return fmt.Errorf("write failed: %w", err)
	}

	fmt.Printf("填洼前图: %s\n填洼后图: %s\n", *before, *after)
	fmt.Printf("湖面数(四连通): %d\n", len(lakes))
	for _, lk := range lakes {
		_, predMin := fill.PredictLake(grid, lk.Cells[0], lk.Level)
		fmt.Printf("湖#%d: 水位S=%d 出口最小高度=%d 水面像素=%d 最大水深=%d(预判=%d)(灰度差) 出口格数=%d\n",
			lk.Label, lk.Level, lk.OutletMin, lk.Pixels, lk.MaxDepth, lk.Level-predMin, len(lk.Outlets))
	}
	fmt.Printf("预判水面像素总数=%d 迭代水面像素总数=%d 逐格不一致=%d\n", predPixels, iterPixels, len(mismatches))
	const maxReport = 32
	for i, idx := range mismatches {
		if i >= maxReport {
			fmt.Printf("  ... 其余 %d 个不一致格从略\n", len(mismatches)-maxReport)
			break
		}
		fmt.Printf("  不一致格: (%d,%d) 预判=%v 迭代=%v\n", idx%grid.Size, idx/grid.Size, predMask[idx], iterMask[idx])
	}
	for _, t := range thresholds {
		cnt, maxd := 0, 0
		for i := range wPri {
			if d := wPri[i] - grid.G[i]; d >= t {
				cnt++
				if d > maxd {
					maxd = d
				}
			}
		}
		fmt.Printf("阈值=%d: 水面像素数=%d 最大水深=%d(灰度差)\n", t, cnt, maxd)
	}
	return nil
}
