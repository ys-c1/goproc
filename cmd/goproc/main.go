package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"goproc/api"
	"goproc/internal/report"
	"goproc/internal/tilt"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	switch os.Args[1] {
	case "gen":
		runGen(os.Args[2:])
	case "tilt":
		runTilt(os.Args[2:])
	case "overview":
		runOverview(os.Args[2:])
	default:
		usage()
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: goproc gen|tilt|overview ...")
	os.Exit(2)
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

// runOverview 只打印预判，不生成地形。
func runOverview(args []string) {
	fs := flag.NewFlagSet("overview", flag.ContinueOnError)
	freq := fs.Float64("freq", 0.02, "基础频率")
	octaves := fs.Int("octaves", 4, "层数")
	lac := fs.Float64("lacunarity", 2, "频率倍增")
	gain := fs.Float64("gain", 0.5, "振幅衰减")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	slope, deg := tilt.Overview(*freq, *octaves, *lac, *gain)
	fmt.Printf("baseFrequency=%.4f octaves=%d lacunarity=%.2f gain=%.2f\n", *freq, *octaves, *lac, *gain)
	fmt.Printf("predicted slope=%.6f  mean tilt=%.3f deg\n", slope, deg)
}

// runTilt 对两组内置参数跑预判、实测、逐档比对并存档。
func runTilt(args []string) {
	fs := flag.NewFlagSet("tilt", flag.ContinueOnError)
	size := fs.Int("size", 512, "边长")
	dir := fs.String("dir", "artifacts/tilt", "输出目录")
	reportPath := fs.String("report", "", "Markdown 报告路径，空则只写 CSV/SVG")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	sets := []struct {
		name string
		opt  api.Options
	}{
		{"A-calm", api.Options{Size: *size, Seed: 7, Octaves: 4, Lacunarity: 2, Gain: 0.5, Frequency: 0.012}},
		{"B-rugged", api.Options{Size: *size, Seed: 7, Octaves: 6, Lacunarity: 2, Gain: 0.62, Frequency: 0.05}},
	}
	var md strings.Builder
	md.WriteString("# 平均倾角预判与实测比对\n\n")
	md.WriteString("倾角按度计，实测排除边界一圈格子；预判与逐档比对均在未归一化高度场上进行。\n\n")
	for _, s := range sets {
		set, err := report.Analyze(s.name, s.opt)
		if err != nil {
			fmt.Fprintln(os.Stderr, "analyze failed:", err)
			os.Exit(1)
		}
		out := filepath.Join(*dir, s.name)
		if err := set.Write(out); err != nil {
			fmt.Fprintln(os.Stderr, "write failed:", err)
			os.Exit(1)
		}
		set.WriteMarkdown(&md)
		fmt.Printf("%s: pred %.3f deg, meas %.3f deg, worst band dev %.3f deg -> %s\n",
			s.name, set.Overview, set.Measured.MeanTilt, set.WorstAbsDev, out)
	}
	if *reportPath != "" {
		if err := os.MkdirAll(filepath.Dir(*reportPath), 0o755); err != nil {
			fmt.Fprintln(os.Stderr, "mkdir failed:", err)
			os.Exit(1)
		}
		if err := os.WriteFile(*reportPath, []byte(md.String()), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write report failed:", err)
			os.Exit(1)
		}
		fmt.Printf("report: %s\n", *reportPath)
	}
}
