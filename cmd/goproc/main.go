package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"goproc/api"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "gen":
		runGen(os.Args[2:])
	case "erode":
		runErode(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage:")
	fmt.Fprintln(os.Stderr, "  goproc gen --size N --seed S --octaves K --out F")
	fmt.Fprintln(os.Stderr, "  goproc erode --size N --seed S --octaves K --steps 10,50,200 --dir D")
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

func runErode(args []string) {
	fs := flag.NewFlagSet("erode", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 1, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	stepsArg := fs.String("steps", "10,50,200", "侵蚀迭代步数清单，逗号分隔")
	dir := fs.String("dir", "artifacts/erosion", "侵蚀图与报告存档目录")
	k := fs.Float64("k", 0.05, "通量系数")
	cell := fs.Float64("cellsize", 1.0, "格距 d")
	quantum := fs.Float64("quantum", 1e-4, "定点量化步长 q（仅定点版）")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	stepList, err := parseSteps(*stepsArg)
	if err != nil {
		fmt.Fprintln(os.Stderr, "bad --steps:", err)
		os.Exit(2)
	}
	h, err := api.Generate(api.Options{Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5, Seed: *seed, Frequency: 0.02})
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		os.Exit(1)
	}
	cfg := api.ErosionConfig{K: *k, CellSize: *cell, Quantum: *quantum}
	report, err := h.ConservationStudy(stepList, cfg, *dir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "study failed:", err)
		os.Exit(1)
	}
	printReport(report, *dir)
}

func parseSteps(s string) ([]int, error) {
	parts := strings.Split(s, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("非法步数 %q", p)
		}
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("至少需要一个步数")
	}
	return out, nil
}

func printReport(r *api.StudyReport, dir string) {
	fmt.Printf("初始总高度 H0 = %.6f，格距 d = %v，K = %v，q = %v，阈值 = %g\n",
		r.InitialTotal, r.CellSize, r.K, r.Quantum, r.Threshold)
	fmt.Println()
	fmt.Printf("%-8s %-10s %-14s %-12s %-18s %-8s\n", "版本", "步数", "末态总高度", "归一高差", "首次偏差步(绝对量)", "判定")
	for i, steps := range r.Steps {
		printRow("fixed", steps, r.Fixed.Results[i])
		printRow("naive", steps, r.Naive.Results[i])
	}
	fmt.Println()
	fmt.Printf("fixed 三个步数全部达标: %v\n", r.AllFixedPass)
	fmt.Printf("naive 三个步数全部达标: %v\n", r.AllNaivePass)
	fmt.Printf("侵蚀图与 report.json 已存档至 %s\n", dir)
}

func printRow(name string, steps int, res api.VariantResultView) {
	first := "无"
	if res.FirstDeviationStep != 0 {
		first = fmt.Sprintf("第 %d 步 (%.3e)", res.FirstDeviationStep, res.FirstDeviationAbs)
	}
	verdict := "PASS"
	if !res.Pass {
		verdict = "FAIL"
	}
	fmt.Printf("%-8s %-10d %-14.6f %-12.3e %-18s %-8s\n",
		name, steps, res.FinalTotal, res.RelDiff, first, verdict)
}
