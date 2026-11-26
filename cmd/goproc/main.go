package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"goproc/api"
	"goproc/internal/hmap"
	"goproc/internal/shadow"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	switch os.Args[1] {
	case "gen":
		runGen(os.Args[2:])
	case "shadow":
		runShadow(os.Args[2:])
	default:
		usage()
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage:")
	fmt.Fprintln(os.Stderr, "  goproc gen    --size N --seed S --octaves K --out F")
	fmt.Fprintln(os.Stderr, "  goproc shadow --size N --seed S --az1 A --el1 E --az2 A --el2 E --outdir D")
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

// runShadow 对两个光照角分别跑步进法与包络扫掠法，逐格比对并输出报告与阴影图。
func runShadow(args []string) {
	fs := flag.NewFlagSet("shadow", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 7, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	zscale := fs.Float64("zscale", 16, "高度放大倍数（归一化高度乘此值再参与遮挡判定）")
	az1 := fs.Float64("az1", 45, "光照角 1 方位角（度，指向光源）")
	el1 := fs.Float64("el1", 30, "光照角 1 高度角（度）")
	az2 := fs.Float64("az2", 300, "光照角 2 方位角（度）")
	el2 := fs.Float64("el2", 15, "光照角 2 高度角（度）")
	outdir := fs.String("outdir", "artifacts", "阴影图输出目录")
	maxList := fs.Int("maxlist", 64, "不一致格明细最多打印条数")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	hm, err := api.Generate(api.Options{Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5, Seed: *seed, Frequency: 0.02})
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		os.Exit(1)
	}
	h := make([]float64, len(hm.Data))
	for i, v := range hm.Data {
		h[i] = v * *zscale
	}
	if err := os.MkdirAll(*outdir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir failed:", err)
		os.Exit(1)
	}
	lights := []shadow.Light{
		{Azimuth: *az1, Elevation: *el1},
		{Azimuth: *az2, Elevation: *el2},
	}
	for i, l := range lights {
		report(h, hm.Size, l, i+1, *outdir, *maxList)
	}
}

func report(h []float64, size int, l shadow.Light, idx int, outdir string, maxList int) {
	march := shadow.March(h, size, l)
	env := shadow.Envelope(h, size, l)
	sweep := make([]bool, size*size)
	for i, e := range env {
		sweep[i] = h[i] <= e
	}
	mis := shadow.Diff(h, env, march, sweep, size)
	nMarch, nSweep := shadow.Count(march), shadow.Count(sweep)
	rate := float64(len(mis)) / float64(size*size)

	fmt.Printf("== 光照角 %d: 方位 %.4g° 高度 %.4g° ==\n", idx, l.Azimuth, l.Elevation)
	fmt.Printf("阴影格数: 步进法 %d, 包络法 %d, 不一致 %d, 误差率 %.6f (%.4f%%)\n",
		nMarch, nSweep, len(mis), rate, rate*100)

	rm, rs := shadow.CountRows(march, size), shadow.CountRows(sweep, size)
	rd := make([]int, size)
	for _, m := range mis {
		rd[m.Y]++
	}
	fmt.Println("逐行(行号: 步进/包络/差异):")
	for y := 0; y < size; y++ {
		fmt.Printf("  y=%-4d %d/%d/%d\n", y, rm[y], rs[y], rd[y])
	}

	fmt.Println("不一致格明细 (x,y) h(p) E(p) 步进/包络:")
	for i, m := range mis {
		if i >= maxList {
			fmt.Printf("  ... 其余 %d 格从略\n", len(mis)-maxList)
			break
		}
		fmt.Printf("  (%d,%d) h=%.6f E=%.6f march=%v sweep=%v\n",
			m.X, m.Y, m.Height, m.Envelope, m.March, m.Sweep)
	}

	tag := fmt.Sprintf("az%g_el%g", l.Azimuth, l.Elevation)
	writePGM(filepath.Join(outdir, "shadow_"+tag+"_march.pgm"), size, shadow.ToMask(march))
	writePGM(filepath.Join(outdir, "shadow_"+tag+"_sweep.pgm"), size, shadow.ToMask(sweep))
	diffMask := make([]float64, size*size)
	for _, m := range mis {
		diffMask[m.Y*size+m.X] = 1
	}
	writePGM(filepath.Join(outdir, "shadow_"+tag+"_diff.pgm"), size, diffMask)
	fmt.Printf("阴影图已存档: %s/shadow_%s_{march,sweep,diff}.pgm\n\n", outdir, tag)
}

func writePGM(path string, size int, v []float64) {
	if err := os.WriteFile(path, hmap.EncodePGM(size, v), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write failed:", err)
		os.Exit(1)
	}
}
