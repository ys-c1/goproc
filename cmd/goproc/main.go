package main

import (
	"flag"
	"fmt"
	"os"

	"goproc/api"
	"goproc/internal/analyze"
	"goproc/internal/normal"
	"goproc/internal/viz"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "gen":
		runGen(os.Args[2:])
	case "normal":
		runNormal(os.Args[2:])
	case "analyze":
		runAnalyze(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage:")
	fmt.Fprintln(os.Stderr, "  goproc gen     [flags] --out map.pgm")
	fmt.Fprintln(os.Stderr, "  goproc normal  [flags] --outdir DIR   (输出 normal_map.png 与 azimuth_map.png)")
	fmt.Fprintln(os.Stderr, "  goproc analyze [flags] --outdir DIR   (占比表、汇总 JSON 与两张可视化)")
}

func commonFlags(fs *flag.FlagSet) (size *int, seed *uint64, octaves *int) {
	size = fs.Int("size", 128, "边长")
	seed = fs.Uint64("seed", 1, "种子")
	octaves = fs.Int("octaves", 4, "层数")
	return size, seed, octaves
}

func optionsFrom(size int, seed uint64, octaves int) api.Options {
	return api.Options{Size: size, Octaves: octaves, Lacunarity: 2, Gain: 0.5, Seed: seed, Frequency: 0.02}
}

func runGen(args []string) {
	fs := flag.NewFlagSet("gen", flag.ContinueOnError)
	size, seed, octaves := commonFlags(fs)
	out := fs.String("out", "map.pgm", "输出文件")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	h, err := api.Generate(optionsFrom(*size, *seed, *octaves))
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

func runNormal(args []string) {
	fs := flag.NewFlagSet("normal", flag.ContinueOnError)
	size, seed, octaves := commonFlags(fs)
	outdir := fs.String("outdir", "artifacts", "输出目录")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	h, err := api.Generate(optionsFrom(*size, *seed, *octaves))
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		os.Exit(1)
	}
	field := normal.FromHeightmap(h.Size, h.Data)
	if err := writePNG(*outdir, "normal_map.png", func(f *os.File) error {
		return viz.WriteNormalMap(f, field)
	}); err != nil {
		fatal(err)
	}
	if err := writePNG(*outdir, "azimuth_map.png", func(f *os.File) error {
		return viz.WriteAzimuthMap(f, field)
	}); err != nil {
		fatal(err)
	}
	fmt.Printf("wrote normal maps to %s size=%d\n", *outdir, h.Size)
}

func runAnalyze(args []string) {
	fs := flag.NewFlagSet("analyze", flag.ContinueOnError)
	size, seed, octaves := commonFlags(fs)
	outdir := fs.String("outdir", "artifacts", "输出目录")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	opt := optionsFrom(*size, *seed, *octaves)
	rep, err := analyze.Run(opt, *outdir)
	if err != nil {
		fatal(err)
	}
	// 两张可视化与表格同目录。
	h, err := api.Generate(opt)
	if err != nil {
		fatal(err)
	}
	field := normal.FromHeightmap(h.Size, h.Data)
	if err := writePNG(*outdir, "normal_map.png", func(f *os.File) error {
		return viz.WriteNormalMap(f, field)
	}); err != nil {
		fatal(err)
	}
	if err := writePNG(*outdir, "azimuth_map.png", func(f *os.File) error {
		return viz.WriteAzimuthMap(f, field)
	}); err != nil {
		fatal(err)
	}
	fmt.Printf("analysis complete: beta=%.3f R2=%.4f flagged_spec_8=%v flagged_iso_8=%v\n",
		rep.Beta.Beta, rep.Beta.RSquared, rep.FlaggedSpec8, rep.FlaggedIso8)
}

func writePNG(dir, name string, encode func(*os.File) error) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	f, err := os.Create(dir + "/" + name)
	if err != nil {
		return err
	}
	defer f.Close()
	return encode(f)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, err)
	os.Exit(1)
}
