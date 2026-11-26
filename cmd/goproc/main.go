package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"goproc/api"
	"goproc/internal/divide"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "gen":
		runGen(os.Args[2:])
	case "divide":
		runDivide(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage:")
	fmt.Fprintln(os.Stderr, "  goproc gen    --size N --seed S --octaves K --out F")
	fmt.Fprintln(os.Stderr, "  goproc divide --size N --seed S --octaves K --freq F --scale M --f64 F --prefix P")
}

func commonFlags(fs *flag.FlagSet) (size *int, seed *uint64, octaves *int, freq *float64) {
	size = fs.Int("size", 128, "边长")
	seed = fs.Uint64("seed", 1, "种子")
	octaves = fs.Int("octaves", 4, "分形层数")
	freq = fs.Float64("freq", 0.02, "基础频率")
	return size, seed, octaves, freq
}

func runGen(args []string) {
	fs := flag.NewFlagSet("gen", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 1, "种子")
	octaves := fs.Int("octaves", 4, "分形层数")
	out := fs.String("out", "map.pgm", "输出文件")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	h, err := api.Generate(api.Options{Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5, Seed: *seed, Frequency: 0.02})
	if err != nil {
		fail("generate failed:", err)
	}
	if err := os.WriteFile(*out, h.ToPGM(), 0o644); err != nil {
		fail("write failed:", err)
	}
	fmt.Printf("wrote %s size=%d\n", *out, h.Size)
}

func runDivide(args []string) {
	fs := flag.NewFlagSet("divide", flag.ContinueOnError)
	size, seed, octaves, freq := commonFlags(fs)
	scale := fs.Int("scale", 5, "PNG 放大倍数")
	f64Path := fs.String("f64", "", "高度图 f64 输出路径")
	prefix := fs.String("prefix", "divide", "产物前缀（divide.txt / gray.png / overlay.png / crest.txt）")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}

	h, err := api.Generate(api.Options{Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5, Seed: *seed, Frequency: *freq})
	if err != nil {
		fail("generate failed:", err)
	}
	dir, horiz, vert, oppH, oppV, sinkCount, err := divide.Compute(h.Size, h.Data)
	if err != nil {
		fail("divide failed:", err)
	}
	crest := divide.Crest(h.Size, h.Data)
	segs := divide.Trace(append(append([]divide.Edge{}, horiz...), vert...))

	if *f64Path != "" {
		writeFile(*f64Path, divide.EncodeF64(h.Size, h.Data))
	}
	writeFile(*prefix+"_divide.txt", encodeEdges(h.Size, horiz, vert))
	writeFile(*prefix+"_opposite.txt", encodeEdges(h.Size, oppH, oppV))
	writeFile(*prefix+"_gray.png", divide.RenderGray(h.Size, h.Data, *scale))
	writeFile(*prefix+"_overlay.png", divide.RenderOverlay(h.Size, h.Data, horiz, vert, *scale))
	writeFile(*prefix+"_crest.txt", encodeCrest(crest))

	fmt.Printf("wrote %s* : 汇=%d，分水线 H=%d V=%d 合计=%d，连通线段=%d；严格反向边 H=%d V=%d；对照脊线像素=%d\n",
		*prefix, sinkCount, len(horiz), len(vert), len(horiz)+len(vert), len(segs),
		len(oppH), len(oppV), len(crest))
	_ = dir
}

func encodeEdges(n int, horiz, vert []divide.Edge) []byte {
	var b strings.Builder
	fmt.Fprintf(&b, "# kind x y  size=%d H=%d V=%d\n", n, len(horiz), len(vert))
	for _, e := range horiz {
		fmt.Fprintf(&b, "H %d %d\n", e.X, e.Y)
	}
	for _, e := range vert {
		fmt.Fprintf(&b, "V %d %d\n", e.X, e.Y)
	}
	return []byte(b.String())
}

func encodeCrest(pts []divide.Point) []byte {
	var b strings.Builder
	b.WriteString("# x y\n")
	for _, p := range pts {
		fmt.Fprintf(&b, "%d %d\n", p.X, p.Y)
	}
	return []byte(b.String())
}

func writeFile(path string, data []byte) {
	if err := os.WriteFile(path, data, 0o644); err != nil {
		fail("write failed:", err)
	}
}

func fail(msg string, err error) {
	fmt.Fprintln(os.Stderr, msg, err)
	os.Exit(1)
}
