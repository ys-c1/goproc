package main

import (
	"flag"
	"fmt"
	"os"

	"goproc/api"
	"goproc/internal/seam"
)

func main() {
	if len(os.Args) < 2 || os.Args[1] != "gen" {
		fmt.Fprintln(os.Stderr, "usage: goproc gen --size N --seed S --octaves K --out F [--cells C --period P]")
		os.Exit(2)
	}
	fs := flag.NewFlagSet("gen", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 1, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	out := fs.String("out", "map.pgm", "输出文件")
	cells := fs.Int("cells", 0, "平铺时第零层横跨的格点数，0 表示按 --freq 生成")
	period := fs.Int("period", 0, "置换表周期（格点），0 表示 256")
	freq := fs.Float64("freq", 0.02, "基础频率")
	repeat := fs.Int("repeat", 0, "拼接重数，0 表示只导出单图")
	if err := fs.Parse(os.Args[2:]); err != nil {
		os.Exit(2)
	}
	opt := api.Options{
		Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5,
		Seed: *seed, Frequency: *freq, TileCells: *cells, Period: *period,
	}
	h, err := api.Generate(opt)
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		os.Exit(1)
	}
	data := h.ToPGM()
	if *repeat > 0 {
		data = h.ToPGMRepeat(*repeat)
	}
	if err := os.WriteFile(*out, data, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write failed:", err)
		os.Exit(1)
	}
	m := seam.Measure(h)
	fmt.Printf("wrote %s size=%d tileable=%v seam_jump=%.3e (<%.3e) fault=%v\n",
		*out, h.Size, h.Options().Tileable(), m.JumpMax(), seam.GrayStep, m.Fault())
}
