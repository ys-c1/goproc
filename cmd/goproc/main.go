package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"goproc/api"
	"goproc/internal/hmap"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: goproc gen|shade [flags]")
		os.Exit(2)
	}
	switch os.Args[1] {
	case "gen":
		runGen(os.Args[2:])
	case "shade":
		runShade(os.Args[2:])
	default:
		fmt.Fprintln(os.Stderr, "usage: goproc gen|shade [flags]")
		os.Exit(2)
	}
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

// lightSpec 是三档固定光照。坐标系：X 向右、Y 向下、Z 向上；
// 方位角从 +X 起逆时针，45° 指向左上，135° 指向右上。
type lightSpec struct {
	name    string
	azimuth float64
	alt     float64
}

var lights = []lightSpec{
	{"nw", math.Pi / 4, math.Pi / 4},     // 左上前
	{"top", 0, math.Pi / 2},              // 正上
	{"ne", 3 * math.Pi / 4, math.Pi / 4}, // 右上前
}

type metadata struct {
	Size       int     `json:"size"`
	Seed       uint64  `json:"seed"`
	Octaves    int     `json:"octaves"`
	Frequency  float64 `json:"frequency"`
	Dx         float64 `json:"dx"`
	Convention string  `json:"convention"`
	BinFormat  string  `json:"bin_format"`
}

type angleStat struct {
	Name        string       `json:"name"`
	AzimuthDeg  float64      `json:"azimuth_deg"`
	AltitudeDeg float64      `json:"altitude_deg"`
	Linear      hmap.Moments `json:"linear"`
	Clipped     hmap.Moments `json:"clipped"`
}

func writeBin(path string, v []float64) error {
	buf := make([]byte, 8*len(v))
	for i, x := range v {
		// 小端写出，避免再引一个 import 块的折行
		u := math.Float64bits(x)
		for k := 0; k < 8; k++ {
			buf[i*8+k] = byte(u >> (8 * k))
		}
	}
	return os.WriteFile(path, buf, 0o644)
}

func runShade(args []string) {
	fs := flag.NewFlagSet("shade", flag.ContinueOnError)
	size := fs.Int("size", 256, "边长")
	seed := fs.Uint64("seed", 7, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	freq := fs.Float64("frequency", 0.02, "基础频率")
	dx := fs.Float64("dx", 1, "格距（像素单位；存档光照用单位像素梯度）")
	outdir := fs.String("outdir", "artifacts/shade", "输出目录")
	raw := fs.Bool("raw", false, "保留未归一化的原始 fBm（用于分辨率等价性逐点比较）")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	opt := api.Options{
		Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5,
		Seed: *seed, Frequency: *freq,
	}
	var h *api.Heightmap
	if *raw {
		data, err := api.GenerateRaw(opt)
		if err != nil {
			fmt.Fprintln(os.Stderr, "generate failed:", err)
			os.Exit(1)
		}
		h = &api.Heightmap{Size: *size, Data: data}
	} else {
		var err error
		h, err = api.Generate(opt)
		if err != nil {
			fmt.Fprintln(os.Stderr, "generate failed:", err)
			os.Exit(1)
		}
	}
	if err := os.MkdirAll(*outdir, 0o755); err != nil {
		fail(err)
	}
	md := metadata{
		Size: *size, Seed: *seed, Octaves: *octaves, Frequency: *freq, Dx: *dx,
		Convention: "图像坐标 X 右 Y 下 Z 上；方位角自 +X 逆时针，45°=左上，135°=右上；仰角 90°=正上",
		BinFormat:  "little-endian float64，行优先，长度 size*size；raw=true 时 height.bin 为未归一化 fBm",
	}
	if b, err := json.MarshalIndent(md, "", "  "); err == nil {
		_ = os.WriteFile(filepath.Join(*outdir, "metadata.json"), b, 0o644)
	}
	if !*raw {
		if err := os.WriteFile(filepath.Join(*outdir, "height.pgm"), h.ToPGM(), 0o644); err != nil {
			fail(err)
		}
	}
	if err := writeBin(filepath.Join(*outdir, "height.bin"), h.Data); err != nil {
		fail(err)
	}

	gx, gy := h.Gradients()
	// 梯度图共用一次：关于零对称映射到 0..255。
	maxAbs := 0.0
	for i := range gx {
		if a := math.Abs(gx[i]); a > maxAbs {
			maxAbs = a
		}
		if a := math.Abs(gy[i]); a > maxAbs {
			maxAbs = a
		}
	}
	toPGM := func(v []float64) []byte {
		sym := make([]float64, len(v))
		for i, x := range v {
			sym[i] = (x/maxAbs + 1) / 2
		}
		return hmap.EncodePGM(h.Size, sym)
	}
	must := func(err error) {
		if err != nil {
			fail(err)
		}
	}
	must(os.WriteFile(filepath.Join(*outdir, "gradient_x.pgm"), toPGM(gx), 0o644))
	must(os.WriteFile(filepath.Join(*outdir, "gradient_y.pgm"), toPGM(gy), 0o644))
	must(writeBin(filepath.Join(*outdir, "gradient_x.bin"), gx))
	must(writeBin(filepath.Join(*outdir, "gradient_y.bin"), gy))

	stats := make([]angleStat, 0, len(lights))
	for _, ls := range lights {
		l := hmap.Light{Azimuth: ls.azimuth, Altitude: ls.alt}
		linear := hmap.ShadeLinear(gx, gy, l)
		clipped, zeroFrac := hmap.Shade(gx, gy, l)
		dir := filepath.Join(*outdir, ls.name)
		must(os.MkdirAll(dir, 0o755))
		must(writeBin(filepath.Join(dir, "linear.bin"), linear))
		must(writeBin(filepath.Join(dir, "shaded.bin"), clipped))
		must(os.WriteFile(filepath.Join(dir, "shaded.pgm"), hmap.EncodePGM(h.Size, clipped), 0o644))

		// 每档光向对齐的梯度分量 gp = g·e_xy（水平光向单位向量），
		// 是该角度真正的“光照驱动梯度图”。正上光无水平分量，gp 全零。
		lx, ly, _ := l.Vector()
		hnorm := math.Hypot(lx, ly)
		gp := make([]float64, len(gx))
		if ls.alt < math.Pi/2-1e-9 && hnorm > 0 {
			ex, ey := lx/hnorm, ly/hnorm
			for i := range gx {
				gp[i] = gx[i]*ex + gy[i]*ey
			}
		}
		gpMax := 0.0
		for _, x := range gp {
			if a := math.Abs(x); a > gpMax {
				gpMax = a
			}
		}
		gpPGM := make([]float64, len(gp))
		for i, x := range gp {
			gpPGM[i] = (x/gpMax + 1) / 2
		}
		must(os.WriteFile(filepath.Join(dir, "gradient_parallel.pgm"), hmap.EncodePGM(h.Size, gpPGM), 0o644))
		must(writeBin(filepath.Join(dir, "gradient_parallel.bin"), gp))

		st := angleStat{
			Name:        ls.name,
			AzimuthDeg:  ls.azimuth * 180 / math.Pi,
			AltitudeDeg: ls.alt * 180 / math.Pi,
			Linear:      hmap.Stats(linear),
			Clipped:     hmap.Stats(clipped),
		}
		st.Linear.NegFrac = zeroFrac
		stats = append(stats, st)
		fmt.Printf("%s: linear skew=%.6f clipped skew=%.6f negfrac=%.4f\n",
			ls.name, st.Linear.Skew, st.Clipped.Skew, zeroFrac)
	}
	b, err := json.MarshalIndent(stats, "", "  ")
	if err != nil {
		fail(err)
	}
	must(os.WriteFile(filepath.Join(*outdir, "shading_stats.json"), b, 0o644))
	fmt.Printf("wrote %s size=%d\n", *outdir, h.Size)
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "shade failed:", err)
	os.Exit(1)
}
