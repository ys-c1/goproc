// exp 是一次性各向异性奈奎斯特采样实验命令：
// 对每个拉伸比生成朴素网格、奈奎斯特修正网格与过采样参考场，
// 原始值以 little-endian float64 落盘供 Python 侧做频谱/保留度分析。
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"goproc/api"
	"goproc/internal/sampling"
)

type gridMeta struct {
	NX int     `json:"nx"`
	NY int     `json:"ny"`
	HX float64 `json:"hx"`
	HY float64 `json:"hy"`
}

type ratioMeta struct {
	Ratio   float64  `json:"ratio"`
	RX      float64  `json:"rx"`
	RY      float64  `json:"ry"`
	Naive   gridMeta `json:"naive"`
	Fixed   gridMeta `json:"fixed"`
	Ref     gridMeta `json:"ref"`
	NaiveF  string   `json:"naive_raw"`
	FixedF  string   `json:"fixed_raw"`
	RefF    string   `json:"ref_raw"`
	NaivePG string   `json:"naive_pgm"`
	FixedPG string   `json:"fixed_pgm"`
}

type meta struct {
	Seed       uint64      `json:"seed"`
	Octaves    int         `json:"octaves"`
	Lacunarity float64     `json:"lacunarity"`
	Gain       float64     `json:"gain"`
	L          float64     `json:"domain_L"`
	FBase      float64     `json:"f_base"`
	NaiveN     int         `json:"naive_n_per_axis"`
	Ratios     []ratioMeta `json:"ratios"`
}

func writeRaw(path string, v []float64) error {
	buf := make([]byte, 8*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(x))
	}
	return os.WriteFile(path, buf, 0o644)
}

func main() {
	outDir := flag.String("out", "artifacts/exp", "输出目录")
	seed := flag.Uint64("seed", 7, "种子")
	flag.Parse()

	const (
		octaves    = 1
		lacunarity = 2.0
		gain       = 0.5
		L          = 32.0
		fBase      = 1.0 // 单层值噪声的带边频率（周/单位距离）
		naiveN     = 64   // 朴素网格：各轴等距 64 点，h=L/N=0.5=1/(2fBase)
		refOver    = 4    // 参考场相对修正网格的过采样倍数
	)
	ratios := []float64{1, 2, 4, 8}

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fatal(err)
	}
	m := meta{Seed: *seed, Octaves: octaves, Lacunarity: lacunarity, Gain: gain,
		L: L, FBase: fBase, NaiveN: naiveN}

	base := func(nx, ny int, rx, ry float64) api.RectOptions {
		return api.RectOptions{NX: nx, NY: ny, L: L, RX: rx, RY: ry, Octaves: octaves,
			Lacunarity: lacunarity, Gain: gain, Seed: *seed}
	}
	render := func(opt api.RectOptions) *api.RectHeightmap {
		h, err := api.GenerateRect(opt)
		if err != nil {
			fatal(err)
		}
		return h
	}

	for _, r := range ratios {
		rx, ry := 1.0, r
		hXN, hYN := sampling.Steps(fBase, rx, ry)
		fXN, fYN := sampling.Counts(L, hXN, hYN)
		hNaive := L / float64(naiveN)

		naive := render(base(naiveN, naiveN, rx, ry))

		fixed := render(base(fXN, fYN, rx, ry))

		refNX, refNY := refOver*fXN, refOver*fYN
		ref := render(base(refNX, refNY, rx, ry))

		rm := ratioMeta{
			Ratio: r, RX: rx, RY: ry,
			Naive:   gridMeta{naiveN, naiveN, hNaive, hNaive},
			Fixed:   gridMeta{fXN, fYN, hXN, hYN},
			Ref:     gridMeta{refNX, refNY, L / float64(refNX), L / float64(refNY)},
			NaiveF:  fmt.Sprintf("r%g_naive.f64", r),
			FixedF:  fmt.Sprintf("r%g_fixed.f64", r),
			RefF:    fmt.Sprintf("r%g_ref.f64", r),
			NaivePG: fmt.Sprintf("r%g_naive.pgm", r),
			FixedPG: fmt.Sprintf("r%g_fixed.pgm", r),
		}
		must(writeRaw(filepath.Join(*outDir, rm.NaiveF), naive.Data))
		must(writeRaw(filepath.Join(*outDir, rm.FixedF), fixed.Data))
		must(writeRaw(filepath.Join(*outDir, rm.RefF), ref.Data))
		must(os.WriteFile(filepath.Join(*outDir, rm.NaivePG), naive.ToPGM(), 0o644))
		must(os.WriteFile(filepath.Join(*outDir, rm.FixedPG), fixed.ToPGM(), 0o644))
		m.Ratios = append(m.Ratios, rm)
		fmt.Printf("r=%g naive=%dx%d fixed=%dx%d ref=%dx%d\n",
			r, naiveN, naiveN, fXN, fYN, refNX, refNY)
	}

	jb, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		fatal(err)
	}
	must(os.WriteFile(filepath.Join(*outDir, "meta.json"), jb, 0o644))
}

func must(err error) {
	if err != nil {
		fatal(err)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "exp failed:", err)
	os.Exit(1)
}
