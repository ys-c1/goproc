// 别名分析专用导出器：复用 internal/noise 的真实实现，输出裸 float64 二进制。
// 不属于产品代码，只服务 analysis/ 下的实验。
//
// 空间约定：以“细格”为统一空间单位（细格距 = 1）。
//   细网格在 x_f=0..Nf-1 采样，噪声坐标 = x_f * fFine（fFine 单位：周期/细格）
//   粗网格在 x_c=0..Nc-1 采样，对应细坐标 x_f = x_c*Δ，噪声坐标 = x_c*Δ*fFine
// 两者覆盖同一段连续场，粗网格是细网格每 Δ 点取一个的降采样。
package main

import (
	"encoding/binary"
	"flag"
	"math"
	"os"

	"goproc/internal/fbm"
	"goproc/internal/noise"
)

func writeRaw(path string, v []float64) error {
	b := make([]byte, 8*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint64(b[i*8:], math.Float64bits(x))
	}
	return os.WriteFile(path, b, 0o644)
}

func main() {
	mode := flag.String("mode", "single", "single=单倍频2D场; fbm=分形2D场; line=1D长序列")
	seed := flag.Uint64("seed", 7, "种子")
	nf := flag.Int("nf", 1024, "细网格边长（细格距 1）")
	nc := flag.Int("nc", 128, "粗网格边长")
	step := flag.Int("step", 8, "粗格距 Δ（细格单位）")
	ffine := flag.Float64("ffine", 0.5, "细网格单位下的噪声基频（周期/细格）")
	outFine := flag.String("fine", "fine.raw", "细网格输出")
	outCoarse := flag.String("coarse", "coarse.raw", "粗网格输出")
	octaves := flag.Int("octaves", 1, "fbm 模式层数")
	lac := flag.Float64("lac", 2.0, "lacunarity")
	gain := flag.Float64("gain", 0.5, "gain")
	nlines := flag.Int("nlines", 64, "line 模式独立序列条数")
	flag.Parse()

	src := noise.New(*seed)

	if *mode == "line" {
		// 1D：nlines 条独立长序列（不同 y 格线），每条细网格 nf 点，
		// 粗网格 nf/Δ 点。固定 y 取整数格线，保证 1D 切片是合法值噪声。
		ncf := *nf / *step
		fine := make([]float64, *nlines**nf)
		coarse := make([]float64, *nlines*ncf)
		df := *ffine * float64(*step)
		for ln := 0; ln < *nlines; ln++ {
			y := float64(ln * 7) // 整数格线，perm 给出独立 1D 序列
			for x := 0; x < *nf; x++ {
				fine[ln**nf+x] = src.Value(float64(x)**ffine, y)
			}
			for xc := 0; xc < ncf; xc++ {
				coarse[ln*ncf+xc] = src.Value(float64(xc)*df, y)
			}
		}
		if err := writeRaw(*outFine, fine); err != nil {
			panic(err)
		}
		if err := writeRaw(*outCoarse, coarse); err != nil {
			panic(err)
		}
		return
	}

	fine2 := make([]float64, *nf**nf)
	for y := 0; y < *nf; y++ {
		for x := 0; x < *nf; x++ {
			xf, yf := float64(x)**ffine, float64(y)**ffine
			if *mode == "fbm" {
				fine2[y**nf+x] = fbm.Eval(src, xf, yf, *octaves, *lac, *gain)
			} else {
				fine2[y**nf+x] = src.Value(xf, yf)
			}
		}
	}
	if err := writeRaw(*outFine, fine2); err != nil {
		panic(err)
	}

	// 粗网格：同一连续场，坐标放大 Δ，频率等价为 ffine*Δ（周期/粗像素）。
	coarse := make([]float64, *nc**nc)
	df := *ffine * float64(*step)
	for y := 0; y < *nc; y++ {
		for x := 0; x < *nc; x++ {
			xc, yc := float64(x)*df, float64(y)*df
			if *mode == "fbm" {
				coarse[y**nc+x] = fbm.Eval(src, xc, yc, *octaves, *lac, *gain)
			} else {
				coarse[y**nc+x] = src.Value(xc, yc)
			}
		}
	}
	if err := writeRaw(*outCoarse, coarse); err != nil {
		panic(err)
	}
}
