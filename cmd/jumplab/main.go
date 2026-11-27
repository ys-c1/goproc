// jumplab 是插值核连续性的对照实验：
// 在三档频率下分别用三次核（C1）与五次核（C2）生成值噪声场，
// 测量格点处的二阶导跳变，并各导出一份梯度图（PGM）。
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"

	"goproc/internal/hmap"
	"goproc/internal/noise"
)

type stats struct {
	Max    float64
	RMS    float64
	Median float64
	Mean   float64
}

func summarize(v []float64) stats {
	if len(v) == 0 {
		return stats{}
	}
	tmp := append([]float64(nil), v...)
	sort.Float64s(tmp)
	var sum, sq float64
	for _, x := range tmp {
		sum += x
		sq += x * x
	}
	return stats{
		Max:    tmp[len(tmp)-1],
		RMS:    math.Sqrt(sq / float64(len(tmp))),
		Median: tmp[len(tmp)/2],
		Mean:   sum / float64(len(tmp)),
	}
}

type field struct {
	size int
	data []float64
}

func buildField(s *noise.Source, size int, freq float64) *field {
	d := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			d[y*size+x] = s.Value(float64(x)*freq, float64(y)*freq)
		}
	}
	return &field{size: size, data: d}
}

// seams 返回网格线落在像素阵列中的整数像素坐标（格点在像素 g/freq 处）。
func seams(size int, freq float64) []int {
	h := 1.0 / freq
	var out []int
	for g := 1; ; g++ {
		p := int(math.Round(float64(g) * h))
		if p >= size-2 {
			break
		}
		if p >= 2 {
			out = append(out, p)
		}
	}
	return out
}

// heightJumps 返回格线上高度场单侧二阶差分之差（像素域）：
//
//	J_pix = [f(p+2)−2f(p+1)+f(p)] − [f(p)−2f(p−1)+f(p−2)]
//	      = h²·(f''右 − f''左) + O(h³)，h=1/freq 为格点间距（像素）。
//
// 解析上 f” 的跳变等于核端点二阶导差的格点组合（见 docs/kernel-smoothness.md），
// 因此三次核的 J_pix 与 h² 成正比、五次核恒为零（实验中只剩 O(h³) 截断残差）。
func (f *field) heightJumps(freq float64, vertical bool) []float64 {
	n := f.size
	var jumps []float64
	for _, p := range seams(n, freq) {
		for q := 2; q < n-2; q++ {
			at := func(x, y int) float64 { return f.data[y*n+x] }
			var d2L, d2R float64
			if vertical { // 竖缝：对 x 求二阶导
				d2L = at(p, q) - 2*at(p-1, q) + at(p-2, q)
				d2R = at(p+2, q) - 2*at(p+1, q) + at(p, q)
			} else { // 横缝：对 y 求二阶导
				d2L = at(q, p) - 2*at(q, p-1) + at(q, p-2)
				d2R = at(q, p+2) - 2*at(q, p+1) + at(q, p)
			}
			jumps = append(jumps, d2R-d2L)
		}
	}
	return jumps
}

// gradientSecondDiffs 计算梯度分量在每个像素处的二阶差分：
// Δ²f_x(p) = f_x(p+1) − 2f_x(p) + f_x(p−1)，f_x 用前向差分。
// 若格点处存在周期跳变，格点像素（seam）的 |Δ²f_x| 均值会显著高于
// 非格点背景像素（off-seam）；C² 连续时两侧统计不可区分，比值趋于 1。
// 返回按像素分类的全部取值（竖缝方向）。
func (f *field) gradientSecondDiffs(freq float64) (seamVals, offVals []float64) {
	n := f.size
	seamSet := map[int]bool{}
	for _, p := range seams(n, freq) {
		seamSet[p] = true
	}
	for p := 2; p < n-2; p++ {
		for q := 2; q < n-2; q++ {
			fx := func(x, y int) float64 { return f.data[y*n+x+1] - f.data[y*n+x] }
			val := fx(p+1, q) - 2*fx(p, q) + fx(p-1, q)
			if seamSet[p] {
				seamVals = append(seamVals, val)
			} else {
				offVals = append(offVals, val)
			}
		}
	}
	return seamVals, offVals
}

func meanAbs(v []float64) float64 {
	var s float64
	for _, x := range v {
		s += math.Abs(x)
	}
	return s / float64(len(v))
}

// gradientMap 用中心差分计算梯度幅值，归一化到零到一。
func (f *field) gradientMap() []float64 {
	n := f.size
	out := make([]float64, n*n)
	for y := 1; y < n-1; y++ {
		for x := 1; x < n-1; x++ {
			dx := (f.data[y*n+x+1] - f.data[y*n+x-1]) * 0.5
			dy := (f.data[(y+1)*n+x] - f.data[(y-1)*n+x]) * 0.5
			out[y*n+x] = math.Hypot(dx, dy)
		}
	}
	hmap.Normalize(out)
	return out
}

func main() {
	size := flag.Int("size", 256, "边长像素")
	seed := flag.Uint64("seed", 7, "种子")
	outdir := flag.String("outdir", "artifacts/jumplab", "输出目录")
	flag.Parse()

	freqs := []float64{0.01, 0.025, 0.05}
	kernels := []noise.Kernel{noise.CubicKernel{}, noise.QuinticKernel{}}

	if err := os.MkdirAll(*outdir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	type row struct {
		freq      float64
		delta     float64 // 每格点像素数 δ=1/freq
		kernel    string
		height    stats
		gradSeam  stats
		gradOff   stats
		seamRatio float64 // 格点像素 |Δ²f_x| 均值 / 非格点背景均值
		physRMS   float64 // J_pix·δ²：三次核应随频率近似常数，五次核随 δ 趋于 0
		truncRMS  float64 // J_pix·δ³：五次核截断残差在此归一化下近似常数
	}
	var rows []row

	for fi, freq := range freqs {
		for _, k := range kernels {
			src := noise.NewWithKernel(*seed, k)
			f := buildField(src, *size, freq)

			jv := f.heightJumps(freq, true)
			jh := f.heightJumps(freq, false)
			allJ := append(append([]float64(nil), jv...), jh...)

			gSeam, gOff := f.gradientSecondDiffs(freq)
			absSeam := make([]float64, len(gSeam))
			for i, v := range gSeam {
				absSeam[i] = math.Abs(v)
			}
			mSeam, mOff := meanAbs(gSeam), meanAbs(gOff)
			delta := 1.0 / freq
			hstat := summarize(allJ)
			r := row{
				freq:      freq,
				delta:     delta,
				kernel:    k.Name(),
				height:    hstat,
				gradSeam:  summarize(absSeam),
				gradOff:   summarize(absAll(gOff)),
				seamRatio: mSeam / mOff,
				physRMS:   hstat.RMS * delta * delta,
				truncRMS:  hstat.RMS * delta * delta * delta,
			}
			rows = append(rows, r)

			// 逐缝序列（每缝对 q 取 |J| 的最大值），用于核对周期性
			seqFile := filepath.Join(*outdir, fmt.Sprintf("jumps_f%g_%s.csv", freq, k.Name()))
			w, err := os.Create(seqFile)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			fmt.Fprintln(w, "seam_pixel,max_abs_height_jump,grad_seam_second_diff_abs")
			sm := seams(*size, freq)
			idx := 0
			for _, p := range sm {
				var mH, mG float64
				for q := 2; q < *size-2; q++ {
					if v := math.Abs(allJ[idx]); v > mH {
						mH = v
					}
					if v := math.Abs(gSeam[idx]); v > mG {
						mG = v
					}
					idx++
				}
				fmt.Fprintf(w, "%d,%.6f,%.6f\n", p, mH, mG)
			}
			w.Close()

			// 中间频率各存一份梯度图
			if fi == len(freqs)/2 {
				gm := f.gradientMap()
				pgm := filepath.Join(*outdir, fmt.Sprintf("gradient_%s_f%g.pgm", k.Name(), freq))
				if err := os.WriteFile(pgm, hmap.EncodePGM(*size, gm), 0o644); err != nil {
					fmt.Fprintln(os.Stderr, err)
					os.Exit(1)
				}
			}
		}
	}

	// 机器可读汇总
	mf, err := os.Create(filepath.Join(*outdir, "jump_summary.txt"))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer mf.Close()
	fmt.Fprintln(mf, "frequency,delta_px,kernel,height_jump_max,height_jump_rms,phys_jump_rms_Jpix_d2,trunc_jump_rms_Jpix_d3,grad_seam_abs_max,grad_seam_mean_abs,grad_off_mean_abs,seam_to_off_ratio")
	for _, r := range rows {
		fmt.Fprintf(mf, "%g,%g,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.4f\n",
			r.freq, r.delta, r.kernel, r.height.Max, r.height.RMS, r.physRMS, r.truncRMS,
			r.gradSeam.Max, r.gradSeam.Mean, r.gradOff.Mean, r.seamRatio)
	}

	fmt.Println("freq  | kernel  | Jpix rms | Jpix·δ²(物理跳变) | Jpix·δ³(截断律) | 格点/背景比")
	for _, r := range rows {
		fmt.Printf("%-5g | %-7s | %.2e | %16.4f | %14.4f | %7.3f\n",
			r.freq, r.kernel, r.height.RMS, r.physRMS, r.truncRMS, r.seamRatio)
	}
	fmt.Println("输出目录:", *outdir)
}

func absAll(v []float64) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = math.Abs(x)
	}
	return out
}
