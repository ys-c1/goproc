// 命令 quantile 按 docs/height-quantile.md 的显式预判分位高度，并与实测逐档比对。
// 输出：stdout 报告、quantiles.tsv、cdf_predicted.svg、cdf_empirical.svg（分别存档）。
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"goproc/internal/fbm"
	"goproc/internal/noise"
)

var quantiles = []float64{0.10, 0.50, 0.90}

// fade 与 internal/noise 中的五次缓和曲线相同，这里为积分常数重实现。
func fade(t float64) float64 { return t * t * t * (t*(t*6-15) + 10) }

// varH 是格点值的精确方差：值表为 0..255 的置换，h = k/127.5 - 1。
func varH() float64 {
	s := 0.0
	for k := 0; k < 256; k++ {
		h := float64(k)/127.5 - 1
		s += h * h
	}
	return s / 256
}

// eFade2 数值积分 ∫₀¹ fade² dt（精确值 181/462，这里用数值积分避免魔法数）。
func eFade2() float64 {
	const n = 1 << 20
	dt := 1.0 / n
	s := 0.0
	for i := 0; i < n; i++ {
		f := fade((float64(i) + 0.5) * dt)
		s += f * f
	}
	return s * dt
}

// normInv 是标准正态分位数函数，Acklam 有理逼近，误差约 1e-9。
func normInv(p float64) float64 {
	a := [...]float64{-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00}
	b := [...]float64{-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01}
	c := [...]float64{-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00}
	d := [...]float64{7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00}
	const plow, phigh = 0.02425, 1 - 0.02425
	if p < plow {
		q := math.Sqrt(-2 * math.Log(p))
		return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q + c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)
	}
	if p > phigh {
		q := math.Sqrt(-2 * math.Log(1-p))
		return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q + c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q + 1)
	}
	q := p - 0.5
	r := q * q
	return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r + a[5]) * q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r + 1)
}

func normCDF(x float64) float64 { return 0.5 * (1 + math.Erf(x/math.Sqrt2)) }

// empQuantile 是 type-7 经验分位。
func empQuantile(sorted []float64, q float64) float64 {
	n := len(sorted)
	h := q * float64(n-1)
	i := int(h)
	if i >= n-1 {
		return sorted[n-1]
	}
	f := h - float64(i)
	return sorted[i]*(1-f) + sorted[i+1]*f
}

// writeCDFSVG 把一条累计分布曲线写成独立 SVG，标出一成、五成、九成三档。
func writeCDFSVG(path, title string, xs, ys []float64, qx, qy []float64, sampleNote string) error {
	const W, H = 680, 480
	const ml, mr, mt, mb = 70.0, 20.0, 40.0, 46.0
	xmin, xmax := xs[0], xs[len(xs)-1]
	px := func(x float64) float64 { return ml + (x-xmin)/(xmax-xmin)*(W-ml-mr) }
	py := func(y float64) float64 { return mt + (1-y)*(H-mt-mb) }
	var b strings.Builder
	fmt.Fprintf(&b, `<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="monospace" font-size="12">`, W, H)
	fmt.Fprintf(&b, `<rect width="%d" height="%d" fill="white"/>`, W, H)
	fmt.Fprintf(&b, `<text x="%d" y="20" text-anchor="middle">%s</text>`, W/2, title)
	fmt.Fprintf(&b, `<text x="%d" y="36" text-anchor="middle" font-size="11" fill="#555">%s</text>`, W/2, sampleNote)
	for _, g := range []float64{0.1, 0.5, 0.9} {
		fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#bbb" stroke-dasharray="4 3"/>`, ml, py(g), W-mr, py(g))
		fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" text-anchor="end" fill="#555">%.1f</text>`, ml-6, py(g)+4, g)
	}
	fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#bbb" stroke-dasharray="4 3"/>`, ml, py(0), W-mr, py(0))
	fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" text-anchor="end" fill="#555">0</text>`, ml-6, py(0)+4)
	fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" text-anchor="end" fill="#555">1</text>`, ml-6, py(1)+4)
	for i := 0; i <= 4; i++ {
		x := xmin + float64(i)*(xmax-xmin)/4
		fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" text-anchor="middle" fill="#555">%.3f</text>`, px(x), H-mb+16, x)
		fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#ddd"/>`, px(x), mt, px(x), H-mb)
	}
	fmt.Fprintf(&b, `<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="none" stroke="black"/>`, ml, mt, W-ml-mr, H-mt-mb)
	b.WriteString(`<polyline fill="none" stroke="#0645ad" stroke-width="1.5" points="`)
	for i := range xs {
		fmt.Fprintf(&b, "%.1f,%.1f ", px(xs[i]), py(ys[i]))
	}
	b.WriteString(`"/>`)
	for i := range qx {
		fmt.Fprintf(&b, `<circle cx="%.1f" cy="%.1f" r="4" fill="#c00"/>`, px(qx[i]), py(qy[i]))
	}
	b.WriteString(`</svg>`)
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func main() {
	size := flag.Int("size", 1024, "边长像素数")
	seed := flag.Uint64("seed", 1, "种子")
	octaves := flag.Int("octaves", 4, "层数 N")
	gain := flag.Float64("gain", 0.5, "振幅衰减 g（谱斜率参数）")
	lacunarity := flag.Float64("lacunarity", 2.0, "频率倍增 L")
	frequency := flag.Float64("frequency", 0.02, "基础频率 f0")
	outdir := flag.String("outdir", "artifacts/quantile", "存档目录")
	flag.Parse()

	// —— 解析预判（docs/height-quantile.md §2-§5）：只含 (g, N) 与核常数 ——
	sOct2 := varH() * math.Pow(2*eFade2(), 2)
	g, N := *gain, float64(*octaves)
	s1 := (1 - math.Pow(g, N)) / (1 - g)
	s2 := (1 - math.Pow(g, 2*N)) / (1 - g*g)
	sigmaF := math.Sqrt(sOct2 * s2 / (s1 * s1))
	pred := make([]float64, len(quantiles))
	for i, q := range quantiles {
		pred[i] = sigmaF * normInv(q)
	}

	// —— 实测：生成同参数原始高度场（不归一化） ——
	src := noise.New(*seed)
	data := make([]float64, *size**size)
	for y := 0; y < *size; y++ {
		for x := 0; x < *size; x++ {
			data[y**size+x] = fbm.Eval(src, float64(x)**frequency, float64(y)**frequency,
				*octaves, *lacunarity, *gain)
		}
	}
	n := len(data)
	sorted := make([]float64, n)
	copy(sorted, data)
	sort.Float64s(sorted)
	meas := make([]float64, len(quantiles))
	for i, q := range quantiles {
		meas[i] = empQuantile(sorted, q)
	}
	lo, hi := sorted[0], sorted[n-1]

	// 归一化高度图对照：仿射变换用本图实测 min/max。
	normSorted := make([]float64, n)
	for i, v := range sorted {
		normSorted[i] = (v - lo) / (hi - lo)
	}

	// 统计误差尺度：主导层（第 0 层）独立格数 ≈ (size·f0)²。
	nEff := math.Pow(float64(*size)**frequency, 2)

	fmt.Printf("参数: size=%d seed=%d octaves=%d gain=%.3g lacunarity=%.3g frequency=%.3g\n",
		*size, *seed, *octaves, *gain, *lacunarity, *frequency)
	fmt.Printf("采样数 n = %d（%d×%d）；有效独立样本约 n_eff = %.0f（主导层独立格数）\n", n, *size, *size, nEff)
	fmt.Printf("核常数 σ_oct² = %.6f；σ_F = %.6f（仅由 gain=%.3g、octaves=%d 决定）\n\n", sOct2, sigmaF, *gain, *octaves)

	fmt.Println("分位对照（原始高度，逐行）:")
	fmt.Println("q      预判         实测         偏差(实测-预判)  偏差/σ_F   统计标准误 se")
	var tsv strings.Builder
	tsv.WriteString("q\tpred_raw\tmeas_raw\tdiff_raw\tpred_norm\tmeas_norm\tdiff_norm\n")
	for i, q := range quantiles {
		z := normInv(q)
		se := sigmaF * math.Sqrt(q*(1-q)/nEff) / (math.Exp(-z*z/2) / math.Sqrt(2*math.Pi))
		pn := (pred[i] - lo) / (hi - lo)
		mn := empQuantile(normSorted, q)
		fmt.Printf("%.2f   %+.6f   %+.6f   %+ .6f      %+.4f   %.6f\n",
			q, pred[i], meas[i], meas[i]-pred[i], (meas[i]-pred[i])/sigmaF, se)
		fmt.Fprintf(&tsv, "%.2f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
			q, pred[i], meas[i], meas[i]-pred[i], pn, mn, mn-pn)
	}

	fmt.Println("\n尾部区间偏差（q=0.10 与 q=0.90，单独列出）:")
	for _, i := range []int{0, len(quantiles) - 1} {
		d := meas[i] - pred[i]
		fmt.Printf("q=%.2f  预判 %+.6f  实测 %+.6f  偏差 %+.6f（%+.4f σ_F）\n",
			quantiles[i], pred[i], meas[i], d, d/sigmaF)
	}

	fmt.Println("\n归一化高度图对照（min/max 为本图实测值，仿射换算预判）:")
	fmt.Println("q      预判         实测         偏差")
	for i, q := range quantiles {
		pn := (pred[i] - lo) / (hi - lo)
		mn := empQuantile(normSorted, q)
		fmt.Printf("%.2f   %.6f     %.6f     %+.6f\n", q, pn, mn, mn-pn)
	}

	if err := os.MkdirAll(*outdir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir:", err)
		os.Exit(1)
	}
	note := fmt.Sprintf("n = %d samples (size=%d, seed=%d, octaves=%d, gain=%.3g)", n, *size, *seed, *octaves, *gain)

	// 预判 CDF：N(0, σ_F²)。
	px := make([]float64, 401)
	py := make([]float64, 401)
	for i := range px {
		px[i] = -4*sigmaF + 8*sigmaF*float64(i)/400
		py[i] = normCDF(px[i] / sigmaF)
	}
	qy := []float64{quantiles[0], quantiles[1], quantiles[2]}
	if err := writeCDFSVG(filepath.Join(*outdir, "cdf_predicted.svg"),
		"Predicted CDF: N(0, σ_F²)", px, py, pred, qy, note); err != nil {
		fmt.Fprintln(os.Stderr, "svg:", err)
		os.Exit(1)
	}

	// 实测 CDF：降采样到约 2000 点。
	step := n/2000 + 1
	ex := make([]float64, 0, 2001)
	ey := make([]float64, 0, 2001)
	for i := 0; i < n; i += step {
		ex = append(ex, sorted[i])
		ey = append(ey, (float64(i)+0.5)/float64(n))
	}
	ex = append(ex, sorted[n-1])
	ey = append(ey, 1)
	if err := writeCDFSVG(filepath.Join(*outdir, "cdf_empirical.svg"),
		"Empirical CDF (raw fBm heights)", ex, ey, meas, qy, note); err != nil {
		fmt.Fprintln(os.Stderr, "svg:", err)
		os.Exit(1)
	}

	if err := os.WriteFile(filepath.Join(*outdir, "quantiles.tsv"), []byte(tsv.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "tsv:", err)
		os.Exit(1)
	}
	fmt.Printf("\n存档: %s/{cdf_predicted.svg, cdf_empirical.svg, quantiles.tsv}\n", *outdir)
}
