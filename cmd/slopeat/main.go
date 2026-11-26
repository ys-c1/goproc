// slopeat 坡度统计实验：对 K=1..kmax 扫描，比较理论预判与实测，
// 并对 --groups 指定的两组层数各输出一张坡度直方图（SVG）。
package main

import (
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"goproc/api"
	"goproc/internal/slope"
)

const tan30 = 0.5773502691896258 // tan(30°)

func deg(g float64) float64 { return math.Atan(g) * 180 / math.Pi }

type row struct {
	K         int
	predRange float64
	measRange float64
	predS     float64
	predMax   float64 // 度，R=N/1000 分位
	measMax   float64 // 度，同分位实测均值
	measMaxSD float64
	trueMax   float64 // 度，实测逐图最大值取平均
	predEx    float64 // >30° 预判占比
	measEx    float64 // >30° 实测占比
}

func measure(size, k, seed int) (q, tmax, ex, rng float64) {
	opt := api.Options{Size: size, Octaves: k, Lacunarity: 2, Gain: 0.5, Seed: uint64(seed), Frequency: 0.02}
	data, rawRange := slope.Generate(opt)
	sl := slope.Slopes(data, size)
	sort.Float64s(sl)
	n := float64(size * size)
	p := 1 - 1000/n
	q = slope.Quantile(sl, p)
	tmax = sl[len(sl)-1]
	above := 0
	for _, g := range sl {
		if g > tan30 {
			above++
		}
	}
	ex = float64(above) / float64(len(sl))
	return q, tmax, ex, rawRange
}

func main() {
	size := flag.Int("size", 512, "边长")
	nseeds := flag.Int("seeds", 8, "种子数（种子取 1..N）")
	kmax := flag.Int("kmax", 8, "扫描的最大层数")
	groups := flag.String("groups", "2,8", "直方图层数，逗号分隔")
	out := flag.String("out", "artifacts/slope", "输出目录")
	flag.Parse()

	if err := os.MkdirAll(*out, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	p := slope.DefaultParams()
	n := float64(*size * *size)
	recur := n / 1000 // 重现格数 R(N,K) = N/1000
	nEff := slope.NEff(*size, p.F0)
	quantileP := 1 - 1000/n

	var rows []row
	for k := 1; k <= *kmax; k++ {
		s := slope.PredS(k, p, nEff)
		r := row{
			K:         k,
			predRange: slope.PredRange(k, p, nEff),
			predS:     s,
			predMax:   deg(slope.RayleighQuantile(s, recur)),
			predEx:    slope.ExceedProb(s, tan30),
		}
		var qs, ms, es, rs []float64
		for seed := 1; seed <= *nseeds; seed++ {
			q, tmax, ex, rng := measure(*size, k, seed)
			qs = append(qs, deg(q))
			ms = append(ms, deg(tmax))
			es = append(es, ex)
			rs = append(rs, rng)
		}
		r.measMax, r.measMaxSD = meanSD(qs)
		r.trueMax, _ = meanSD(ms)
		r.measEx, _ = meanSD(es)
		r.measRange, _ = meanSD(rs)
		rows = append(rows, r)
	}

	var gks []int
	for _, s := range strings.Split(*groups, ",") {
		k, err := strconv.Atoi(strings.TrimSpace(s))
		if err != nil || k < 1 || k > *kmax {
			fmt.Fprintln(os.Stderr, "非法直方图层数:", s)
			os.Exit(2)
		}
		gks = append(gks, k)
	}
	var histFiles []string
	for _, k := range gks {
		var pool []float64
		for seed := 1; seed <= *nseeds; seed++ {
			opt := api.Options{Size: *size, Octaves: k, Lacunarity: 2, Gain: 0.5, Seed: uint64(seed), Frequency: 0.02}
			data, _ := slope.Generate(opt)
			pool = append(pool, slope.Slopes(data, *size)...)
		}
		s := slope.PredS(k, p, nEff)
		f := filepath.Join(*out, fmt.Sprintf("hist-k%d.svg", k))
		writeHistSVG(f, k, pool, s)
		histFiles = append(histFiles, f)
		fmt.Println("wrote", f)
	}

	rep := filepath.Join(*out, "report.md")
	writeReport(rep, *size, *nseeds, recur, quantileP, nEff, rows, gks, histFiles)
	fmt.Println("wrote", rep)
	printTable(rows)
}

func meanSD(v []float64) (m, sd float64) {
	for _, x := range v {
		m += x
	}
	m /= float64(len(v))
	for _, x := range v {
		sd += (x - m) * (x - m)
	}
	sd = math.Sqrt(sd / float64(len(v)))
	return m, sd
}

func dev(pred, meas float64) float64 {
	if meas == 0 {
		return 0
	}
	return (pred - meas) / meas * 100
}

func printTable(rows []row) {
	fmt.Printf("\n%-3s %9s %9s %9s %9s %9s %8s %9s %9s\n",
		"K", "E[R]预", "E[R]实", "最大预°", "最大实°", "真极值°", "偏差%", ">30°预", ">30°实")
	for _, r := range rows {
		fmt.Printf("%-3d %9.3f %9.3f %9.3f %9.3f %9.3f %8.1f %9.2e %9.2e\n",
			r.K, r.predRange, r.measRange, r.predMax, r.measMax, r.trueMax,
			dev(r.predMax, r.measMax), r.predEx, r.measEx)
	}
}

func pct(x float64) string {
	if x == 0 {
		return "0"
	}
	if x < 1e-4 {
		return fmt.Sprintf("%.1e", x)
	}
	return fmt.Sprintf("%.4f%%", x*100)
}

func writeReport(path string, size, nseeds int, recur, qp, nEff float64, rows []row, gks []int, hists []string) {
	var b strings.Builder
	fmt.Fprintf(&b, "# 坡度统计实验报告\n\n")
	fmt.Fprintf(&b, "参数：边长 %d（总格数 N=%d），种子 1..%d，f0=0.02，lacunarity=2，gain=0.5。\n", size, size*size, nseeds)
	fmt.Fprintf(&b, "重现格数 R(N,K) = N/1000 = %.1f（与 K 无关，K 通过 s_K 进入预判），分位 p = 1−1000/N = %.6f。\n", recur, qp)
	fmt.Fprintf(&b, "有效独立样本数 N_eff = (S·f0)² = %.1f。坡度角 θ 满足 tanθ = |∇h|（垂直单位=整个值域，水平单位=1 像素）。\n\n", nEff)
	fmt.Fprintf(&b, "理论推导见 docs/slope-theory.md。\n\n")

	fmt.Fprintf(&b, "## 按层数扫描\n\n")
	fmt.Fprintf(&b, "| K | E[R] 预判 | E[R] 实测 | s_K 预判 | 最大坡度预判° | 最大坡度实测° | 实测真极值° | 偏差 | >30°预判 | >30°实测 |\n")
	fmt.Fprintf(&b, "|---|---|---|---|---|---|---|---|---|---|\n")
	for _, r := range rows {
		fmt.Fprintf(&b, "| %d | %.3f | %.3f | %.5f | %.3f | %.3f±%.3f | %.3f | %+.1f%% | %s | %s |\n",
			r.K, r.predRange, r.measRange, r.predS, r.predMax, r.measMax, r.measMaxSD, r.trueMax,
			dev(r.predMax, r.measMax), pct(r.predEx), pct(r.measEx))
	}

	fmt.Fprintf(&b, "\n## 两组层数对比\n\n")
	fmt.Fprintf(&b, "| 层数 | 行 | 最大坡度° | >30° 占比 |\n|---|---|---|---|\n")
	for _, k := range gks {
		r := rows[k-1]
		fmt.Fprintf(&b, "| K=%d | 预判 | %.3f | %s |\n", k, r.predMax, pct(r.predEx))
		fmt.Fprintf(&b, "| K=%d | 实测 | %.3f±%.3f | %s |\n", k, r.measMax, r.measMaxSD, pct(r.measEx))
	}

	fmt.Fprintf(&b, "\n## 偏差超过两成的层数\n\n")
	flagged := false
	for _, r := range rows {
		if d := dev(r.predMax, r.measMax); math.Abs(d) > 20 {
			fmt.Fprintf(&b, "- K=%d：偏差 %+.1f%%\n", r.K, d)
			flagged = true
		}
	}
	if !flagged {
		fmt.Fprintf(&b, "无。\n")
	}
	fmt.Fprintf(&b, "\n原因分析见 docs/slope-theory.md 第 5 节。\n")

	fmt.Fprintf(&b, "\n## 坡度直方图\n\n")
	for i, k := range gks {
		fmt.Fprintf(&b, "- K=%d：[%s](%s)\n", k, filepath.Base(hists[i]), filepath.Base(hists[i]))
	}

	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

// writeHistSVG 输出对数纵轴的坡度直方图，并叠加 Rayleigh 预判密度。
func writeHistSVG(path string, k int, samples []float64, s float64) {
	sort.Float64s(samples)
	// 横轴用度，上限取实测最大与预判 99.99% 分位的较大者。
	hi := math.Max(deg(samples[len(samples)-1]), deg(slope.RayleighQuantile(s, 1e4))) * 1.05
	nb := 120
	bins := make([]int, nb)
	for _, g := range samples {
		d := deg(g)
		i := int(d / hi * float64(nb))
		if i >= nb {
			i = nb - 1
		}
		if i >= 0 {
			bins[i]++
		}
	}
	maxC := 1
	for _, c := range bins {
		if c > maxC {
			maxC = c
		}
	}

	const W, H = 880, 520
	const ml, mr, mt, mb = 70.0, 20.0, 40.0, 60.0
	pw, ph := float64(W)-ml-mr, float64(H)-mt-mb
	yOf := func(c float64) float64 {
		if c < 1 {
			c = 1
		}
		return mt + ph*(1-math.Log10(c)/math.Log10(float64(maxC)))
	}
	xOf := func(d float64) float64 { return ml + pw*d/hi }

	var b strings.Builder
	fmt.Fprintf(&b, `<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="monospace">`, W, H)
	fmt.Fprintf(&b, `<rect width="%d" height="%d" fill="white"/>`, W, H)
	fmt.Fprintf(&b, `<text x="%.0f" y="24" font-size="16">K=%d 坡度直方图（%d 样本，对数纵轴）与 Rayleigh 预判</text>`, ml, k, len(samples))

	// 横轴刻度（度）
	for t := 0.0; t <= hi; t += math.Max(0.5, math.Round(hi/8*2)/2) {
		fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#ccc"/>`, xOf(t), mt, xOf(t), mt+ph)
		fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" font-size="11" text-anchor="middle">%.1f°</text>`, xOf(t), mt+ph+18, t)
	}
	// 纵轴刻度（10 的幂）
	for e := 0; ; e++ {
		c := math.Pow(10, float64(e))
		if c > float64(maxC) {
			break
		}
		fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#eee"/>`, ml, yOf(c), ml+pw, yOf(c))
		fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" font-size="11" text-anchor="end">10^%d</text>`, ml-6, yOf(c)+4, e)
	}

	// 直方图柱
	for i, c := range bins {
		if c == 0 {
			continue
		}
		x0 := xOf(float64(i) / float64(nb) * hi)
		x1 := xOf(float64(i+1) / float64(nb) * hi)
		fmt.Fprintf(&b, `<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="#7ab" opacity="0.8"/>`,
			x0, yOf(float64(c)), x1-x0-0.5, mt+ph-yOf(float64(c)))
	}

	// 预判密度曲线：p_θ(θ) = (tanθ/s²)·exp(−tan²θ/2s²)·sec²θ·(π/180)，按柱宽与样本数缩放
	binW := hi / float64(nb)
	var pts []string
	for i := 0; i <= 400; i++ {
		d := hi * float64(i) / 400
		th := d * math.Pi / 180
		tg := math.Tan(th)
		pdf := tg / (s * s) * math.Exp(-tg*tg/(2*s*s)) / (math.Cos(th) * math.Cos(th)) * math.Pi / 180
		cnt := pdf * float64(len(samples)) * binW
		if cnt < 0.5 {
			continue
		}
		pts = append(pts, fmt.Sprintf("%.1f,%.1f", xOf(d), yOf(cnt)))
	}
	fmt.Fprintf(&b, `<polyline points="%s" fill="none" stroke="#c33" stroke-width="2"/>`, strings.Join(pts, " "))

	fmt.Fprintf(&b, `<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="none" stroke="#333"/>`, ml, mt, pw, ph)
	fmt.Fprintf(&b, `<text x="%.1f" y="%.0f" font-size="12" fill="#c33">― Rayleigh 预判 s_K=%.5f</text>`, ml+pw-260, mt+16, s)
	fmt.Fprintf(&b, `<text x="%.1f" y="%.0f" font-size="12" fill="#7ab">■ 实测</text>`, ml+pw-260, mt+34)
	fmt.Fprintf(&b, `</svg>`)

	if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
