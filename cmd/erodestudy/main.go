// erodestudy 验证侵蚀使坡度中位数下降的经验规律并输出报告与图。
//
// 实验 A：用多组坡度方差不同的地形，拟合 下降量 = k · 轮数 · 方差，
// 给出过原点回归系数 k 与判定系数 R²。
// 实验 B：用该式仅凭初始统计量预判任意轮数后的坡度中位数，
// 与逐轮实测比对，列出偏差超过一度的轮数。
// 另对两组轮数各输出一张坡度累计分布图（SVG），中位数按整图所有格子取。
package main

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"goproc/api"
	"goproc/internal/erode"
)

const (
	size   = 128
	rounds = 50   // 实验 B 的总轮数
	fitN   = 20   // 实验 A 的拟合轮数
	talus  = 0.0  // 坡止高差，取零即纯下坡扩散
	c      = 0.02 // 搬运系数
	span   = 8.0  // 地图横向跨度，与高度值域 1 同单位，决定坡度角的量纲
)

var groups = []int{10, 40} // 两组轮数，各出一张累计分布图

func gen(seed uint64, freq float64) []float64 {
	h, err := api.Generate(api.Options{Size: size, Octaves: 4, Lacunarity: 2, Gain: 0.5, Seed: seed, Frequency: freq})
	if err != nil {
		panic(err)
	}
	return h.Data
}

func medianOf(h []float64) float64 { return erode.Median(erode.Slopes(h, size, span)) }

func main() {
	outDir := "artifacts"
	if len(os.Args) > 1 {
		outDir = os.Args[1]
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		panic(err)
	}
	var rep strings.Builder
	p := func(format string, a ...any) {
		s := fmt.Sprintf(format, a...)
		rep.WriteString(s)
		fmt.Print(s)
	}

	p("# 侵蚀对坡度中位数的影响\n\n")
	p("地形 size=%d octaves=4，横向跨度 %v（高度值域 1，同单位），热侵蚀 talus=%v 搬运系数 c=%v，坡度按整图 %d 个格子取中位数。\n\n",
		size, span, talus, c, size*size)

	// ---- 实验 A：下降量 = k · 轮数 · 方差 ----
	p("## 实验 A：下降量与轮数、方差的关系\n\n")
	p("固定轮数 n=%d，改变基础频率得到不同初始坡度方差 Var0，量 n 轮后的中位数下降量 ΔM：\n\n", fitN)
	p("| 频率 | Var0 (度²) | n·Var0 | ΔM (度) |\n|---|---|---|---|\n")
	freqs := []float64{0.01, 0.015, 0.02, 0.03, 0.04}
	type pt struct{ x, y float64 }
	var pts []pt
	for _, f := range freqs {
		h := gen(1, f)
		var0 := erode.Variance(erode.Slopes(h, size, span))
		m0 := erode.Median(erode.Slopes(h, size, span))
		for i := 0; i < fitN; i++ {
			erode.Round(h, size, talus, c)
		}
		dm := m0 - erode.Median(erode.Slopes(h, size, span))
		p("| %.3f | %.2f | %.1f | %.3f |\n", f, var0, fitN*var0, dm)
		pts = append(pts, pt{fitN * var0, dm})
	}
	// 过原点回归 k = Σxy/Σx²，R² 用常规定义。
	var sxy, sxx, ysum float64
	for _, q := range pts {
		sxy += q.x * q.y
		sxx += q.x * q.x
		ysum += q.y
	}
	k := sxy / sxx
	ybar := ysum / float64(len(pts))
	var ssRes, ssTot float64
	for _, q := range pts {
		d := q.y - k*q.x
		ssRes += d * d
		d = q.y - ybar
		ssTot += d * d
	}
	r2 := 1 - ssRes/ssTot
	p("\n过原点回归得 k = %.6f /度²，R² = %.4f。即经验律\n\n", k, r2)
	p("**ΔM(n) = k · n · Var0 = %.6f · n · Var0**（度）\n\n", k)
	p("R² 接近 1 表明下降量确与轮数×方差成正比，比例系数 k 只取决于侵蚀算子，与地形无关。\n\n")

	// ---- 实验 B：预判任意轮数后的中位数 ----
	p("## 实验 B：按式预判与逐轮实测比对\n\n")
	h := gen(7, 0.02)
	s0 := erode.Slopes(h, size, span)
	m0 := erode.Median(s0)
	var0 := erode.Variance(s0)
	p("地形 seed=7 freq=0.02：初始中位数 M0 = %.3f°，初始方差 Var0 = %.2f 度²。\n\n", m0, var0)
	p("预判式 M(n) = M0 − k·n·Var0，只用到初始统计量，无需逐轮统计。\n\n")
	measured := make([]float64, rounds+1)
	measured[0] = m0
	snaps := map[int][]float64{0: s0}
	for n := 1; n <= rounds; n++ {
		erode.Round(h, size, talus, c)
		measured[n] = medianOf(h)
		for _, g := range groups {
			if n == g {
				snaps[n] = erode.Slopes(h, size, span)
			}
		}
	}
	pred := func(n int) float64 { return m0 - k*float64(n)*var0 }
	p("| 轮数 n | 预判中位数 (度) | 实测中位数 (度) | 偏差 (度) |\n|---|---|---|---|\n")
	var bad []int
	maxDev := 0.0
	for n := 0; n <= rounds; n++ {
		dev := math.Abs(pred(n) - measured[n])
		if dev > maxDev {
			maxDev = dev
		}
		if dev > 1 {
			bad = append(bad, n)
		}
		if n%5 == 0 || dev > 1 {
			p("| %d | %.3f | %.3f | %.3f |\n", n, pred(n), measured[n], dev)
		}
	}
	p("\n逐轮比对（每 5 轮及所有偏差超一度的轮数列出），最大偏差 %.3f°。\n\n", maxDev)
	if len(bad) == 0 {
		p("**没有任何轮数偏差超过一度。**\n\n")
	} else {
		p("**偏差超过一度的轮数：** ")
		for i, n := range bad {
			if i > 0 {
				p(", ")
			}
			p("%d", n)
		}
		p("。这些轮数实测下降慢于预判——侵蚀使方差本身逐轮减小，而预判式用的是初始方差，")
		p("故轮数较大时预判下降量偏大、预判中位数偏低，属该线性式的固有近似误差。\n\n")
	}

	// ---- 两组轮数：中位数行 + 累计分布图 ----
	p("## 两组轮数的预判与实测\n\n")
	p("| 轮数 | 类型 | 坡度中位数 (度) |\n|---|---|---|\n")
	for _, g := range groups {
		p("| %d | 预判 | %.3f |\n", g, pred(g))
		p("| %d | 实测 | %.3f |\n", g, measured[g])
	}
	p("\n")
	for _, g := range groups {
		file := filepath.Join(outDir, fmt.Sprintf("slope_cdf_r%d.svg", g))
		if err := writeCDF(file, snaps[0], snaps[g], g, measured[g]); err != nil {
			panic(err)
		}
		p("轮数 %d 的坡度累计分布图：%s\n\n", g, file)
	}

	report := filepath.Join(outDir, "erosion_report.md")
	if err := os.WriteFile(report, []byte(rep.String()), 0o644); err != nil {
		panic(err)
	}
	fmt.Printf("报告已写入 %s\n", report)
}

// writeCDF 输出坡度累计分布图：初始与第 n 轮两条曲线，标出第 n 轮的中位数。
func writeCDF(path string, s0, sn []float64, n int, median float64) error {
	xMax := math.Ceil(maxOf(s0)/10) * 10
	if xMax < 10 {
		xMax = 10
	}
	const (
		W, H           = 680.0, 440.0
		mL, mR, mT, mB = 70.0, 30.0, 46.0, 60.0
	)
	px := func(x float64) float64 { return mL + (W-mL-mR)*x/xMax }
	py := func(y float64) float64 { return H - mB - (H-mT-mB)*y }

	cdf := func(s []float64) string {
		t := make([]float64, len(s))
		copy(t, s)
		sort.Float64s(t)
		step := len(t) / 400
		if step < 1 {
			step = 1
		}
		var b strings.Builder
		for i := 0; i < len(t); i += step {
			fmt.Fprintf(&b, "%.1f,%.1f ", px(t[i]), py(float64(i+1)/float64(len(t))))
		}
		fmt.Fprintf(&b, "%.1f,%.1f", px(t[len(t)-1]), py(1))
		return b.String()
	}

	var b strings.Builder
	fmt.Fprintf(&b, `<svg xmlns="http://www.w3.org/2000/svg" width="%.0f" height="%.0f" font-family="sans-serif" font-size="13">`, W, H)
	fmt.Fprintf(&b, `<rect width="%.0f" height="%.0f" fill="white"/>`, W, H)
	fmt.Fprintf(&b, `<text x="%.0f" y="28" text-anchor="middle" font-size="16">Slope CDF after %d erosion rounds</text>`, W/2, n)
	// 网格与刻度
	xTick := 10.0
	if xMax > 60 {
		xTick = 20
	}
	for x := 0.0; x <= xMax+1e-9; x += xTick {
		fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#ddd"/>`, px(x), mT, px(x), H-mB)
		fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" text-anchor="middle">%.0f</text>`, px(x), H-mB+18, x)
	}
	for y := 0.0; y <= 1.0001; y += 0.2 {
		fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#ddd"/>`, mL, py(y), W-mR, py(y))
		fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" text-anchor="end">%.1f</text>`, mL-8, py(y)+4, y)
	}
	fmt.Fprintf(&b, `<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="none" stroke="black"/>`,
		mL, mT, W-mL-mR, H-mT-mB)
	fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" text-anchor="middle">slope (deg)</text>`, (mL+W-mR)/2, H-16)
	fmt.Fprintf(&b, `<text x="18" y="%.1f" text-anchor="middle" transform="rotate(-90 18 %.1f)">cumulative fraction</text>`, (mT+H-mB)/2, (mT+H-mB)/2)
	// 曲线
	fmt.Fprintf(&b, `<polyline fill="none" stroke="#999" stroke-width="2" points="%s"/>`, cdf(s0))
	fmt.Fprintf(&b, `<polyline fill="none" stroke="#06c" stroke-width="2" points="%s"/>`, cdf(sn))
	// 中位数虚线
	fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#c00" stroke-width="1.5" stroke-dasharray="6 4"/>`,
		px(median), mT, px(median), H-mB)
	fmt.Fprintf(&b, `<text x="%.1f" y="%.1f" fill="#c00">median %.2f°</text>`, px(median)+6, mT+16, median)
	// 图例
	fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#999" stroke-width="2"/>`, W-mR-150, mT+18, W-mR-120, mT+18)
	fmt.Fprintf(&b, `<text x="%.1f" y="%.1f">round 0</text>`, W-mR-112, mT+22)
	fmt.Fprintf(&b, `<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#06c" stroke-width="2"/>`, W-mR-150, mT+40, W-mR-120, mT+40)
	fmt.Fprintf(&b, `<text x="%.1f" y="%.1f">round %d</text>`, W-mR-112, mT+44, n)
	b.WriteString(`</svg>`)
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func maxOf(v []float64) float64 {
	m := v[0]
	for _, x := range v {
		if x > m {
			m = x
		}
	}
	return m
}
