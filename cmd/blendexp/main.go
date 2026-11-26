// blendexp 生成低频/高频两层 fBm，在水平过渡带内用三种权重混合，
// 逐行导出接缝跳变、棱能量及其理论最小值、权重曲线偏差。
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"goproc/internal/blend"
	"goproc/internal/fbm"
	"goproc/internal/hmap"
	"goproc/internal/noise"
)

type method struct {
	name string
	kind string // func | optimal
	wf   blend.Weight
}

func main() {
	size := flag.Int("size", 256, "边长（格）")
	seed := flag.Uint64("seed", 7, "噪声种子")
	loOct := flag.Int("lo-octaves", 3, "低频层数")
	hiOct := flag.Int("hi-octaves", 6, "高频层数")
	loFreq := flag.Float64("lo-freq", 0.01, "低频基础频率")
	hiFreq := flag.Float64("hi-freq", 0.04, "高频基础频率")
	widths := flag.String("widths", "8,16,32", "过渡带宽度（格），逗号分隔")
	epsRel := flag.Float64("eps-rel", 0.15, "最优权重正则相对阈值（相对 Δmid 的 rms）")
	outDir := flag.String("out", "artifacts/exp", "输出目录")
	flag.Parse()

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fail(err)
	}
	src := noise.New(*seed)
	n := *size
	l := make([]float64, n*n)
	h := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			l[y*n+x] = fbm.Eval(src, float64(x)**loFreq, float64(y)**loFreq, *loOct, 2.0, 0.5)
			h[y*n+x] = fbm.Eval(src, float64(x)**hiFreq, float64(y)**hiFreq, *hiOct, 2.0, 0.5)
		}
	}

	ws := parseIntList(*widths)
	methods := []method{
		{name: "linear", kind: "func", wf: blend.Linear},
		{name: "smoothstep", kind: "func", wf: blend.Smoothstep},
		{name: "optimal", kind: "optimal"},
	}

	rowsPath := filepath.Join(*outDir, "rows.csv")
	rf, err := os.Create(rowsPath)
	if err != nil {
		fail(err)
	}
	defer rf.Close()
	cw := csv.NewWriter(rf)
	header := []string{"width", "method", "row", "ja", "jb", "ja_abs", "jb_abs",
		"energy", "visible", "bound_eps", "gap_rel", "w_rms_dist", "eps2"}
	must(cw.Write(header))

	summary := [][]string{{"width", "method", "ja_mean_abs", "jb_mean_abs", "ja_max_abs", "jb_max_abs",
		"energy_mean", "visible_mean", "bound_mean", "gap_rel_mean", "w_rms_dist_mean"}}

	for _, W := range ws {
		a := (n - W) / 2
		b := a + W
		center := n / 2
		z0, z1 := center-W, center+W // 裁剪窗口含一倍带宽余量
		if z0 < 1 {
			z0 = 1
		}
		if z1 > n-2 {
			z1 = n - 2
		}

		// 每行入口对齐：h' = h - (h[a]-l[a])，保证 Δ(a)=0。
		ha := make([][]float64, n)
		la := make([][]float64, n)
		for y := 0; y < n; y++ {
			ha[y] = make([]float64, n)
			la[y] = make([]float64, n)
			copy(ha[y], h[y*n:(y+1)*n])
			copy(la[y], l[y*n:(y+1)*n])
			off := ha[y][a] - la[y][a]
			for x := range ha[y] {
				ha[y][x] -= off
			}
		}

		// 权重文件按宽度独立，重跑时先清空避免追加累积。
		weightPath := filepath.Join(*outDir, fmt.Sprintf("weights_w%d.csv", W))
		must(os.WriteFile(weightPath, nil, 0o644))

		for _, m := range methods {
			blended := make([]float64, n*n)
			var wCenter []float64 // 中心行实际使用的权重
			for y := 0; y < n; y++ {
				dm := blend.EdgeMidDeltas(la[y], ha[y], a, b)
				eps2 := blend.Eps2FromD(dm, *epsRel)
				var w []float64
				if m.kind == "optimal" {
					w = blend.OptimalWeights(dm, eps2)
				} else {
					w = blend.SampleWeight(m.wf, a, b)
				}
				row := blend.BlendRowSamples(la[y], ha[y], a, b, w)
				copy(blended[y*n:(y+1)*n], row)

				ja, jb := blend.SeamJumps(row, a, b, 1)
				e := blend.RidgeEnergy(w, dm, eps2, 1)
				vis := blend.VisibleRidge(w, dm, 1)
				// 理论最小值取与求解器同一正则问题的 Cauchy 下界，Δ 过零时仍有限。
				rb := blend.RegularizedBound(dm, eps2, 1)
				wOpt := blend.OptimalWeights(dm, eps2)
				dist := blend.RMSWeightDistance(w, wOpt)
				gap := (e - rb) / rb
				must(cw.Write([]string{
					strconv.Itoa(W), m.name, strconv.Itoa(y),
					strconv.FormatFloat(ja, 'g', 17, 64),
					strconv.FormatFloat(jb, 'g', 17, 64),
					strconv.FormatFloat(math.Abs(ja), 'g', 17, 64),
					strconv.FormatFloat(math.Abs(jb), 'g', 17, 64),
					strconv.FormatFloat(e, 'g', 17, 64),
					strconv.FormatFloat(vis, 'g', 17, 64),
					strconv.FormatFloat(rb, 'g', 17, 64),
					strconv.FormatFloat(gap, 'g', 17, 64),
					strconv.FormatFloat(dist, 'g', 17, 64),
					strconv.FormatFloat(eps2, 'g', 17, 64),
				}))
				if y == center {
					wCenter = w
				}
			}

			// 全图 PGM（各自独立归一化，仅用于可视化）。
			pgm := make([]float64, len(blended))
			copy(pgm, blended)
			hmap.Normalize(pgm)
			must(os.WriteFile(filepath.Join(*outDir, fmt.Sprintf("blend_w%d_%s.pgm", W, m.name)),
				hmap.EncodePGM(n, pgm), 0o644))

			// 接缝附近裁剪：中心行原始（未归一化）浮点，供 Python 渲染放大图。
			cropPath := filepath.Join(*outDir, fmt.Sprintf("crop_w%d_%s.csv", W, m.name))
			writeCrop(cropPath, blended, la[center], ha[center], n, z0, z1, center, a, b)
			// 中心行权重曲线（三种方法各一段）。
			appendWeightCurve(weightPath, m.name, wCenter)

			cw.Flush()
			must(cw.Error())
			summary = append(summary, collect(rowsPath, W, m.name)...)
		}
	}
	cw.Flush()
	must(cw.Error())

	sp := filepath.Join(*outDir, "summary.csv")
	sf, err := os.Create(sp)
	if err != nil {
		fail(err)
	}
	defer sf.Close()
	sw := csv.NewWriter(sf)
	for _, r := range summary {
		must(sw.Write(r))
	}
	sw.Flush()
	must(sw.Error())
	fmt.Println("wrote", rowsPath, "and", sp)
}

func writeCrop(path string, blended2d []float64, lRow, hRow []float64, n, z0, z1, cy, a, b int) {
	f, err := os.Create(path)
	if err != nil {
		fail(err)
	}
	defer f.Close()
	cw := csv.NewWriter(f)
	must(cw.Write([]string{"x", "blend_center_row", "L", "H", "in_band"}))
	for x := z0; x <= z1; x++ {
		in := 0
		if x >= a && x <= b {
			in = 1
		}
		must(cw.Write([]string{
			strconv.Itoa(x),
			strconv.FormatFloat(blended2d[cy*n+x], 'g', 8, 64),
			strconv.FormatFloat(lRow[x], 'g', 8, 64),
			strconv.FormatFloat(hRow[x], 'g', 8, 64),
			strconv.Itoa(in),
		}))
	}
	cw.Flush()
	must(cw.Error())
}

func appendWeightCurve(path, name string, w []float64) {
	info, err := os.Stat(path)
	if err != nil {
		fail(err)
	}
	newFile := info.Size() == 0 // 预清空后的空文件同样需要写表头
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		fail(err)
	}
	defer f.Close()
	cw := csv.NewWriter(f)
	if newFile {
		must(cw.Write([]string{"t", "method", "w"}))
	}
	for i := range w {
		t := float64(i) / float64(len(w)-1)
		must(cw.Write([]string{strconv.FormatFloat(t, 'g', 6, 64), name,
			strconv.FormatFloat(w[i], 'g', 8, 64)}))
	}
	cw.Flush()
	must(cw.Error())
}

func collect(rowsPath string, W int, name string) [][]string {
	f, err := os.Open(rowsPath)
	if err != nil {
		fail(err)
	}
	defer f.Close()
	r := csv.NewReader(f)
	all, err := r.ReadAll()
	if err != nil {
		fail(err)
	}
	var jaSum, jbSum, jaMax, jbMax, eSum, visSum, cbSum, gapSum, dSum float64
	k := 0
	for _, rec := range all[1:] {
		if rec[0] != strconv.Itoa(W) || rec[1] != name {
			continue
		}
		jaAbs := atof(rec[5])
		jbAbs := atof(rec[6])
		jaSum += jaAbs
		jbSum += jbAbs
		if jaAbs > jaMax {
			jaMax = jaAbs
		}
		if jbAbs > jbMax {
			jbMax = jbAbs
		}
		eSum += atof(rec[7])
		visSum += atof(rec[8])
		cbSum += atof(rec[9])
		gapSum += atof(rec[10])
		dSum += atof(rec[11])
		k++
	}
	if k == 0 {
		return nil
	}
	return [][]string{{
		strconv.Itoa(W), name,
		fnum(jaSum / float64(k)), fnum(jbSum / float64(k)),
		fnum(jaMax), fnum(jbMax),
		fnum(eSum / float64(k)), fnum(visSum / float64(k)),
		fnum(cbSum / float64(k)),
		fnum(gapSum / float64(k)), fnum(dSum / float64(k)),
	}}
}

func parseIntList(s string) []int {
	var out []int
	cur := 0
	has := false
	for _, c := range s {
		if c == ',' {
			if has {
				out = append(out, cur)
			}
			cur, has = 0, false
			continue
		}
		cur = cur*10 + int(c-'0')
		has = true
	}
	if has {
		out = append(out, cur)
	}
	return out
}

func atof(s string) float64 {
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		fail(err)
	}
	return v
}

func fnum(v float64) string { return strconv.FormatFloat(v, 'g', 6, 64) }

func must(err error) {
	if err != nil {
		fail(err)
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "blendexp:", err)
	os.Exit(1)
}
