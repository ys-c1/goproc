// 等高线拓扑判定实验：两种无追踪预判 vs marching squares 逐曲线追踪。
//
// 用法：
//
//	go run ./cmd/contour --size 128 --seed 7 --octaves 4 --outdir artifacts/contour
//
// 输出：
//
//	heightmap.pgm / heightmap.f64   高度图（与 gen 子命令同一数据源）
//	levels_d0.10.csv / levels_d0.07.csv   逐档预判与追踪比对表
//	curves_d0.10.json / curves_d0.07.json 全部追踪曲线（供绘图）
//	summary.txt                     汇总与差异（含具体高度值）
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"goproc/api"
	"goproc/internal/contour"
)

func main() {
	fs := flag.NewFlagSet("contour", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 7, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	outdir := fs.String("outdir", "artifacts/contour", "输出目录")
	if err := fs.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	if err := os.MkdirAll(*outdir, 0o755); err != nil {
		fail(err)
	}

	h, err := api.Generate(api.Options{Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5, Seed: *seed, Frequency: 0.02})
	if err != nil {
		fail(err)
	}
	if err := os.WriteFile(filepath.Join(*outdir, "heightmap.pgm"), h.ToPGM(), 0o644); err != nil {
		fail(err)
	}
	if err := os.WriteFile(filepath.Join(*outdir, "heightmap.f64"), f64Bytes(h.Data), 0o644); err != nil {
		fail(err)
	}
	g := &contour.Grid{N: h.Size, Z: h.Data}

	intervals := []float64{0.10, 0.07}
	var summary string
	summary += fmt.Sprintf("heightmap size=%d seed=%d octaves=%d range=[%.6f, %.6f]\n",
		h.Size, *seed, *octaves, h.Data[minIdx(h.Data)], h.Data[maxIdx(h.Data)])
	summary += "网格约定: 四邻域边(水平/垂直)参与跨越与连通性; 对角线不是边; 边界边=两端点都在区域边界上; 边界格=最外层一圈格\n\n"

	for _, d := range intervals {
		levels := makeLevels(d)
		var rows []levelRow
		for _, want := range levels {
			hh, shift := contour.NudgeLevel(g, want)
			r := contour.Analyze(g, hh)
			if msg := r.Check(); msg != "" {
				summary += "MISMATCH " + msg + "\n"
			}
			rows = append(rows, levelRow{want, r, shift})
		}

		tag := fmt.Sprintf("d%g", d)
		writeCSV(filepath.Join(*outdir, "levels_"+tag+".csv"), rows)
		writeCurvesJSON(filepath.Join(*outdir, "curves_"+tag+".json"), h.Size, rows)

		summary += fmt.Sprintf("=== 等高距 Δ=%.2f（%d 档）===\n", d, len(rows))
		summary += "h_used     B  O_pred(B/2) O_trace  V     E     L_pred(2c-b-chi) L_pred(region) c  b  holes L_trace\n"
		var sumB, sumO, sumL int
		for _, rw := range rows {
			r := rw.r
			summary += fmt.Sprintf("%.6f  %3d  %5d       %5d   %5d %5d  %5d       %5d        %3d %3d %4d  %5d\n",
				r.H, r.BoundaryCrossings, r.PredOpenG, r.TraceOpen,
				r.GraphNodes, r.GraphEdges, r.PredCloseG, r.PredCloseR,
				r.RegionComps, r.RegionBoundary, r.BackgroundHoles, r.TraceClosed)
			sumB += r.BoundaryCrossings
			sumO += r.TraceOpen
			sumL += r.TraceClosed
		}
		summary += fmt.Sprintf("合计: 边界跨越 %d (偶数校验), 开曲线 %d, 闭合环 %d\n\n", sumB, sumO, sumL)
	}

	// 附加结构性校验：B 必为偶数；图段与节点恒等式 E-V = -O（路径贡献 -1、环贡献 0）。
	var extra string
	for _, d := range intervals {
		for _, want := range makeLevels(d) {
			hh, _ := contour.NudgeLevel(g, want)
			r := contour.Analyze(g, hh)
			if r.BoundaryCrossings%2 != 0 {
				extra += fmt.Sprintf("h=%.6f B=%d 非偶数\n", r.H, r.BoundaryCrossings)
			}
			if r.GraphEdges-r.GraphNodes != -r.TraceOpen {
				extra += fmt.Sprintf("h=%.6f 段图恒等式 E-V=-O 不成立\n", r.H)
			}
		}
	}
	if extra == "" {
		summary += "全部档位: 预判与追踪完全一致, B 偶数校验全部通过。\n"
	} else {
		summary += extra
	}

	if err := os.WriteFile(filepath.Join(*outdir, "summary.txt"), []byte(summary), 0o644); err != nil {
		fail(err)
	}
	fmt.Print(summary)
}

// makeLevels 取严格位于 (0,1) 内的等间隔档位。
func makeLevels(d float64) []float64 {
	var out []float64
	for k := 1; ; k++ {
		v := d * float64(k)
		if v >= 1 {
			break
		}
		out = append(out, v)
	}
	return out
}

func writeCSV(path string, rows []levelRow) {
	f, err := os.Create(path)
	if err != nil {
		fail(err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	_ = w.Write([]string{
		"h_wanted", "h_used", "nudge",
		"B_boundary_crossings", "O_pred_boundary", "O_trace",
		"V_nodes", "E_segments", "L_pred_local_euler",
		"c_foreground_comps", "b_boundary_comps", "holes", "chi_local", "L_pred_region",
		"L_trace", "match",
	})
	for _, rw := range rows {
		r := rw.r
		match := "ok"
		if r.Check() != "" {
			match = "MISMATCH:" + r.Check()
		}
		_ = w.Write([]string{
			strconv.FormatFloat(rw.levelWanted, 'f', 2, 64),
			strconv.FormatFloat(r.H, 'f', 6, 64),
			strconv.FormatFloat(rw.nudge, 'g', 2, 64),
			strconv.Itoa(r.BoundaryCrossings),
			strconv.Itoa(r.PredOpenG), strconv.Itoa(r.TraceOpen),
			strconv.Itoa(r.GraphNodes), strconv.Itoa(r.GraphEdges), strconv.Itoa(r.PredCloseG),
			strconv.Itoa(r.RegionComps), strconv.Itoa(r.RegionBoundary), strconv.Itoa(r.BackgroundHoles),
			strconv.Itoa(r.ChiLocal),
			strconv.Itoa(r.PredCloseR), strconv.Itoa(r.TraceClosed), match,
		})
	}
	w.Flush()
}

type levelRow struct {
	levelWanted float64
	r           *contour.LevelResult
	nudge       float64
}

type curveJSON struct {
	Open bool        `json:"open"`
	Pts  [][]float64 `json:"pts"`
}

type levelJSON struct {
	H                  float64     `json:"h"`
	BoundaryCrossings  int         `json:"boundary_crossings"`
	OpenPred           int         `json:"open_pred"`
	OpenTrace          int         `json:"open_trace"`
	ClosedPredLocalEuler int         `json:"closed_pred_local_euler"`
	ClosedPredRegion   int         `json:"closed_pred_region"`
	ClosedTrace        int         `json:"closed_trace"`
	Curves             []curveJSON `json:"curves"`
}

func writeCurvesJSON(path string, gridN int, rows []levelRow) {
	var doc struct {
		GridSize int         `json:"grid_size"`
		Levels   []levelJSON `json:"levels"`
	}
	doc.GridSize = gridN
	for _, rw := range rows {
		r := rw.r
		lj := levelJSON{
			H: r.H, BoundaryCrossings: r.BoundaryCrossings,
			OpenPred: r.PredOpenG, OpenTrace: r.TraceOpen,
			ClosedPredLocalEuler: r.PredCloseG, ClosedPredRegion: r.PredCloseR,
			ClosedTrace: r.TraceClosed,
		}
		for _, c := range r.Curves {
			cj := curveJSON{Open: c.Open}
			for _, p := range c.Pts {
				cj.Pts = append(cj.Pts, []float64{p.X, p.Y})
			}
			lj.Curves = append(lj.Curves, cj)
		}
	doc.Levels = append(doc.Levels, lj)
	}
	b, err := json.Marshal(doc)
	if err != nil {
		fail(err)
	}
	if err := os.WriteFile(path, b, 0o644); err != nil {
		fail(err)
	}
}

func f64Bytes(v []float64) []byte {
	b := make([]byte, 8*len(v))
	for i, x := range v {
		u := math.Float64bits(x)
		b[8*i] = byte(u)
		b[8*i+1] = byte(u >> 8)
		b[8*i+2] = byte(u >> 16)
		b[8*i+3] = byte(u >> 24)
		b[8*i+4] = byte(u >> 32)
		b[8*i+5] = byte(u >> 40)
		b[8*i+6] = byte(u >> 48)
		b[8*i+7] = byte(u >> 56)
	}
	return b
}

func minIdx(v []float64) int {
	m := 0
	for i := 1; i < len(v); i++ {
		if v[i] < v[m] {
			m = i
		}
	}
	return m
}

func maxIdx(v []float64) int {
	m := 0
	for i := 1; i < len(v); i++ {
		if v[i] > v[m] {
			m = i
		}
	}
	return m
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "contour:", err)
	os.Exit(1)
}
