// Command gully 完成“双阈值合取预判冲沟核心格 → 两组强度侵蚀 →
// 侵蚀后逐格统计核对 → 八邻域连通段 → 标注图/报告”的完整闭环。
//
// 用法（仓库根目录）：
//
//	go run ./cmd/gully
//
// 全部产物默认写入 artifacts/gully/，并把两张标注图与摘要复制到 artifacts/ 根。
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"goproc/api"
	"goproc/internal/gully"
	"goproc/internal/hmap"
	"goproc/internal/hydro"
)

type coordRec struct {
	Index int    `json:"index"`
	X     int    `json:"x"`
	Y     int    `json:"y"`
	Kind  string `json:"kind"` // missing=预判有实测无；extra=实测有预判无
}

type groupMetrics struct {
	ID                 string     `json:"id"`
	Label              string     `json:"label"`
	K                  float64    `json:"K"`
	Rounds             int        `json:"rounds"`
	DCore              float64    `json:"dCore"`
	BankCap            float64    `json:"bankCap"`
	PredictedCoreCells int        `json:"predictedCoreCells"`
	MeasuredCoreCells  int        `json:"measuredCoreCells"`
	Segments8Pred      int        `json:"segments8Predicted"`
	Segments8Meas      int        `json:"segments8Measured"`
	MissingCells       int        `json:"missingCells"`
	ExtraCells         int        `json:"extraCells"`
	Match              bool       `json:"cellByCellMatch"`
	MismatchCoords     []coordRec `json:"mismatchCoordinates"`
	NearRadius         int        `json:"nearRadius"`
	NearViolations     int        `json:"nearViolations"`
	BankCells          int        `json:"bankCells"`
	ClippedCells       int        `json:"floorClippedCells"`
	MaxDrive           float64    `json:"maxDriveOnCore"`
	MaxCut             float64    `json:"maxCut"`
	TotalCut           float64    `json:"totalCut"`
	MinCoreHeight0     float64    `json:"minInitialHeightOnCore"`
	AnnotationPNG      string     `json:"annotationPng"`
	ComparisonCSV      string     `json:"comparisonCsv"`
	MismatchCSV        string     `json:"mismatchCsv"`
	MismatchCoordsTXT  string     `json:"mismatchCoordinatesTxt"`
}

type metricsFile struct {
	Terrain struct {
		Size       int     `json:"size"`
		Seed       uint64  `json:"seed"`
		Octaves    int     `json:"octaves"`
		Lacunarity float64 `json:"lacunarity"`
		Gain       float64 `json:"gain"`
		Frequency  float64 `json:"frequency"`
		Cells      int     `json:"cells"`
	} `json:"terrain"`
	FlowRule          string           `json:"flowRule"`
	AccumulationRule  string           `json:"accumulationRule"`
	MaxAccumulation   int              `json:"maxAccumulation"`
	SinkCells         int              `json:"sinkCells"`
	SlopeRule         string           `json:"slopeRule"`
	SegmentRule       string           `json:"segmentRule"`
	Thresholds        gully.Thresholds `json:"thresholds"`
	SlopeThresholdDeg float64          `json:"slopeThresholdDegrees"`
	DCore             float64          `json:"dCore"`
	Prediction        struct {
		CoreCells    int   `json:"coreCells"`
		Segments8    int   `json:"segments8"`
		SegmentSizes []int `json:"segmentSizes"`
	} `json:"prediction"`
	Groups []*groupMetrics `json:"groups"`
}

func main() {
	size := flag.Int("size", 128, "边长")
	seed := flag.Uint64("seed", 7, "种子")
	out := flag.String("out", "artifacts/gully", "完整产物目录")
	root := flag.String("artifact-root", "artifacts", "供快速查看的根产物目录")
	flag.Parse()

	if err := run(*size, *seed, *out, *root); err != nil {
		fmt.Fprintln(os.Stderr, "gully failed:", err)
		os.Exit(1)
	}
}

func run(size int, seed uint64, outDir, rootDir string) error {
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	if err := os.MkdirAll(rootDir, 0o755); err != nil {
		return err
	}

	// ---------- 1. 确定性基准地形 ----------
	opt := api.Options{Size: size, Octaves: 4, Lacunarity: 2, Gain: 0.5, Seed: seed, Frequency: 0.02}
	hm, err := api.Generate(opt)
	if err != nil {
		return err
	}
	n := hm.Size
	N := n * n
	h0 := hm.Data

	// ---------- 2. 侵蚀前即可完成的水文场与合取预判 ----------
	f := hydro.Build(h0, n)
	th := gully.CanonicalThresholds()
	pred := gully.CoreMask(f, th)
	predComp, segs := gully.Components8(pred, n)
	predCount := len(segs)
	predCells := countTrue(pred)

	segmentSizes := make([]int, 0, len(segs))
	var minCoreH = math.Inf(1)
	var maxDriveCore float64
	probe := gully.IntensityGroups()[0] // 仅用其 cap 参数算 drive 上界
	for _, s := range segs {
		segmentSizes = append(segmentSizes, s.Count)
	}
	for i, v := range pred {
		if v {
			if h0[i] < minCoreH {
				minCoreH = h0[i]
			}
			if d := gully.Drive(f, th, probe, i); d > maxDriveCore {
				maxDriveCore = d
			}
		}
	}

	// ---------- 3. 预判产物：核心格坐标、段表、掩膜、原始高程 ----------
	if err := writeCoreCellsCSV(filepath.Join(outDir, "predicted_core_cells.csv"), f, th, predComp); err != nil {
		return err
	}
	if err := writeSegmentsCSV(filepath.Join(outDir, "segments8.csv"), segs); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(outDir, "height_before.pgm"), hm.ToPGM(), 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(outDir, "predicted_core_mask.pgm"),
		hmap.EncodePGM(n, maskToFloat(pred)), 0o644); err != nil {
		return err
	}
	if err := writeF64(filepath.Join(outDir, "height_before.f64"), h0); err != nil {
		return err
	}

	// ---------- 4. 两组强度：跑完整侵蚀并逐格核对 ----------
	groups := gully.IntensityGroups()
	mf := metricsFile{}
	mf.Terrain.Size = size
	mf.Terrain.Seed = seed
	mf.Terrain.Octaves = opt.Octaves
	mf.Terrain.Lacunarity = opt.Lacunarity
	mf.Terrain.Gain = opt.Gain
	mf.Terrain.Frequency = opt.Frequency
	mf.Terrain.Cells = N
	mf.FlowRule = "D8 八邻域最陡下坡，坡度=落差/距离(斜距sqrt2)，并列取方向编号最小(E,SE,S,SW,W,NW,N,NE)，无严格下坡为终点"
	mf.AccumulationRule = "每格初始贡献1，按高程降序把上游格数累加给唯一接收格（含本格）"
	maxAcc := 0
	for _, a := range f.Acc {
		if a > maxAcc {
			maxAcc = a
		}
	}
	mf.MaxAccumulation = maxAcc
	mf.SinkCells = f.SinkCount()
	mf.SlopeRule = "D8 最大正坡度 max((h_i-h_j)/L_ij)"
	mf.SegmentRule = "八邻域（Chebyshev 距离1）BFS 连通分量"
	mf.Thresholds = th
	mf.SlopeThresholdDeg = th.Slope * 180 / math.Pi
	mf.DCore = gully.DCore
	mf.Prediction.CoreCells = predCells
	mf.Prediction.Segments8 = predCount
	mf.Prediction.SegmentSizes = segmentSizes

	rootPNGs := map[string][2]string{} // id -> [ascii root, chinese root]

	// 先跑完两组，渲染时共用同一绝对深度标尺，让强度差异在图上可辨。
	results := map[string]*gully.Result{}
	depthScale := gully.DCore
	for _, p := range groups {
		r := gully.Erode(h0, f, th, pred, p)
		results[p.Name] = r
		if r.MaxCut > depthScale {
			depthScale = r.MaxCut
		}
	}

	for _, p := range groups {
		res := results[p.Name]
		meas := gully.MeasuredCore(res.Cut, gully.DCore)
		measComp, measSegs := gully.Components8(meas, n)
		diff := gully.Compare(pred, meas)
		zone, violations := gully.NearCheck(pred, res.Cut, n, 1, gully.EpsEroded)

		mc := make([]coordRec, 0, len(diff.Missing)+len(diff.Extra))
		for _, i := range diff.Missing {
			x, y := hydro.XY(n, i)
			mc = append(mc, coordRec{Index: i, X: x, Y: y, Kind: "missing_predicted_but_not_measured"})
		}
		for _, i := range diff.Extra {
			x, y := hydro.XY(n, i)
			mc = append(mc, coordRec{Index: i, X: x, Y: y, Kind: "extra_measured_but_not_predicted"})
		}

		gm := &groupMetrics{
			ID: p.Name, Label: p.Label, K: p.K, Rounds: p.Rounds, DCore: gully.DCore,
			BankCap:            p.BankCap,
			PredictedCoreCells: predCells, MeasuredCoreCells: countTrue(meas),
			Segments8Pred: predCount, Segments8Meas: len(measSegs),
			MissingCells: len(diff.Missing), ExtraCells: len(diff.Extra), Match: diff.Match(),
			MismatchCoords: mc, NearRadius: 1, NearViolations: len(violations),
			BankCells: res.BankCells, ClippedCells: res.Clipped,
			MaxDrive: res.MaxDrive, MaxCut: res.MaxCut, TotalCut: res.TotalCut,
			MinCoreHeight0:    minCoreH,
			AnnotationPNG:     fmt.Sprintf("annotation_%s.png", p.Name),
			ComparisonCSV:     fmt.Sprintf("comparison_%s.csv", p.Name),
			MismatchCSV:       fmt.Sprintf("mismatches_%s.csv", p.Name),
			MismatchCoordsTXT: fmt.Sprintf("mismatch_coordinates_%s.txt", p.Name),
		}
		mf.Groups = append(mf.Groups, gm)

		// 全格逐格比对 CSV
		if err := writeComparisonCSV(filepath.Join(outDir, gm.ComparisonCSV),
			n, f, pred, predComp, meas, measComp, res.Cut); err != nil {
			return err
		}
		// 不一致格坐标 CSV / TXT（零行也要留带头文件）
		if err := writeMismatchCSV(filepath.Join(outDir, gm.MismatchCSV), n, diff, res.Cut); err != nil {
			return err
		}
		if err := writeMismatchTXT(filepath.Join(outDir, gm.MismatchCoordsTXT), p, n, diff); err != nil {
			return err
		}
		if err := writeF64(filepath.Join(outDir, fmt.Sprintf("height_after_%s.f64", p.Name)), res.H1); err != nil {
			return err
		}

		statLine := fmt.Sprintf("tauAcc>%d tauSlope>%.3f | core pred=%d meas=%d | seg8=%d | mismatch=%d (missing %d extra %d) | near1 violations=%d",
			th.Acc, th.Slope, predCells, gm.MeasuredCoreCells, predCount,
			len(mc), len(diff.Missing), len(diff.Extra), len(violations))
		pngPath := filepath.Join(outDir, gm.AnnotationPNG)
		if err := RenderAnnotationPNG(pngPath, h0, res.Cut, n, pred, zone, predComp, segs, diff,
			strings.ToUpper(p.Name), fmt.Sprintf("%.4f x%d rounds", p.K, p.Rounds), statLine, depthScale); err != nil {
			return err
		}

		// 根目录便捷副本（英文名 + 中文名各一张）
		asciiRoot := filepath.Join(rootDir, fmt.Sprintf("gully_annotation_%s.png", p.Name))
		var zhRoot string
		if p.Name == "weak" {
			zhRoot = filepath.Join(rootDir, "冲沟标注图_弱强度组.png")
		} else {
			zhRoot = filepath.Join(rootDir, "冲沟标注图_高强度组.png")
		}
		if err := copyFile(pngPath, asciiRoot); err != nil {
			return err
		}
		if err := copyFile(pngPath, zhRoot); err != nil {
			return err
		}
		rootPNGs[p.Name] = [2]string{asciiRoot, zhRoot}
	}

	// ---------- 5. 指标 JSON ----------
	jsonBytes, err := json.MarshalIndent(mf, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(outDir, "metrics.json"), jsonBytes, 0o644); err != nil {
		return err
	}
	if err := copyFile(filepath.Join(outDir, "metrics.json"), filepath.Join(rootDir, "gully_metrics.json")); err != nil {
		return err
	}

	// ---------- 6. 汇总表（指标各占一行）与中文报告 ----------
	if err := writeSummaryCSV(filepath.Join(outDir, "summary_table.csv"), mf); err != nil {
		return err
	}
	if err := writeGroupRowsCSV(filepath.Join(outDir, "summary_by_group.csv"), mf); err != nil {
		return err
	}
	if err := copyFile(filepath.Join(outDir, "summary_table.csv"), filepath.Join(rootDir, "gully_summary_table.csv")); err != nil {
		return err
	}
	if err := copyFile(filepath.Join(outDir, "summary_by_group.csv"), filepath.Join(rootDir, "gully_summary_by_group.csv")); err != nil {
		return err
	}
	md := buildMarkdown(mf)
	if err := os.WriteFile(filepath.Join(outDir, "report.md"), []byte(md), 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(rootDir, "gully_report.md"), []byte(md), 0o644); err != nil {
		return err
	}
	txt := buildPlainLines(mf)
	if err := os.WriteFile(filepath.Join(outDir, "summary_lines.txt"), []byte(txt), 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(rootDir, "gully_summary_lines.txt"), []byte(txt), 0o644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(rootDir, "两组强度核心格与连通段.txt"), []byte(txt), 0o644); err != nil {
		return err
	}

	// 同步独立证明文档（从仓库根运行时存在；缺失不影响主流程）。
	if data, err := os.ReadFile(filepath.Join("docs", "gully-core-proof.md")); err == nil {
		_ = os.WriteFile(filepath.Join(outDir, "gully-core-proof.md"), data, 0o644)
		_ = os.WriteFile(filepath.Join(rootDir, "gully_core_proof.md"), data, 0o644)
	}

	// 控制台也打印关键四行，便于直接看到
	fmt.Print(buildPlainLines(mf))
	fmt.Printf("\n产物目录: %s\n标注图: %s, %s\n", outDir,
		rootPNGs["weak"][1], rootPNGs["strong"][1])
	return nil
}

func countTrue(m []bool) int {
	c := 0
	for _, v := range m {
		if v {
			c++
		}
	}
	return c
}

func maskToFloat(m []bool) []float64 {
	v := make([]float64, len(m))
	for i, b := range m {
		if b {
			v[i] = 1
		}
	}
	return v
}

func writeF64(path string, v []float64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var buf [8]byte
	for _, x := range v {
		bits := math.Float64bits(x)
		for k := 0; k < 8; k++ {
			buf[k] = byte(bits >> (8 * k)) // little endian
		}
		if _, err := f.Write(buf[:]); err != nil {
			return err
		}
	}
	return nil
}

func writeCoreCellsCSV(path string, f *hydro.Field, th gully.Thresholds, comp []int) error {
	fp, err := os.Create(path)
	if err != nil {
		return err
	}
	defer fp.Close()
	w := csv.NewWriter(fp)
	defer w.Flush()
	_ = w.Write([]string{"segment8", "x", "y", "index", "acc", "slope", "slope_deg"})
	n := f.N
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			i := hydro.Idx(n, x, y)
			if comp[i] == 0 {
				continue
			}
			_ = w.Write([]string{
				itoa(comp[i]), itoa(x), itoa(y), itoa(i), itoa(f.Acc[i]),
				fmt.Sprintf("%.12f", f.Slope[i]),
				fmt.Sprintf("%.6f", f.Slope[i]*180/math.Pi),
			})
		}
	}
	return w.Error()
}

func writeSegmentsCSV(path string, segs []gully.Segment) error {
	fp, err := os.Create(path)
	if err != nil {
		return err
	}
	defer fp.Close()
	w := csv.NewWriter(fp)
	defer w.Flush()
	_ = w.Write([]string{"segment8", "count", "minX", "minY", "maxX", "maxY", "centroidX", "centroidY", "cells_index"})
	for _, s := range segs {
		cells := make([]string, 0, len(s.Cells))
		for _, i := range s.Cells {
			cells = append(cells, itoa(i))
		}
		_ = w.Write([]string{
			itoa(s.ID), itoa(s.Count), itoa(s.MinX), itoa(s.MinY), itoa(s.MaxX), itoa(s.MaxY),
			fmt.Sprintf("%.4f", s.CentroidX), fmt.Sprintf("%.4f", s.CentroidY),
			strings.Join(cells, " "),
		})
	}
	return w.Error()
}

func writeComparisonCSV(path string, n int, f *hydro.Field, pred []bool, predComp []int, meas []bool, measComp []int, cut []float64) error {
	fp, err := os.Create(path)
	if err != nil {
		return err
	}
	defer fp.Close()
	w := csv.NewWriter(fp)
	defer w.Flush()
	_ = w.Write([]string{"x", "y", "index", "acc", "slope", "pred_core", "segment8_pred", "cut", "meas_core", "segment8_meas", "match"})
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			i := hydro.Idx(n, x, y)
			match := pred[i] == meas[i]
			_ = w.Write([]string{
				itoa(x), itoa(y), itoa(i), itoa(f.Acc[i]), fmt.Sprintf("%.12f", f.Slope[i]),
				bool01(pred[i]), itoa(predComp[i]), fmt.Sprintf("%.12e", cut[i]),
				bool01(meas[i]), itoa(measComp[i]), bool01(match),
			})
		}
	}
	return w.Error()
}

func writeMismatchCSV(path string, n int, d gully.Diff, cut []float64) error {
	fp, err := os.Create(path)
	if err != nil {
		return err
	}
	defer fp.Close()
	w := csv.NewWriter(fp)
	defer w.Flush()
	_ = w.Write([]string{"x", "y", "index", "kind", "cut"})
	for _, i := range d.Missing {
		x, y := hydro.XY(n, i)
		_ = w.Write([]string{itoa(x), itoa(y), itoa(i), "missing_predicted_but_not_measured", fmt.Sprintf("%.12e", cut[i])})
	}
	for _, i := range d.Extra {
		x, y := hydro.XY(n, i)
		_ = w.Write([]string{itoa(x), itoa(y), itoa(i), "extra_measured_but_not_predicted", fmt.Sprintf("%.12e", cut[i])})
	}
	return w.Error()
}

func writeMismatchTXT(path string, p gully.ErodeParams, n int, d gully.Diff) error {
	var b strings.Builder
	fmt.Fprintf(&b, "不一致格坐标报告 —— %s（%s, K=%.4f, rounds=%d）\n", p.Label, p.Name, p.K, p.Rounds)
	b.WriteString("坐标格式 (x,y)，x 向右增、y 向下增，原点为画幅左上角；index=y*N+x。\n")
	fmt.Fprintf(&b, "逐格比对结果：missing(预判有/实测无)=%d，extra(实测有/预判无)=%d，合计不一致=%d。\n",
		len(d.Missing), len(d.Extra), len(d.Missing)+len(d.Extra))
	if len(d.Missing)+len(d.Extra) == 0 {
		b.WriteString("不一致格：无。全图所有格子预判与侵蚀后实测完全一致，因此没有需要上报的坐标。\n")
	} else {
		b.WriteString("不一致格坐标列表：\n")
		for _, i := range d.Missing {
			x, y := hydro.XY(n, i)
			fmt.Fprintf(&b, "  (x=%d,y=%d) index=%d kind=missing\n", x, y, i)
		}
		for _, i := range d.Extra {
			x, y := hydro.XY(n, i)
			fmt.Fprintf(&b, "  (x=%d,y=%d) index=%d kind=extra\n", x, y, i)
		}
	}
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func writeSummaryCSV(path string, mf metricsFile) error {
	fp, err := os.Create(path)
	if err != nil {
		return err
	}
	defer fp.Close()
	w := csv.NewWriter(fp)
	defer w.Flush()
	w.Write([]string{"指标", fmt.Sprintf("%s_K%.4f", mf.Groups[0].Label, mf.Groups[0].K),
		fmt.Sprintf("%s_K%.4f", mf.Groups[1].Label, mf.Groups[1].K)})
	w.Write([]string{"核心格数", itoa(mf.Groups[0].PredictedCoreCells), itoa(mf.Groups[1].PredictedCoreCells)})
	w.Write([]string{"连通段条数（八邻域）", itoa(mf.Groups[0].Segments8Pred), itoa(mf.Groups[1].Segments8Pred)})
	return w.Error()
}

// writeGroupRowsCSV 是另一种等价排版：每个强度组占一行，行内给出两个指标。
func writeGroupRowsCSV(path string, mf metricsFile) error {
	fp, err := os.Create(path)
	if err != nil {
		return err
	}
	defer fp.Close()
	w := csv.NewWriter(fp)
	defer w.Flush()
	w.Write([]string{"强度组", "K", "核心格数", "连通段条数（八邻域）", "逐格不一致数"})
	for _, g := range mf.Groups {
		w.Write([]string{g.Label, fmt.Sprintf("%.4f", g.K),
			itoa(g.PredictedCoreCells), itoa(g.Segments8Pred),
			itoa(g.MissingCells + g.ExtraCells)})
	}
	return w.Error()
}

func buildPlainLines(mf metricsFile) string {
	var b strings.Builder
	b.WriteString("两组强度核心统计（核心格数、连通段条数各占一行/组）\n")
	for _, g := range mf.Groups {
		fmt.Fprintf(&b, "%s（K=%.4f）核心格数：%d（预判 %d / 侵蚀后实测 %d）\n",
			g.Label, g.K, g.PredictedCoreCells, g.PredictedCoreCells, g.MeasuredCoreCells)
		fmt.Fprintf(&b, "%s（K=%.4f）连通段条数（八邻域）：%d（预判 %d / 实测 %d）\n",
			g.Label, g.K, g.Segments8Pred, g.Segments8Pred, g.Segments8Meas)
	}
	// 题目要求的“每个指标占一行、两强度为两列”的转置表纯文本版
	b.WriteString("\n指标\\强度\t弱强度组\t高强度组\n")
	b.WriteString(fmt.Sprintf("核心格数\t%d\t%d\n", mf.Groups[0].PredictedCoreCells, mf.Groups[1].PredictedCoreCells))
	b.WriteString(fmt.Sprintf("连通段条数（八邻域）\t%d\t%d\n", mf.Groups[0].Segments8Pred, mf.Groups[1].Segments8Pred))
	return b.String()
}

func copyFile(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0o644)
}

func itoa(i int) string { return fmt.Sprintf("%d", i) }
func bool01(b bool) string {
	if b {
		return "1"
	}
	return "0"
}
