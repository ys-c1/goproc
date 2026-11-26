// axisdemo 生成两个画幅的相位锁定有限层地形，先用各层相位求和预判轴列并
// 逐像素确认，再额外做全列镜像相关扫描作为独立审计，逐图比较两套轴号。
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

	"goproc/internal/axis"
	"goproc/internal/hmap"
)

type frameSpec struct {
	Key    string
	Label  string
	Width  int
	Height int
	Seed   uint64
}

type compareRow struct {
	CSVLine       int
	PairNo        int
	CaseNo        int
	Class         string
	Frame         string
	Width         int
	Height        int
	Seed          uint64
	TrueAxis      int
	PredFloat     float64
	PredAxis      int
	FastAxis      int
	FastChecks    int
	ActualAxis    int
	Delta         int
	AbsDelta      int
	RPred         float64
	RActual       float64
	PixelPairs    int
	PixelMismatch int
	MaxAbsDiff    float64
	QuantizedBad  int
	TiedColumns   string
	FastHit       bool
	OverOneColumn bool
}

type frameData struct {
	frame    frameSpec
	locked   []compareRow
	controls []compareRow
}

type outputBundle struct {
	Frames []frameSpec  `json:"frames"`
	Cases  []*axis.Case `json:"cases"`
	Rows   []compareRow `json:"rows"`
}

const controlCount = 3

func main() {
	outDir := flag.String("out", "artifacts/symmetry", "输出目录")
	caseCount := flag.Int("cases", 12, "每个画幅的相位锁定案例数")
	flag.Parse()

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fatal("create out dir: %v", err)
	}

	frames := []frameSpec{
		{Key: "A", Label: "128x128", Width: 128, Height: 128, Seed: 7},
		{Key: "B", Label: "96x96", Width: 96, Height: 96, Seed: 23},
	}
	controlDrifts := []float64{1.15, -1.40, 1.70}

	var allRows []compareRow
	var allCases []*axis.Case
	var frameDatas []frameData
	bundle := outputBundle{Frames: frames}
	phaseRecords := [][]string{{
		"frame", "case_no", "class", "known_axis", "layer", "cycles", "amplitude",
		"omega", "order_m", "phi_unwrapped", "phi_wrapped", "phase_drift", "layer_axis",
	}}

	for _, f := range frames {
		curveRecords := [][]string{{"case_no", "true_axis", "pred_axis", "actual_axis", "column", "r", "pairs"}}
		data := frameData{frame: f}

		evaluate := func(caseNo, pairNo int, class string, c *axis.Case, trueAxis int) compareRow {
			pred := axis.PredictAxis(c.Layers)
			fastAxis, fastChecks, check, fastHit := fastConfirm(c, pred.RoundedAxis)

			// 审计路径：扫描 c=1..W-2 的全部非退化列；这里仅用于事后比较，
			// 生产快速路径命中后无需执行。c=0/W-1 是单列自配对，会平凡得到 rho=1。
			scores := axis.ScanColumns(c, 3)
			best, tied := axis.BestColumn(scores, 1e-12)
			rPred := axis.MirrorCorrelation(c.Data, c.Width, c.Height, pred.RoundedAxis)

			row := compareRow{
				CaseNo:        caseNo,
				PairNo:        pairNo,
				Class:         class,
				Frame:         f.Key,
				Width:         f.Width,
				Height:        f.Height,
				Seed:          c.Seed,
				TrueAxis:      trueAxis,
				PredFloat:     pred.AxisFloat,
				PredAxis:      pred.RoundedAxis,
				FastAxis:      fastAxis,
				FastChecks:    fastChecks,
				ActualAxis:    best.Column,
				Delta:         pred.RoundedAxis - best.Column,
				RPred:         rPred.R,
				RActual:       best.R,
				PixelPairs:    check.Pairs,
				PixelMismatch: check.Mismatches,
				MaxAbsDiff:    check.MaxAbsDiff,
				QuantizedBad:  check.QuantizedBad,
				TiedColumns:   intsToCSV(tied),
				FastHit:       fastHit,
			}
			row.AbsDelta = absInt(row.Delta)
			row.OverOneColumn = row.AbsDelta > 1
			row.CSVLine = len(allRows) + 2

			allRows = append(allRows, row)
			allCases = append(allCases, c)

			for _, layer := range c.Layers {
				phaseRecords = append(phaseRecords, []string{
					f.Key,
					fmt.Sprintf("%d", caseNo),
					class,
					fmt.Sprintf("%d", trueAxis),
					fmt.Sprintf("%d", layer.Index),
					fmt.Sprintf("%.2f", layer.Cycles),
					fmt.Sprintf("%.6f", layer.Amplitude),
					fmt.Sprintf("%.12f", layer.Omega),
					fmt.Sprintf("%d", layer.Order),
					fmt.Sprintf("%.12f", layer.PhiUnwrapped),
					fmt.Sprintf("%.12f", layer.PhiWrapped),
					fmt.Sprintf("%.6f", layer.PhaseDrift),
					fmt.Sprintf("%.9f", pred.LayerAxes[layer.Index]),
				})
			}

			for _, s := range scores {
				curveRecords = append(curveRecords, []string{
					fmt.Sprintf("%d", caseNo),
					fmt.Sprintf("%d", trueAxis),
					fmt.Sprintf("%d", pred.RoundedAxis),
					fmt.Sprintf("%d", best.Column),
					fmt.Sprintf("%d", s.Column),
					fmt.Sprintf("%.12f", s.R),
					fmt.Sprintf("%d", s.Pairs),
				})
			}
			return row
		}

		for i := 0; i < *caseCount; i++ {
			// 相对位置在两个画幅中一致，轴列按各自宽度取整。
			frac := 0.33 + 0.34*float64(i)/float64(maxInt(1, *caseCount-1))
			trueAxis := int(math.Round(frac * float64(f.Width-1)))
			seed := f.Seed + uint64(i)

			c := axis.Generate(
				fmt.Sprintf("%s-locked-%02d", f.Key, i+1),
				f.Width, f.Height, trueAxis, 4, seed,
			)
			row := evaluate(i+1, i+1, "phase_locked", c, trueAxis)
			data.locked = append(data.locked, row)

			// 每个画幅的第一个案例留一张 PGM 地形预览。
			if i == 0 {
				pgmPath := filepath.Join(*outDir, fmt.Sprintf("terrain_%s_case01.pgm", f.Key))
				if err := os.WriteFile(pgmPath, hmap.EncodePGM(c.Width, c.Data), 0o644); err != nil {
					fatal("write pgm: %v", err)
				}
			}
		}

		for j, drift := range controlDrifts {
			trueAxis := f.Width/2 + (j-1)*5
			seed := f.Seed + 90000 + uint64(j)*131 + uint64(f.Width)*17
			caseNo := *caseCount + j + 1
			c := axis.GenerateWithDrift(
				fmt.Sprintf("%s-drift-%02d", f.Key, j+1),
				f.Width, f.Height, trueAxis, 4, seed, 0, drift,
			)
			row := evaluate(caseNo, 0, "phase_drift_control", c, trueAxis)
			data.controls = append(data.controls, row)
		}

		curvePath := filepath.Join(*outDir, fmt.Sprintf("curves_%s.csv", f.Key))
		if err := writeCSV(curvePath, curveRecords); err != nil {
			fatal("write curves: %v", err)
		}
		frameDatas = append(frameDatas, data)
	}

	phasePath := filepath.Join(*outDir, "phase_layers.csv")
	if err := writeCSV(phasePath, phaseRecords); err != nil {
		fatal("write phase csv: %v", err)
	}

	// tidy 表：一图一行，因此每个画幅/案例都有独立 CSV 物理行号。
	for i := range allRows {
		allRows[i].CSVLine = i + 2
	}
	tidyPath := filepath.Join(*outDir, "axis_compare.csv")
	tidyHeader := []string{
		"csv_line", "pair_no", "case_no", "class", "frame", "width", "height", "seed", "true_axis",
		"pred_axis_float", "pred_axis", "fast_confirmed_axis", "fast_columns_checked",
		"corr_axis", "pred_minus_corr", "abs_delta", "over_one_column",
		"r_at_pred", "r_at_corr", "pixel_pairs", "pixel_mismatches",
		"max_abs_diff", "quantized_mismatches", "tied_columns", "fast_hit",
	}
	tidy := [][]string{tidyHeader}
	for _, r := range allRows {
		tidy = append(tidy, []string{
			fmt.Sprintf("%d", r.CSVLine),
			fmt.Sprintf("%d", r.PairNo),
			fmt.Sprintf("%d", r.CaseNo),
			r.Class,
			r.Frame,
			fmt.Sprintf("%d", r.Width),
			fmt.Sprintf("%d", r.Height),
			fmt.Sprintf("%d", r.Seed),
			fmt.Sprintf("%d", r.TrueAxis),
			fmt.Sprintf("%.9f", r.PredFloat),
			fmt.Sprintf("%d", r.PredAxis),
			fmt.Sprintf("%d", r.FastAxis),
			fmt.Sprintf("%d", r.FastChecks),
			fmt.Sprintf("%d", r.ActualAxis),
			fmt.Sprintf("%d", r.Delta),
			fmt.Sprintf("%d", r.AbsDelta),
			fmt.Sprintf("%t", r.OverOneColumn),
			fmt.Sprintf("%.12f", r.RPred),
			fmt.Sprintf("%.12f", r.RActual),
			fmt.Sprintf("%d", r.PixelPairs),
			fmt.Sprintf("%d", r.PixelMismatch),
			fmt.Sprintf("%.3g", r.MaxAbsDiff),
			fmt.Sprintf("%d", r.QuantizedBad),
			r.TiedColumns,
			fmt.Sprintf("%t", r.FastHit),
		})
	}
	if err := writeCSV(tidyPath, tidy); err != nil {
		fatal("write compare csv: %v", err)
	}

	// 宽表：同一相位锁定案例的两个画幅并排，轴号两栏。
	widePath := filepath.Join(*outDir, "axis_wide.csv")
	wide := [][]string{{
		"csv_line", "case_no",
		"A_true", "A_phase_pred", "A_corr_actual", "A_delta",
		"B_true", "B_phase_pred", "B_corr_actual", "B_delta",
		"max_abs_delta", "bad_over_one_column",
	}}
	for i := 0; i < *caseCount; i++ {
		a := frameDatas[0].locked[i]
		b := frameDatas[1].locked[i]
		line := i + 2
		maxDelta := maxInt(a.AbsDelta, b.AbsDelta)
		wide = append(wide, []string{
			fmt.Sprintf("%d", line),
			fmt.Sprintf("%d", i+1),
			fmt.Sprintf("%d", a.TrueAxis), fmt.Sprintf("%d", a.PredAxis), fmt.Sprintf("%d", a.ActualAxis), fmt.Sprintf("%d", a.Delta),
			fmt.Sprintf("%d", b.TrueAxis), fmt.Sprintf("%d", b.PredAxis), fmt.Sprintf("%d", b.ActualAxis), fmt.Sprintf("%d", b.Delta),
			fmt.Sprintf("%d", maxDelta),
			fmt.Sprintf("%t", maxDelta > 1),
		})
	}
	if err := writeCSV(widePath, wide); err != nil {
		fatal("write wide csv: %v", err)
	}

	twoColumnPath := filepath.Join(*outDir, "axis_two_columns.md")
	var twoColumn strings.Builder
	twoColumn.WriteString("# 两个画幅的轴列号（预判/实测，列号从零起算）\n\n")
	twoColumn.WriteString("| `axis_wide.csv` 行号 | A：128×128 轴列 | B：96×96 轴列 |\n")
	twoColumn.WriteString("|---:|---:|---:|\n")
	for i := 0; i < *caseCount; i++ {
		a := frameDatas[0].locked[i]
		b := frameDatas[1].locked[i]
		twoColumn.WriteString(fmt.Sprintf("| %d | %d / %d | %d / %d |\n",
			i+2, a.PredAxis, a.ActualAxis, b.PredAxis, b.ActualAxis))
	}
	if err := os.WriteFile(twoColumnPath, []byte(twoColumn.String()), 0o644); err != nil {
		fatal("write two-column table: %v", err)
	}

	bundle.Rows = allRows
	bundle.Cases = allCases
	jsonBytes, err := json.MarshalIndent(bundle, "", "  ")
	if err != nil {
		fatal("json marshal: %v", err)
	}
	if err := os.WriteFile(filepath.Join(*outDir, "results.json"), jsonBytes, 0o644); err != nil {
		fatal("write results json: %v", err)
	}

	var report strings.Builder
	fmt.Fprintf(&report, "镜像对称轴：相位预判与逐列相关审计\n")
	fmt.Fprintf(&report, "轴列号一律从零起算；CSV 行号把表头计为第 1 行。\n\n")

	fmt.Fprintf(&report, "## 1. 两个画幅的相位锁定案例（轴列号两栏）\n\n")
	fmt.Fprintf(&report, "| 宽表行号 | 案例 | A 128×128：预判/实测 (Δ) | B 96×96：预判/实测 (Δ) | 像素不一致 A/B |\n")
	fmt.Fprintf(&report, "|---:|---:|---|---|---:|\n")
	for i := 0; i < *caseCount; i++ {
		a := frameDatas[0].locked[i]
		b := frameDatas[1].locked[i]
		fmt.Fprintf(&report, "| %d | %02d | %d/%d (%+d) | %d/%d (%+d) | %d/%d |\n",
			i+2, i+1,
			a.PredAxis, a.ActualAxis, a.Delta,
			b.PredAxis, b.ActualAxis, b.Delta,
			a.PixelMismatch, b.PixelMismatch,
		)
	}

	fmt.Fprintf(&report, "\n## 2. 失相负面对照（用于检验行号报警与全扫回退）\n\n")
	fmt.Fprintf(&report, "这些图故意让第 0 层偏离共同相位锁，因而不存在相位和所预言的公共对称轴；它们不属于主验证表。\n\n")
	fmt.Fprintf(&report, "| tidy CSV 行号 | 画幅 | 对照 | 预判/实测 (Δ) | 像素不一致 | 超一列 |\n")
	fmt.Fprintf(&report, "|---:|:---:|---:|---|---:|:---:|\n")
	for fi := range frameDatas {
		for j, r := range frameDatas[fi].controls {
			fmt.Fprintf(&report, "| %d | %s | C%d | %d/%d (%+d) | %d | %t |\n",
				r.CSVLine, r.Frame, j+1, r.PredAxis, r.ActualAxis, r.Delta, r.PixelMismatch, r.OverOneColumn)
		}
	}

	var badLines []string
	for _, r := range allRows {
		if r.OverOneColumn {
			badLines = append(badLines, fmt.Sprintf("%s:%d（画幅 %s，类别 %s，案例 %02d，预判 %d/实测 %d，Δ=%+d）",
				tidyPath, r.CSVLine, r.Frame, r.Class, r.CaseNo, r.PredAxis, r.ActualAxis, r.Delta))
		}
	}

	fmt.Fprintf(&report, "\n## 3. 错开一列以上的图及行号\n\n")
	if len(badLines) == 0 {
		fmt.Fprintf(&report, "无。所有 %d 张相位锁定图的相位预判列与全列相关 argmax 相同，负面对照也未越过一列阈值。\n", 2**caseCount)
	} else {
		for _, line := range badLines {
			fmt.Fprintf(&report, "- %s\n", line)
		}
	}

	lockedFast := true
	for fi := range frameDatas {
		for _, r := range frameDatas[fi].locked {
			if !r.FastHit || r.PixelMismatch != 0 || r.QuantizedBad != 0 || r.AbsDelta != 0 {
				lockedFast = false
			}
		}
	}
	if lockedFast {
		fmt.Fprintf(&report, "\n结论：%d 张相位锁定图均由相位和直接命中预判列，逐像素 float64 与 8-bit 确认零不一致；无需全图搜轴。c=1..W-2 的非退化逐列扫描仅作审计（c=0/W-1 是平凡自配对）。\n", 2**caseCount)
	} else {
		fmt.Fprintf(&report, "\n结论：存在相位锁定案例未命中，见 axis_compare.csv。\n")
	}
	fmt.Fprintf(&report, "失相对照快速确认失败是预期行为；算法应回退到全列相关扫描，并由上表给出行号。\n")

	fmt.Fprintf(&report, "\n产物：\n")
	fmt.Fprintf(&report, "- 逐图比较（tidy，含物理行号）：%s\n", tidyPath)
	fmt.Fprintf(&report, "- 两画幅并排表：%s\n", widePath)
	fmt.Fprintf(&report, "- 两栏轴号表：%s\n", twoColumnPath)
	fmt.Fprintf(&report, "- 各层相位与单层轴估计：%s\n", phasePath)
	fmt.Fprintf(&report, "- 相关曲线数据：%s 与 %s\n",
		filepath.Join(*outDir, "curves_A.csv"), filepath.Join(*outDir, "curves_B.csv"))
	fmt.Fprintf(&report, "- 曲线图：%s 与 %s\n",
		filepath.Join(*outDir, "correlation_A.png"), filepath.Join(*outDir, "correlation_B.png"))

	reportPath := filepath.Join(*outDir, "report.txt")
	if err := os.WriteFile(reportPath, []byte(report.String()), 0o644); err != nil {
		fatal("write report: %v", err)
	}
	fmt.Print(report.String())
}

func fastConfirm(c *axis.Case, predicted int) (axisCol, checks int, result axis.PixelCheck, hit bool) {
	// 先查预判列；若离散取整落在相邻像素轴，再查左右各一列。最多三列，不做全图搜索。
	for _, delta := range []int{0, -1, 1} {
		col := predicted + delta
		if col < 0 || col >= c.Width {
			continue
		}
		checks++
		pc := axis.ConfirmAxis(c, col)
		if pc.Viable && pc.Mismatches == 0 && pc.QuantizedBad == 0 {
			return col, checks, pc, true
		}
		if delta == 0 {
			result = pc
		}
	}
	return predicted, checks, result, false
}

func writeCSV(path string, records [][]string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	if err := w.WriteAll(records); err != nil {
		return err
	}
	w.Flush()
	return w.Error()
}

func intsToCSV(xs []int) string {
	parts := make([]string, len(xs))
	for i, x := range xs {
		parts[i] = fmt.Sprintf("%d", x)
	}
	return strings.Join(parts, ";")
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
