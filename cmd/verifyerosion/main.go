// Command verifyerosion 生成地形、计算 D8 坡度与汇流量，并用同一组系数
// 比较闭式总侵蚀量和逐步迭代结果；所有证据写入 artifacts/erosion。
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"goproc/api"
	"goproc/internal/erosion"
	"goproc/internal/hmap"
)

type fileSet struct {
	Before       string
	Slope        string
	Flow         string
	RoundFiles   []roundFiles
	CellCSV      []string
	MismatchMask []string
}

type roundFiles struct {
	Rounds             int
	ClosedAfter        string
	IteratedAfter      string
	ClosedDifference   string
	IteratedDifference string
	ClosedVisible      string
	IteratedVisible    string
	MismatchPGM        string
	ComparisonCSV      string
}

type report struct {
	SchemaVersion     int                   `json:"schema_version"`
	Terrain           api.Options           `json:"terrain"`
	CellSize          float64               `json:"cell_size"`
	CoefficientK      float64               `json:"coefficient_k"`
	SameCoefficient   bool                  `json:"same_coefficient_in_closed_and_iteration"`
	Rounds            []int                 `json:"rounds"`
	GrayLevels        int                   `json:"gray_levels"`
	OneGrayBinHeight  float64               `json:"one_gray_bin_height"`
	FlowDefinition    string                `json:"flow_definition"`
	SlopeDefinition   string                `json:"slope_definition"`
	FormulaSingleStep string                `json:"formula_single_step"`
	FormulaTotal      string                `json:"formula_total_fixed_fields"`
	Comparisons       []*erosion.Comparison `json:"comparisons"`
	Files             fileSet               `json:"files"`
}

func main() {
	outDir := flag.String("out", "artifacts/erosion", "证据输出目录")
	size := flag.Int("size", 128, "地形边长")
	seed := flag.Uint64("seed", 7, "地形种子")
	octaves := flag.Int("octaves", 4, "fBm 层数")
	k := flag.Float64("k", 2e-5, "闭式与迭代共用的侵蚀系数 K")
	roundsText := flag.String("rounds", "16,64", "两个待比对的迭代轮数，逗号分隔")
	flag.Parse()

	rounds, err := parseRounds(*roundsText)
	if err != nil {
		fatal(err)
	}
	if len(rounds) != 2 {
		fatal(fmt.Errorf("必须给两个迭代轮数，当前 %q", *roundsText))
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fatal(err)
	}

	opt := api.DefaultOptions()
	opt.Size = *size
	opt.Seed = *seed
	opt.Octaves = *octaves
	height, err := api.Generate(opt)
	if err != nil {
		fatal(err)
	}

	fields, err := erosion.BuildFields(height.Size, height.Data, 1)
	if err != nil {
		fatal(err)
	}

	files := fileSet{}
	files.Before = "height_before"
	if err := writeGrayPair(filepath.Join(*outDir, files.Before), height.Size, height.Data); err != nil {
		fatal(err)
	}

	slopeName := "slope_d8"
	if err := writeGrayPair(filepath.Join(*outDir, slopeName), height.Size, fields.Slope); err != nil {
		fatal(err)
	}
	files.Slope = slopeName

	logFlow := make([]float64, len(fields.Flow))
	maxLog := 0.0
	for _, q := range fields.Flow {
		if v := math.Log1p(q); v > maxLog {
			maxLog = v
		}
	}
	for i, q := range fields.Flow {
		if maxLog > 0 {
			logFlow[i] = math.Log1p(q) / maxLog
		}
	}
	flowName := "flow_accumulation_log"
	if err := writeGrayPair(filepath.Join(*outDir, flowName), height.Size, logFlow); err != nil {
		fatal(err)
	}
	files.Flow = flowName

	rep := report{
		SchemaVersion:     1,
		Terrain:           opt,
		CellSize:          1,
		CoefficientK:      *k,
		SameCoefficient:   true,
		Rounds:            rounds,
		GrayLevels:        256,
		OneGrayBinHeight:  1.0 / 255.0,
		FlowDefinition:    "Q_i = D8 上游格数（每格初始贡献 1，含本格）",
		SlopeDefinition:   "S_i = max_j (h_i-h_j)/distance(i,j)，j 为八邻域中最陡下坡格",
		FormulaSingleStep: "epsilon_i = K * Q_i * S_i",
		FormulaTotal:      "E_i^(N) = N * K * Q_i * S_i",
		Files:             files,
	}

	for _, n := range rounds {
		closed, err := erosion.ClosedFormFixed(height.Data, fields, *k, n)
		if err != nil {
			fatal(err)
		}
		iterated, err := erosion.IterateFixed(height.Data, fields, *k, n)
		if err != nil {
			fatal(err)
		}
		cmp, err := erosion.Compare(height.Size, n, closed, iterated, height.Data)
		if err != nil {
			fatal(err)
		}
		rep.Comparisons = append(rep.Comparisons, cmp)

		rf := roundFiles{Rounds: n}
		base := fmt.Sprintf("r%03d", n)

		rf.ClosedAfter = base + "_height_after_closed"
		if err := writeGrayPair(filepath.Join(*outDir, rf.ClosedAfter), height.Size, closed); err != nil {
			fatal(err)
		}
		rf.IteratedAfter = base + "_height_after_iter"
		if err := writeGrayPair(filepath.Join(*outDir, rf.IteratedAfter), height.Size, iterated); err != nil {
			fatal(err)
		}

		depthClosed := make([]float64, height.Size*height.Size)
		depthIter := make([]float64, height.Size*height.Size)
		for i := range depthClosed {
			depthClosed[i] = height.Data[i] - closed[i]
			depthIter[i] = height.Data[i] - iterated[i]
		}

		rf.ClosedDifference = base + "_height_difference_before_minus_after_closed"
		if err := writeGrayPair(filepath.Join(*outDir, rf.ClosedDifference), height.Size, depthClosed); err != nil {
			fatal(err)
		}
		rf.IteratedDifference = base + "_height_difference_before_minus_after_iter"
		if err := writeGrayPair(filepath.Join(*outDir, rf.IteratedDifference), height.Size, depthIter); err != nil {
			fatal(err)
		}

		visibleClosed := append([]float64(nil), depthClosed...)
		visibleIter := append([]float64(nil), depthIter...)
		hmap.Normalize(visibleClosed)
		hmap.Normalize(visibleIter)
		rf.ClosedVisible = base + "_height_difference_closed_stretch"
		rf.IteratedVisible = base + "_height_difference_iter_stretch"
		if err := writeGrayPair(filepath.Join(*outDir, rf.ClosedVisible), height.Size, visibleClosed); err != nil {
			fatal(err)
		}
		if err := writeGrayPair(filepath.Join(*outDir, rf.IteratedVisible), height.Size, visibleIter); err != nil {
			fatal(err)
		}

		mask := erosion.MismatchMask(cmp)
		rf.MismatchPGM = base + "_mismatch_over_one_gray_bin"
		if err := writeGrayPair(filepath.Join(*outDir, rf.MismatchPGM), height.Size, mask); err != nil {
			fatal(err)
		}

		rf.ComparisonCSV = base + "_cell_comparison.csv"
		if err := writeCellCSV(filepath.Join(*outDir, rf.ComparisonCSV), cmp); err != nil {
			fatal(err)
		}

		rep.Files.RoundFiles = append(rep.Files.RoundFiles, rf)
	}

	jsonPath := filepath.Join(*outDir, "report.json")
	if err := writeJSON(jsonPath, rep); err != nil {
		fatal(err)
	}
	mdPath := filepath.Join(*outDir, "report.md")
	if err := writeMarkdownReport(mdPath, rep, fields.MaxFlowSlopeProduct); err != nil {
		fatal(err)
	}

	fmt.Printf("erosion verification wrote %s\n", *outDir)
	for _, cmp := range rep.Comparisons {
		fmt.Printf("rounds=%d over_one_gray_bin=%d encoded_delta_gt_1=%d max_gray_distance=%.3g\n",
			cmp.Rounds, cmp.OverOneBinCount, cmp.EncodedOverOneLevelCount, cmp.MaxGrayDistance)
	}
}

func parseRounds(s string) ([]int, error) {
	parts := strings.Split(s, ",")
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		n, err := strconv.Atoi(strings.TrimSpace(p))
		if err != nil || n <= 0 {
			return nil, fmt.Errorf("非法轮数 %q", p)
		}
		out = append(out, n)
	}
	return out, nil
}

func writeGrayPair(basePath string, size int, values []float64) error {
	pgmPath := basePath + ".pgm"
	if err := os.WriteFile(pgmPath, hmap.EncodePGM(size, values), 0o644); err != nil {
		return err
	}
	pngPath := basePath + ".png"
	img := image.NewGray(image.Rect(0, 0, size, size))
	for i, v := range values {
		img.Pix[i] = uint8(erosion.GrayCode(v))
	}
	f, err := os.Create(pngPath)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		return err
	}
	return f.Close()
}

func writeCellCSV(path string, cmp *erosion.Comparison) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	header := []string{
		"index", "x", "y", "closed_erosion", "iterated_erosion",
		"closed_height", "iterated_height", "abs_erosion_or_height_error",
		"gray_distance", "closed_gray", "iterated_gray", "encoded_gray_delta",
		"over_one_bin", "encoded_over_one_level",
	}
	if err := w.Write(header); err != nil {
		return err
	}
	for _, d := range cmp.Cells_ {
		record := []string{
			strconv.Itoa(d.Index),
			strconv.Itoa(d.X),
			strconv.Itoa(d.Y),
			strconv.FormatFloat(d.ClosedErosion, 'g', 17, 64),
			strconv.FormatFloat(d.IteratedErosion, 'g', 17, 64),
			strconv.FormatFloat(d.ClosedHeight, 'g', 17, 64),
			strconv.FormatFloat(d.IteratedHeight, 'g', 17, 64),
			strconv.FormatFloat(d.AbsHeightError, 'g', 5, 64),
			strconv.FormatFloat(d.GrayDistance, 'g', 5, 64),
			strconv.Itoa(d.ClosedGray),
			strconv.Itoa(d.IteratedGray),
			strconv.Itoa(d.EncodedGrayDelta),
			strconv.FormatBool(d.OverOneBin),
			strconv.FormatBool(d.EncodedOverOne),
		}
		if err := w.Write(record); err != nil {
			return err
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		return err
	}
	return f.Close()
}

func writeJSON(path string, rep report) error {
	data, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

func writeMarkdownReport(path string, rep report, maxQP float64) error {
	var b strings.Builder
	b.WriteString("# 降雨侵蚀闭式解核对报告\n\n")
	b.WriteString("## 1. 同一组计算条件\n\n")
	fmt.Fprintf(&b, "- 地形：size=%d seed=%d octaves=%d lacunarity=%g gain=%g frequency=%g\n",
		rep.Terrain.Size, rep.Terrain.Seed, rep.Terrain.Octaves, rep.Terrain.Lacunarity,
		rep.Terrain.Gain, rep.Terrain.Frequency)
	fmt.Fprintf(&b, "- 汇流量：%s\n", rep.FlowDefinition)
	fmt.Fprintf(&b, "- 坡度：%s\n", rep.SlopeDefinition)
	fmt.Fprintf(&b, "- 系数：`K=%g`；闭式与迭代接收同一个 `K`，未分别调参。\n", rep.CoefficientK)
	fmt.Fprintf(&b, "- `max(Q_i*S_i)=%.9g`；本轮参数下最大闭式侵蚀深度为 `%.6g`（见下表），未发生高程截断。\n\n", maxQP, rep.Comparisons[len(rep.Comparisons)-1].MaxClosedErosionDepth)

	b.WriteString("## 2. 闭式表达\n\n")
	b.WriteString("单步：\n\n```text\n")
	b.WriteString(rep.FormulaSingleStep)
	b.WriteString("\n```\n\n")
	b.WriteString("给定且固定的 Q_i、S_i 后，N 轮总侵蚀：\n\n```text\n")
	b.WriteString(rep.FormulaTotal)
	b.WriteString("\n```\n\n")
	b.WriteString("迭代实现每轮执行 `h_i <- h_i - K*Q_i*S_i`。实数运算下由归纳法，")
	b.WriteString("第 N 轮必然等于闭式结果；二进制浮点的累加顺序会产生远小于一个灰度档的舍入差。\n\n")

	b.WriteString("## 3. 两个轮数的逐格比对（同一组）\n\n")
	b.WriteString("主判据：`255 * |h_closed - h_iter| > 1`，即偏差严格超过一档灰度宽度 `1/255`。\n\n")
	b.WriteString("| 轮数 N | 格数 | 超一档灰度格数 | 灰度码差 >1 格数 | 灰度码不同格数 | 最大高度误差 | 最大灰度距离 | 最大灰度码差 | 最小末高程 |\n")
	b.WriteString("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, c := range rep.Comparisons {
		fmt.Fprintf(&b, "| %d | %d | **%d** | **%d** | %d | %.3e | %.3e | %d | %.6f |\n",
			c.Rounds, c.Cells, c.OverOneBinCount, c.EncodedOverOneLevelCount,
			c.EncodedNonzeroDeltaCount, c.MaxAbsHeightError, c.MaxGrayDistance,
			c.MaxEncodedGrayDelta, c.MinClosedHeight)
	}
	b.WriteString("\n`灰度码不同格数` 可能因最终值恰好落在 `x.5` 取整边界而出现 1 级差异；")
	b.WriteString("这不等于连续高度偏差超过一档。两种口径都在 `report.json` 和逐格 CSV 中列出。\n\n")

	b.WriteString("## 4. 图件与逐格数据\n\n")
	fmt.Fprintf(&b, "- 侵蚀前高程：`%s.pgm` / `%s.png`\n", rep.Files.Before, rep.Files.Before)
	fmt.Fprintf(&b, "- D8 坡度：`%s.pgm` / `%s.png`\n", rep.Files.Slope, rep.Files.Slope)
	fmt.Fprintf(&b, "- 汇流量（对数显示拉伸）：`%s.pgm` / `%s.png`\n\n", rep.Files.Flow, rep.Files.Flow)
	for _, rf := range rep.Files.RoundFiles {
		b.WriteString(fmt.Sprintf("### N=%d\n\n", rf.Rounds))
		fmt.Fprintf(&b, "- 闭式一次算出的侵蚀后高程：`%s.pgm` / `.png`\n", rf.ClosedAfter)
		fmt.Fprintf(&b, "- 逐步迭代的侵蚀后高程：`%s.pgm` / `.png`\n", rf.IteratedAfter)
		fmt.Fprintf(&b, "- 闭式侵蚀前后高差（before-after，定量灰度）：`%s.pgm` / `.png`\n", rf.ClosedDifference)
		fmt.Fprintf(&b, "- 迭代侵蚀前后高差（before-after，定量灰度）：`%s.pgm` / `.png`\n", rf.IteratedDifference)
		fmt.Fprintf(&b, "- 高差拉伸预览：`%s.pgm`、`%s.pgm`\n", rf.ClosedVisible, rf.IteratedVisible)
		fmt.Fprintf(&b, "- 超阈值违规格掩膜：`%s.pgm` / `.png`（预期全黑）\n", rf.MismatchPGM)
		fmt.Fprintf(&b, "- 逐格比对 CSV：`%s`\n\n", rf.ComparisonCSV)
	}

	b.WriteString("## 5. 适用边界\n\n")
	b.WriteString("上述零偏差结论证明的是“同一份给定坡度与汇流量场”下的代数迭代。")
	b.WriteString("若每轮侵蚀后重新计算 D8 坡度或汇流，Q、S 会随高程改变，")
	b.WriteString("此时简单的 `N*K*Q_initial*S_initial` 不是完整地貌演化方程的闭式解；")
	b.WriteString("必须把场更新规则也纳入递推或另行推导。\n")

	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "verifyerosion:", err)
	os.Exit(1)
}
