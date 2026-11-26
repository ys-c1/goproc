// Command lodreport 生成地形 LOD 逐级网格、按两组误差阈值选级，
// 逐层级对比对“格距平方律预判比值 / 边界修正式 / 实际有效顶点比”，
// 输出 JSON 明细与每组一份 CSV 比值表。
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
	"strings"

	"goproc/api"
	"goproc/internal/lod"
)

const (
	gridSize  = 257 // 256 个格距区间，可二分到 2×2 顶点
	seed      = uint64(7)
	frequency = 0.02
)

// ThresholdSet 是一组误差阈值（归一化高程单位的最大插值误差容限）。
type ThresholdSet struct {
	ID         string
	Name       string
	SwitchMode string
	Thresholds []float64
}

// 两组阈值在首次测得 errMax 序列后，用圆整数值标定：
// A 组每档恰好推进一个二分级（覆盖全部相邻层级对）；
// B 组档距更宽，产生跨级直切（层级差 d>1）。
var thresholdSets = []ThresholdSet{
	{
		ID:         "A",
		Name:       "逐级严格组",
		SwitchMode: "每档误差容限只放宽到下一个二分级，切换均为相邻层级对 d=1",
		// 圆整 bracketing 实测 errMax：0,.0113,.0340,.0928,.1759,.3877,.5554,.6234,.8518
		Thresholds: []float64{0, 0.012, 0.04, 0.1, 0.2, 0.4, 0.58, 0.7, 0.9},
	},
	{
		ID:         "B",
		Name:       "跨级宽松组",
		SwitchMode: "宽档误差容限，只保留偶数级，一次跨两级直切 d=2",
		// 分别 bracket L2/L4/L6/L8 的实测 errMax
		Thresholds: []float64{0, 0.05, 0.25, 0.6, 1.0},
	},
}

type Config struct {
	Size      int     `json:"size"`
	Seed      uint64  `json:"seed"`
	Octaves   int     `json:"octaves"`
	Frequency float64 `json:"frequency"`
	Cell      float64 `json:"cell"`
	Relief    float64 `json:"relief"`
	SideCells int     `json:"side_cells"`
	MaxLevel  int     `json:"max_level"`
}

type LevelRecord struct {
	Level             int     `json:"level"`
	Step              int     `json:"step_cells"`
	Spacing           float64 `json:"spacing"`
	SideVerts         int     `json:"side_verts"`
	PoolVertices      int     `json:"pool_vertices"`
	EffectiveVertices int     `json:"effective_vertices"`
	EffectiveAltCount int     `json:"effective_vertices_crosscheck"`
	Triangles         int     `json:"triangles"`
	ErrMax            float64 `json:"err_max_norm"`
	ErrRMS            float64 `json:"err_rms_norm"`
}

type Selection struct {
	Threshold float64 `json:"threshold"`
	Level     int     `json:"selected_level"`
}

type PairRow struct {
	SetID          string   `json:"set_id"`
	Pair           string   `json:"pair"`
	FineLevel      int      `json:"fine_level"`
	CoarseLevel    int      `json:"coarse_level"`
	Gap            int      `json:"level_gap"`
	RowKind        string   `json:"row_kind"` // switch_adjacent | switch_skip | reference_adjacent
	SpacingRatio   float64  `json:"spacing_ratio"`
	PredRatioBulk  float64  `json:"predicted_ratio_bulk"`
	PredRatioExact float64  `json:"predicted_ratio_boundary_exact"`
	EffFine        int      `json:"eff_fine"`
	EffCoarse      int      `json:"eff_coarse"`
	ActualRatio    float64  `json:"actual_ratio"`
	DevBulkPct     float64  `json:"deviation_bulk_pct"`
	DevExactPct    float64  `json:"deviation_exact_pct"`
	Exceeds5Pct    bool     `json:"exceeds_5pct"`
	PredEffCoarse  float64  `json:"predicted_eff_coarse_bulk"`
	PredDropPct    float64  `json:"predicted_jump_drop_pct"`
	ActualDropPct  float64  `json:"actual_jump_drop_pct"`
	ThresholdFrom  *float64 `json:"threshold_from"`
	ThresholdTo    *float64 `json:"threshold_to"`
	Explanation    string   `json:"explanation"`
}

type SetReport struct {
	ID          string      `json:"id"`
	Name        string      `json:"name"`
	SwitchMode  string      `json:"switch_mode"`
	Thresholds  []float64   `json:"thresholds"`
	Selections  []Selection `json:"selections"`
	ChainLevels []int       `json:"chain_levels"`
	Rows        []PairRow   `json:"rows"`
	CSVFile     string      `json:"csv_file"`
	ChartFile   string      `json:"chart_file"`
}

type Report struct {
	Config Config        `json:"config"`
	Levels []LevelRecord `json:"levels"`
	Sets   []SetReport   `json:"sets"`
}

func explanation(gap int, exceeds bool, devExact float64) string {
	if !exceeds {
		return "体密度律 E∝1/h² 在细级占主导，边界一行/列可忽略，预判偏差≤5%"
	}
	s := "有限网格边界项所致：每边顶点 n_l=W/2^l+1，粗级边界顶点占比上升，" +
		"实际 E_i/E_j 小于 4^d；改用边界修正式 [(W/2^i+1)/(W/2^j+1)]² 后"
	if math.Abs(devExact) < 1e-9 {
		s += "偏差为 0（浮点精度内），无需另做逐级简化即可解释并预判"
	} else {
		s += fmt.Sprintf("残差仅 %.4f%%", devExact)
	}
	if gap > 1 {
		s += fmt.Sprintf("；本对为跨 %d 级直切，体密度预判比值是 4^%d=%g，边界绝对量随层级差放大", gap, gap, math.Pow(4, float64(gap)))
	}
	return s
}

func main() {
	outDir := flag.String("out", "artifacts", "产物目录")
	flag.Parse()
	dir := *outDir

	h, err := api.Generate(api.Options{
		Size: gridSize, Octaves: 4, Lacunarity: 2.0, Gain: 0.5,
		Seed: seed, Frequency: frequency,
	})
	if err != nil {
		fatal(err)
	}
	g, err := lod.NewGrid(h.Data, gridSize, 1.0, 1.0)
	if err != nil {
		fatal(err)
	}

	maxLevel := g.MaxLevel()
	levels := make([]*lod.Level, maxLevel+1)
	recs := make([]LevelRecord, maxLevel+1)
	errMax := make([]float64, maxLevel+1)
	for l := 0; l <= maxLevel; l++ {
		lv, err := g.BuildLevel(l)
		if err != nil {
			fatal(err)
		}
		levels[l] = lv
		errMax[l] = lv.ErrMax
		eff := lv.Effective()
		eff2 := lod.EffectiveVerticesSorted(lv.Indices)
		recs[l] = LevelRecord{
			Level: l, Step: lv.Step, Spacing: g.Spacing(l),
			SideVerts: lv.SideVerts, PoolVertices: g.PoolVertices(),
			EffectiveVertices: eff, EffectiveAltCount: eff2,
			Triangles: lv.Triangles(), ErrMax: lv.ErrMax, ErrRMS: lv.ErrRMS,
		}
		fmt.Printf("L%d step=%3d side=%3d eff=%6d tris=%7d errMax=%.6f errRMS=%.6f\n",
			l, lv.Step, lv.SideVerts, eff, lv.Triangles(), lv.ErrMax, lv.ErrRMS)
	}

	rep := Report{
		Config: Config{
			Size: gridSize, Seed: seed, Octaves: 4, Frequency: frequency,
			Cell: 1.0, Relief: 1.0, SideCells: g.SideCells(), MaxLevel: maxLevel,
		},
		Levels: recs,
	}

	if err := os.MkdirAll(dir, 0o755); err != nil {
		fatal(err)
	}

	for _, ts := range thresholdSets {
		sr := SetReport{
			ID: ts.ID, Name: ts.Name, SwitchMode: ts.SwitchMode,
			Thresholds: ts.Thresholds,
			CSVFile:    fmt.Sprintf("lod_ratio_table_%s.csv", strings.ToLower(ts.ID)),
			ChartFile:  fmt.Sprintf("lod_switch_%s.png", strings.ToLower(ts.ID)),
		}
		chain := make([]int, 0, len(ts.Thresholds))
		for _, t := range ts.Thresholds {
			l := lod.SelectLevel(errMax, t)
			sr.Selections = append(sr.Selections, Selection{Threshold: t, Level: l})
			if len(chain) == 0 || chain[len(chain)-1] != l {
				chain = append(chain, l)
			}
		}
		sr.ChainLevels = chain

		switchPairs := map[[2]int]bool{}
		type tPair struct{ i, j int }
		var tps []tPair
		for k := 0; k+1 < len(chain); k++ {
			i, j := chain[k], chain[k+1]
			p := [2]int{i, j}
			if !switchPairs[p] {
				switchPairs[p] = true
				tps = append(tps, tPair{i, j})
			}
		}
		threshBetween := func(i, j int) (*float64, *float64) {
			ptr := func(v float64) *float64 { return &v }
			// 找到触发 i->j 的相邻阈值档
			for k := 0; k+1 < len(sr.Selections); k++ {
				if sr.Selections[k].Level == i && sr.Selections[k+1].Level == j {
					return ptr(sr.Selections[k].Threshold), ptr(sr.Selections[k+1].Threshold)
				}
			}
			// 跨多档重复同级时，定位离开 i 与到达 j 的档
			for k := 0; k+1 < len(sr.Selections); k++ {
				a, b := sr.Selections[k], sr.Selections[k+1]
				if a.Level == i && b.Level > i {
					return ptr(a.Threshold), ptr(b.Threshold)
				}
			}
			return nil, nil
		}

		// 1) 全部相邻层级对（完整的“相邻两级”比对基线）
		for i := 0; i < maxLevel; i++ {
			j := i + 1
			kind := "reference_adjacent"
			if switchPairs[[2]int{i, j}] {
				kind = "switch_adjacent"
			}
			row := buildRow(ts.ID, kind, i, j, levels, g.SideCells())
			tf, tt := threshBetween(i, j)
			row.ThresholdFrom, row.ThresholdTo = tf, tt
			row.Explanation = explanation(1, row.Exceeds5Pct, row.DevExactPct)
			sr.Rows = append(sr.Rows, row)
		}
		// 2) 本组阈值实际产生的跨级直切对
		for _, p := range tps {
			if p.j == p.i+1 {
				continue // 已在相邻基线中
			}
			row := buildRow(ts.ID, "switch_skip", p.i, p.j, levels, g.SideCells())
			row.ThresholdFrom, row.ThresholdTo = threshBetween(p.i, p.j)
			row.Explanation = explanation(p.j-p.i, row.Exceeds5Pct, row.DevExactPct)
			sr.Rows = append(sr.Rows, row)
		}

		if err := writeCSV(filepath.Join(dir, sr.CSVFile), sr.Rows); err != nil {
			fatal(err)
		}
		rep.Sets = append(rep.Sets, sr)
		fmt.Printf("set %s chain=%v rows=%d -> %s\n", ts.ID, chain, len(sr.Rows), sr.CSVFile)
	}

	b, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "lod_data.json"), b, 0o644); err != nil {
		fatal(err)
	}
	fmt.Println("wrote", filepath.Join(dir, "lod_data.json"))
}

func buildRow(setID, kind string, i, j int, levels []*lod.Level, w int) PairRow {
	ei, ej := levels[i].Effective(), levels[j].Effective()
	actual := float64(ei) / float64(ej)
	pred := lod.PredictedRatio(i, j)
	exact := lod.ExactGridRatio(w, i, j)
	devBulk := math.Abs(pred-actual) / actual * 100
	devExact := math.Abs(exact-actual) / actual * 100
	predEj := float64(ei) / pred
	return PairRow{
		SetID: setID,
		Pair:  fmt.Sprintf("L%d→L%d", i, j), FineLevel: i, CoarseLevel: j, Gap: j - i,
		RowKind: kind, SpacingRatio: lod.SpacingRatio(i, j),
		PredRatioBulk: pred, PredRatioExact: exact,
		EffFine: ei, EffCoarse: ej, ActualRatio: actual,
		DevBulkPct: devBulk, DevExactPct: devExact, Exceeds5Pct: devBulk > 5.0,
		PredEffCoarse: predEj,
		PredDropPct:   (1 - 1/pred) * 100,
		ActualDropPct: (1 - float64(ej)/float64(ei)) * 100,
	}
}

var csvHeader = []string{
	"set_id", "pair", "fine_level", "coarse_level", "level_gap", "row_kind",
	"spacing_ratio_hj_over_hi", "predicted_ratio_bulk_4powd",
	"predicted_ratio_boundary_exact", "effective_vertices_fine", "effective_vertices_coarse",
	"actual_ratio_eff_fine_over_coarse", "deviation_bulk_pct", "deviation_boundary_exact_pct",
	"exceeds_5pct", "predicted_eff_coarse_from_fine", "predicted_jump_drop_pct",
	"actual_jump_drop_pct", "threshold_from", "threshold_to", "explanation",
}

func writeCSV(path string, rows []PairRow) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	if err := w.Write(csvHeader); err != nil {
		return err
	}
	for _, r := range rows {
		rec := []string{
			r.SetID, r.Pair, strconv.Itoa(r.FineLevel), strconv.Itoa(r.CoarseLevel),
			strconv.Itoa(r.Gap), r.RowKind,
			fmt.Sprintf("%.6g", r.SpacingRatio),
			fmt.Sprintf("%.6f", r.PredRatioBulk),
			fmt.Sprintf("%.9f", r.PredRatioExact),
			strconv.Itoa(r.EffFine), strconv.Itoa(r.EffCoarse),
			fmt.Sprintf("%.9f", r.ActualRatio),
			fmt.Sprintf("%.4f", r.DevBulkPct),
			fmt.Sprintf("%.2e", r.DevExactPct),
			strconv.FormatBool(r.Exceeds5Pct),
			fmt.Sprintf("%.3f", r.PredEffCoarse),
			fmt.Sprintf("%.4f", r.PredDropPct),
			fmt.Sprintf("%.4f", r.ActualDropPct),
			ptrFmt(r.ThresholdFrom), ptrFmt(r.ThresholdTo),
			r.Explanation,
		}
		if err := w.Write(rec); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

func ptrFmt(p *float64) string {
	if p == nil {
		return ""
	}
	return fmt.Sprintf("%.6g", *p)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "lodreport:", err)
	os.Exit(1)
}
