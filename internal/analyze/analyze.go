// Package analyze 串起“谱预判 → 法线实测 → 逐箱比对”的完整流程并落盘产物。
package analyze

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"

	"goproc/api"
	"goproc/internal/normal"
	"goproc/internal/spectrum"
)

// ThresholdPP 是偏差标注阈值，单位百分点。
const ThresholdPP = 3.0

// Row 是一个角度箱的完整比对记录。
type Row struct {
	Index         int     `json:"index"`
	Label         string  `json:"label"`
	AngleLoDeg    float64 `json:"angle_lo_deg"`
	AngleHiDeg    float64 `json:"angle_hi_deg"`
	PredSpectral  float64 `json:"pred_spectral"`  // 二维谱矩高斯近似预判占比
	PredIsotropic float64 `json:"pred_isotropic"` // 谱斜率各向同性基线（1/nb）
	Measured      float64 `json:"measured"`       // 渲染后逐像素实测占比
	Count         int     `json:"count"`          // 实测像素数
	DevSpecPP     float64 `json:"dev_spec_pp"`    // 实测−谱预判，百分点
	DevIsoPP      float64 `json:"dev_iso_pp"`     // 实测−各向同性基线，百分点
	FlagSpec      bool    `json:"flag_spec"`      // |DevSpecPP|>3
	FlagIso       bool    `json:"flag_iso"`       // |DevIsoPP|>3
}

// Report 是一次分析的全部可序列化结果。
type Report struct {
	Options       api.Options       `json:"options"`
	InteriorPx    int               `json:"interior_px"`
	Beta          spectrum.SlopeFit `json:"beta_fit"`
	CovSpectral   [3]float64        `json:"cov_spectral"` // Mxx,Myy,Mxy（矩形窗）
	CovWindowed   [3]float64        `json:"cov_spectral_windowed"`
	CovEmpirical  [3]float64        `json:"cov_empirical"` // 有限差分
	Rows8         []Row             `json:"rows_8"`
	Rows16        []Row             `json:"rows_16"`
	FlaggedSpec8  []int             `json:"flagged_spec_8"`
	FlaggedIso8   []int             `json:"flagged_iso_8"`
	FlaggedSpec16 []int             `json:"flagged_spec_16"`
	FlaggedIso16  []int             `json:"flagged_iso_16"`
}

var compass8 = [8]string{"E", "NE", "N", "NW", "W", "SW", "S", "SE"}

func pct(v float64) float64 { return math.Round(v*100000) / 1000 } // 百分数保留三位小数（0.001%）

func buildRows(nb int, counts []int, measured, pred []float64) []Row {
	rows := make([]Row, nb)
	width := 360.0 / float64(nb)
	iso := 1 / float64(nb)
	for k := 0; k < nb; k++ {
		lo := float64(k) * width
		hi := lo + width
		label := fmt.Sprintf("[%.1f°,%.1f°)", lo, hi)
		if nb == 8 {
			label = fmt.Sprintf("%s %s", compass8[k], label)
		}
		ds := (measured[k] - pred[k]) * 100
		di := (measured[k] - iso) * 100
		rows[k] = Row{
			Index: k, Label: label, AngleLoDeg: lo, AngleHiDeg: hi,
			PredSpectral:  pred[k],
			PredIsotropic: iso,
			Measured:      measured[k],
			Count:         counts[k],
			DevSpecPP:     math.Round(ds*1000) / 1000,
			DevIsoPP:      math.Round(di*1000) / 1000,
			FlagSpec:      math.Abs(ds) > ThresholdPP,
			FlagIso:       math.Abs(di) > ThresholdPP,
		}
	}
	return rows
}

// Run 生成高度图并完成全套分析，返回报告；outDir 非空时写出 CSV/JSON。
func Run(opt api.Options, outDir string) (*Report, error) {
	h, err := api.Generate(opt)
	if err != nil {
		return nil, err
	}
	n := opt.Size
	field := normal.FromHeightmap(n, h.Data)

	// 预判：径向谱斜率 + 二维谱矩（矩形窗与 Hann 窗各一份，检验稳健性）。
	fit := spectrum.RadialSlope(n, h.Data, 2*math.Pi/float64(n)*2, math.Pi*0.8)
	mom := spectrum.SpectralMoments(n, h.Data, false)
	momW := spectrum.SpectralMoments(n, h.Data, true)
	exx, eyy, exy := field.Covariance()

	pred8 := spectrum.SectorProbabilities(8, mom.Mxx, mom.Myy, mom.Mxy)
	pred16 := spectrum.SectorProbabilities(16, mom.Mxx, mom.Myy, mom.Mxy)
	counts8, meas8 := field.Histogram(8)
	counts16, meas16 := field.Histogram(16)

	rep := &Report{
		Options:      opt,
		InteriorPx:   field.Interior,
		Beta:         fit,
		CovSpectral:  [3]float64{mom.Mxx, mom.Myy, mom.Mxy},
		CovWindowed:  [3]float64{momW.Mxx, momW.Myy, momW.Mxy},
		CovEmpirical: [3]float64{exx, eyy, exy},
		Rows8:        buildRows(8, counts8, meas8, pred8),
		Rows16:       buildRows(16, counts16, meas16, pred16),
	}
	for _, r := range rep.Rows8 {
		if r.FlagSpec {
			rep.FlaggedSpec8 = append(rep.FlaggedSpec8, r.Index)
		}
		if r.FlagIso {
			rep.FlaggedIso8 = append(rep.FlaggedIso8, r.Index)
		}
	}
	for _, r := range rep.Rows16 {
		if r.FlagSpec {
			rep.FlaggedSpec16 = append(rep.FlaggedSpec16, r.Index)
		}
		if r.FlagIso {
			rep.FlaggedIso16 = append(rep.FlaggedIso16, r.Index)
		}
	}
	if outDir != "" {
		if err := os.MkdirAll(outDir, 0o755); err != nil {
			return nil, err
		}
		if err := writeCSV(filepath.Join(outDir, "orientation_table_8bin.csv"), rep.Rows8); err != nil {
			return nil, err
		}
		if err := writeCSV(filepath.Join(outDir, "orientation_table_16bin.csv"), rep.Rows16); err != nil {
			return nil, err
		}
		raw, err := json.MarshalIndent(rep, "", "  ")
		if err != nil {
			return nil, err
		}
		if err := os.WriteFile(filepath.Join(outDir, "analysis_summary.json"), raw, 0o644); err != nil {
			return nil, err
		}
	}
	return rep, nil
}

func writeCSV(path string, rows []Row) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	// 加 UTF-8 BOM，方便表格软件直接识别中文环境下的数字格式。
	if _, err := f.Write([]byte{0xef, 0xbb, 0xbf}); err != nil {
		return err
	}
	w := csv.NewWriter(f)
	header := []string{
		"箱", "方位标签", "角度下限(°)", "角度上限(°)",
		"谱预判占比(%)", "各向同性基线(%)", "实测占比(%)", "实测像素数",
		"偏差_对谱预判(百分点)", "偏差_对基线(百分点)",
		"标注_对谱预判(>3pp)", "标注_对基线(>3pp)",
	}
	if err := w.Write(header); err != nil {
		return err
	}
	for _, r := range rows {
		rec := []string{
			strconv.Itoa(r.Index), r.Label,
			strconv.FormatFloat(r.AngleLoDeg, 'f', -1, 64),
			strconv.FormatFloat(r.AngleHiDeg, 'f', -1, 64),
			strconv.FormatFloat(pct(r.PredSpectral), 'f', 3, 64),
			strconv.FormatFloat(pct(r.PredIsotropic), 'f', 3, 64),
			strconv.FormatFloat(pct(r.Measured), 'f', 3, 64),
			strconv.Itoa(r.Count),
			strconv.FormatFloat(r.DevSpecPP, 'f', 3, 64),
			strconv.FormatFloat(r.DevIsoPP, 'f', 3, 64),
			flagText(r.FlagSpec), flagText(r.FlagIso),
		}
		if err := w.Write(rec); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

func flagText(b bool) string {
	if b {
		return "是"
	}
	return "否"
}
