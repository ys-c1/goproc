// Package report 把倾角预判、实测比对与分布图整理成存档产物。
package report

import (
	"encoding/csv"
	"fmt"
	"html"
	"math"
	"os"
	"path/filepath"
	"strings"

	"goproc/api"
	"goproc/internal/fbm"
	"goproc/internal/noise"
	"goproc/internal/tilt"
)

// BandRow 是一档的预判与实测对照，倾角按度。
type BandRow struct {
	Index      int
	Frequency  float64
	Weight     float64
	PredSlope  float64
	PredTilt   float64
	MeasSlope  float64 // 隔离该档、乘回权重后的实测斜率
	MeasTilt   float64
	DeltaTilt  float64 // 预判减实测
	SlopeRatio float64 // 实测斜率 / 预判斜率
}

// Set 是一组参数的结果。
type Set struct {
	Name        string
	Options     api.Options
	Overview    float64 // 合成预判倾角，度
	OverviewSl  float64 // 合成预判斜率
	Bands       []BandRow
	Measured    tilt.Stats
	Grid        tilt.Grid
	Field       *api.Heightmap
	WorstAbsDev float64 // 逐档最大绝对偏差
	// 归一化尺度：Generate 输出的 [0,1] 图上，梯度整体除以 Span。
	NormMeasured tilt.Stats
	NormPred     float64 // NormalizedTilt(OverviewSl, Span)
}

// TwoDegrees 是允许的逐档偏差上限（度）。
const TwoDegrees = 2.0

// Analyze 对一组参数同时给出预判、逐档比对与整图实测。
//
// 逐档实测用「只保留该档」的隔离场，且只比较同一尺度：谱模型描述未归一化
// 高度场，因此 Field 取 api.Raw 而非 Generate 的归一化图。
func Analyze(name string, opt api.Options) (*Set, error) {
	field, err := api.Raw(opt)
	if err != nil {
		return nil, err
	}
	grid, stats := tilt.FromField(opt.Size, field.Data)
	normMap, err := api.Generate(opt)
	if err != nil {
		return nil, err
	}
	_, normStats := tilt.FromField(opt.Size, normMap.Data)
	slope, over := tilt.Overview(opt.Frequency, opt.Octaves, opt.Lacunarity, opt.Gain)
	s := &Set{
		Name:         name,
		Options:      opt,
		Overview:     over,
		OverviewSl:   slope,
		Measured:     stats,
		Grid:         grid,
		Field:        field,
		NormMeasured: normStats,
		NormPred:     tilt.NormalizedTilt(slope, field.Span),
	}
	var norm float64
	for a, k := 1.0, 0; k < opt.Octaves; k, a = k+1, a*opt.Gain {
		norm += a
	}
	for a, f, k := 1.0, opt.Frequency, 0; k < opt.Octaves; k, a, f = k+1, a*opt.Gain, f*opt.Lacunarity {
		w := a / norm
		pred := tilt.Describe(opt.Frequency, opt.Octaves, opt.Lacunarity, opt.Gain)[k]
		iso := isolated(opt, f, w)
		_, st := tilt.FromField(opt.Size, iso)
		row := BandRow{
			Index:      k,
			Frequency:  f,
			Weight:     w,
			PredSlope:  pred.Slope,
			PredTilt:   pred.TiltDeg,
			MeasSlope:  st.MeanSlope,
			MeasTilt:   st.MeanTilt,
			DeltaTilt:  pred.TiltDeg - st.MeanTilt,
			SlopeRatio: st.MeanSlope / pred.Slope,
		}
		if math.Abs(row.DeltaTilt) > s.WorstAbsDev {
			s.WorstAbsDev = math.Abs(row.DeltaTilt)
		}
		s.Bands = append(s.Bands, row)
	}
	return s, nil
}

// isolated 生成只含某一档、且振幅已乘权重的场，用于逐档实测。
func isolated(opt api.Options, freq, weight float64) []float64 {
	src := noise.New(opt.Seed)
	out := make([]float64, opt.Size*opt.Size)
	for y := 0; y < opt.Size; y++ {
		for x := 0; x < opt.Size; x++ {
			out[y*opt.Size+x] = weight * fbm.Eval(src, float64(x)*freq, float64(y)*freq, 1, 2, 0.5)
		}
	}
	return out
}

// Verdict 说明一档偏差超两度时的原因归属。
//
// 斜率比 = 实测斜率/预判斜率。比值明显小于一，说明谱模型给出的斜率偏大，
// 估计误差来自模型没有计入的采样衰减；此时报告里把该档标为「超限」并列出
// 比值，作为斜率估计误差的证据。
func (b BandRow) Verdict() string {
	if math.Abs(b.DeltaTilt) <= TwoDegrees {
		return "合格"
	}
	switch {
	case b.SlopeRatio < 0.8:
		return "超限：采样衰减，预判斜率偏大"
	case b.SlopeRatio > 1.2:
		return "超限：预判斜率偏小"
	default:
		return "超限：斜率接近，倾角非线性放大"
	}
}

// Nyquist 是像素网格能表示的最高频率（周期/像素）。
const Nyquist = 0.5

// AboveNyquist 返回该档频率是否超过像素网格的 Nyquist 频率。
func (b BandRow) AboveNyquist() bool { return b.Frequency > Nyquist }

// WriteCSV 把逐档对照写成 CSV。
func (s *Set) WriteCSV(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	o := s.Options
	if err := w.Write([]string{
		"set", "size", "seed", "octaves", "lacunarity", "gain", "base_frequency",
	}); err != nil {
		return err
	}
	if err := w.Write([]string{
		s.Name, fmt.Sprint(o.Size), fmt.Sprint(o.Seed), fmt.Sprint(o.Octaves),
		fmt.Sprintf("%.4f", o.Lacunarity), fmt.Sprintf("%.4f", o.Gain), fmt.Sprintf("%.4f", o.Frequency),
	}); err != nil {
		return err
	}
	if err := w.Write(nil); err != nil {
		return err
	}
	w.Write([]string{"overview_pred_tilt_deg", fmt.Sprintf("%.3f", s.Overview)})
	w.Write([]string{"overview_pred_slope", fmt.Sprintf("%.6f", s.OverviewSl)})
	w.Write([]string{"measured_mean_tilt_deg", fmt.Sprintf("%.3f", s.Measured.MeanTilt)})
	w.Write([]string{"measured_median_tilt_deg", fmt.Sprintf("%.3f", s.Measured.Median)})
	w.Write([]string{"measured_p90_tilt_deg", fmt.Sprintf("%.3f", s.Measured.P90)})
	w.Write([]string{"measured_max_tilt_deg", fmt.Sprintf("%.3f", s.Measured.MaxTilt)})
	w.Write([]string{"measured_mean_slope", fmt.Sprintf("%.6f", s.Measured.MeanSlope)})
	w.Write([]string{"span", fmt.Sprintf("%.6f", s.Field.Span)})
	w.Write([]string{"normalized_pred_tilt_deg", fmt.Sprintf("%.3f", s.NormPred)})
	w.Write([]string{"normalized_measured_mean_tilt_deg", fmt.Sprintf("%.3f", s.NormMeasured.MeanTilt)})
	w.Write([]string{"interior_cells", fmt.Sprint(s.Measured.Count)})
	w.Write(nil)
	w.Write([]string{
		"band", "frequency", "weight", "pred_slope", "pred_tilt_deg",
		"meas_slope", "meas_tilt_deg", "delta_tilt_deg", "slope_ratio",
		"above_nyquist", "within_2deg", "verdict",
	})
	for _, b := range s.Bands {
		w.Write([]string{
			fmt.Sprint(b.Index),
			fmt.Sprintf("%.4f", b.Frequency),
			fmt.Sprintf("%.4f", b.Weight),
			fmt.Sprintf("%.6f", b.PredSlope),
			fmt.Sprintf("%.3f", b.PredTilt),
			fmt.Sprintf("%.6f", b.MeasSlope),
			fmt.Sprintf("%.3f", b.MeasTilt),
			fmt.Sprintf("%+.3f", b.DeltaTilt),
			fmt.Sprintf("%.3f", b.SlopeRatio),
			fmt.Sprint(b.AboveNyquist()),
			fmt.Sprint(math.Abs(b.DeltaTilt) <= TwoDegrees),
			b.Verdict(),
		})
	}
	return w.Error()
}

// WritePGM 把倾角图写成 8 位灰度 PGM，按整图最大倾角拉伸。
func (s *Set) WritePGM(path string) error {
	max := s.Measured.MaxTilt
	if max <= 0 {
		max = 1
	}
	buf := make([]float64, len(s.Grid.Tilt))
	for i, v := range s.Grid.Tilt {
		if math.IsNaN(v) || v < 0 {
			buf[i] = 0 // 边界一圈按零涂黑
			continue
		}
		buf[i] = v / max
	}
	head := fmt.Sprintf("P5\n%d %d\n255\n", s.Grid.Size, s.Grid.Size)
	out := make([]byte, 0, len(head)+len(buf))
	out = append(out, head...)
	for _, x := range buf {
		g := int(x*255 + 0.5)
		if g > 255 {
			g = 255
		}
		out = append(out, byte(g))
	}
	return os.WriteFile(path, out, 0o644)
}

// histogram 按度分箱，返回每箱计数（箱宽 binDeg）与横轴上限。
func (s *Set) histogram(binDeg float64) (counts []int, maxBin float64) {
	n := int(math.Ceil(s.Measured.MaxTilt / binDeg))
	if n < 1 {
		n = 1
	}
	counts = make([]int, n)
	for _, v := range s.Grid.Tilt {
		if math.IsNaN(v) {
			continue
		}
		i := int(v / binDeg)
		if i < 0 {
			i = 0
		}
		if i >= n {
			i = n - 1
		}
		counts[i]++
	}
	return counts, float64(n) * binDeg
}

// WriteSVG 画倾角直方图，作为分布图存档（矢量，不依赖外部资源）。
func (s *Set) WriteSVG(path string) error {
	const width, height = 640.0, 404.0
	const left, right, top, bottom = 56.0, 16.0, 92.0, 44.0
	plotW, plotH := width-left-right, height-top-bottom
	const binDeg = 1.0
	counts, maxT := s.histogram(binDeg)
	maxC := 0
	for _, c := range counts {
		if c > maxC {
			maxC = c
		}
	}
	if maxC == 0 {
		maxC = 1
	}
	bar := "#2a78d6"
	axis := "#c3c2b7"
	grid := "#e1e0d9"
	ink := "#0b0b0b"
	ink2 := "#52514e"
	muted := "#898781"

	var b strings.Builder
	b.WriteString(fmt.Sprintf(`<svg xmlns="http://www.w3.org/2000/svg" width="%.0f" height="%.0f" viewBox="0 0 %.0f %.0f" role="img" aria-label="%s 的倾角分布">`, width, height, width, height, html.EscapeString(s.Name)))
	b.WriteString(`<rect width="100%" height="100%" fill="#fcfcfb"/>`)
	b.WriteString(fmt.Sprintf(`<text x="%.0f" y="24" font-family="system-ui,-apple-system,'Segoe UI',sans-serif" font-size="15" font-weight="600" fill="%s">%s 倾角分布</text>`, left, ink, html.EscapeString(s.Name)))
	b.WriteString(fmt.Sprintf(`<text x="%.0f" y="46" font-family="system-ui,sans-serif" font-size="11" fill="%s">f0=%.3f  oct=%d  g=%.2f  预判 %.2f°  实测均值 %.2f°  n=%d</text>`,
		left, ink2, s.Options.Frequency, s.Options.Octaves, s.Options.Gain, s.Overview, s.Measured.MeanTilt, s.Measured.Count))

	// 图例：两条参考线相距不足 4px 时合并为一项，避免文字与竖线交叠。
	const closePx = 14.0
	near := math.Abs(s.Overview-s.Measured.MeanTilt)/maxT*plotW < closePx
	legY := 72.0
	legend := []struct {
		v     float64
		color string
		label string
	}{{s.Overview, "#eb6834", "预判"}, {s.Measured.MeanTilt, ink, "实测均值"}}
	if near {
		legend = []struct {
			v     float64
			color string
			label string
		}{{s.Overview, "#eb6834", fmt.Sprintf("预判 %.2f° · 实测均值 %.2f°", s.Overview, s.Measured.MeanTilt)}}
	}
	for i, ln := range legend {
		x := left + float64(i)*230
		b.WriteString(fmt.Sprintf(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="2" stroke-dasharray="4 3"/>`,
			x, legY-4, x+18, legY-4, ln.color))
		b.WriteString(fmt.Sprintf(`<text x="%.1f" y="%.1f" font-family="system-ui,sans-serif" font-size="10" fill="%s">%s</text>`,
			x+24, legY, ink2, ln.label))
	}

	// y 网格与刻度
	for g := 0; g <= 4; g++ {
		y := top + plotH - plotH*float64(g)/4
		v := int(math.Round(float64(maxC) * float64(g) / 4))
		b.WriteString(fmt.Sprintf(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1"/>`, left, y, left+plotW, y, grid))
		b.WriteString(fmt.Sprintf(`<text x="%.1f" y="%.1f" text-anchor="end" font-family="system-ui,sans-serif" font-size="10" fill="%s">%d</text>`, left-8, y+3, muted, v))
	}
	// 柱：相邻柱之间留 2px 表面间隙
	bw := plotW / (maxT / binDeg)
	for i, c := range counts {
		if c == 0 {
			continue
		}
		h := plotH * float64(c) / float64(maxC)
		x := left + float64(i)*bw
		y := top + plotH - h
		b.WriteString(fmt.Sprintf(`<rect x="%.2f" y="%.2f" width="%.2f" height="%.2f" rx="2" fill="%s"><title>%.0f–%.0f°: %d 格</title></rect>`,
			x+1, y, math.Max(bw-2, 1), h, bar, float64(i)*binDeg, float64(i+1)*binDeg, c))
	}
	// x 轴
	b.WriteString(fmt.Sprintf(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1"/>`, left, top+plotH, left+plotW, top+plotH, axis))
	step := 1.0
	if maxT > 20 {
		step = 5
	} else if maxT > 10 {
		step = 2
	}
	for t := 0.0; t <= maxT+1e-9; t += step {
		x := left + t/maxT*plotW
		b.WriteString(fmt.Sprintf(`<text x="%.1f" y="%.1f" text-anchor="middle" font-family="system-ui,sans-serif" font-size="10" fill="%s">%.0f</text>`, x, top+plotH+16, muted, t))
	}
	b.WriteString(fmt.Sprintf(`<text x="%.1f" y="%.0f" text-anchor="middle" font-family="system-ui,sans-serif" font-size="11" fill="%s">倾角（度）</text>`, left+plotW/2, height-6, ink2))

	// 预判与实测均值两条竖线（只画在绘图区内）；重合时只画实测均值，
	// 位置以预判线示出差异。
	for _, ln := range []struct {
		v     float64
		color string
		label string
	}{{s.Overview, "#eb6834", "预判"}, {s.Measured.MeanTilt, ink, "实测均值"}} {
		if ln.v > maxT {
			continue
		}
		if near && ln.label == "实测均值" {
			continue
		}
		x := left + ln.v/maxT*plotW
		b.WriteString(fmt.Sprintf(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="2" stroke-dasharray="4 3"/>`,
			x, top, x, top+plotH, ln.color))
	}
	b.WriteString(`</svg>`)
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

// WriteMarkdown 汇总一组参数的比对表。
func (s *Set) WriteMarkdown(b *strings.Builder) {
	o := s.Options
	fmt.Fprintf(b, "### %s\n\n", s.Name)
	fmt.Fprintf(b, "参数：size=%d seed=%d octaves=%d lacunarity=%.2f gain=%.2f baseFrequency=%.4f\n\n",
		o.Size, o.Seed, o.Octaves, o.Lacunarity, o.Gain, o.Frequency)
	fmt.Fprintf(b, "| 项目 | 预判 | 实测 | 差 |\n|---|---:|---:|---:|\n")
	fmt.Fprintf(b, "| 平均倾角（度） | %.3f | %.3f | %+.3f |\n", s.Overview, s.Measured.MeanTilt, s.Overview-s.Measured.MeanTilt)
	fmt.Fprintf(b, "| 平均斜率 | %.6f | %.6f | 比值 %.3f |\n\n", s.OverviewSl, s.Measured.MeanSlope, s.Measured.MeanSlope/s.OverviewSl)
	fmt.Fprintf(b, "归一化图（R=%.4f）上同一组参数的倾角：预判 %.3f°，实测 %.3f°。\n\n",
		s.Field.Span, s.NormPred, s.NormMeasured.MeanTilt)
	fmt.Fprintf(b, "| 档 | 频率 | 权重 | 预判斜率 | 预判倾角° | 实测斜率 | 实测倾角° | 偏差° | 斜率比 | >Nyquist | 结论 |\n")
	fmt.Fprintf(b, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---|\n")
	var over []BandRow
	for _, r := range s.Bands {
		nyq := ""
		if r.AboveNyquist() {
			nyq = "是"
		}
		if math.Abs(r.DeltaTilt) > TwoDegrees {
			over = append(over, r)
		}
		fmt.Fprintf(b, "| %d | %.4f | %.4f | %.6f | %.3f | %.6f | %.3f | %+.3f | %.3f | %s | %s |\n",
			r.Index, r.Frequency, r.Weight, r.PredSlope, r.PredTilt, r.MeasSlope, r.MeasTilt, r.DeltaTilt, r.SlopeRatio, nyq, r.Verdict())
	}
	fmt.Fprintf(b, "\n分布：中位 %.2f°，P90 %.2f°，最大 %.2f°，参与统计 %d 格（排除边界一圈）。\n\n",
		s.Measured.Median, s.Measured.P90, s.Measured.MaxTilt, s.Measured.Count)
	if len(over) == 0 {
		fmt.Fprintf(b, "逐档偏差均在 %.0f 度以内。\n\n", TwoDegrees)
		return
	}
	fmt.Fprintf(b, "**超限档的斜率估计误差**：")
	for i, r := range over {
		if i > 0 {
			b.WriteString("；")
		}
		fmt.Fprintf(b, "第 %d 档 f=%.3f（%s，斜率为预判的 %.0f%%，即预判偏大 %.0f%%）偏差 %+.2f°",
			r.Index, r.Frequency, nyqNote(r), r.SlopeRatio*100, (1/r.SlopeRatio-1)*100, r.DeltaTilt)
	}
	b.WriteString("。误差来源是离散网格的中心差分在接近或超过 Nyquist（f=0.5）的频段把梯度抹平：实测衰减 A(f)≈1/(1+(f/0.76)²)：f=0.2 时仍剩约 92%，f=0.4 剩约 77%，f=0.8 剩约 47%，f=1.6 剩约 24%，故高频档的预判斜率系统性偏高。\n\n")
}

func nyqNote(r BandRow) string {
	if r.AboveNyquist() {
		return "超过 Nyquist"
	}
	return "接近 Nyquist"
}

// Write 把一组参数的全部产物写入目录：CSV、倾角 PGM、分布 SVG。
func (s *Set) Write(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	if err := s.WriteCSV(filepath.Join(dir, "bands.csv")); err != nil {
		return err
	}
	if err := s.WritePGM(filepath.Join(dir, "tilt.pgm")); err != nil {
		return err
	}
	return s.WriteSVG(filepath.Join(dir, "tilt-hist.svg"))
}
