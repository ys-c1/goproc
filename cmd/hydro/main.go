// hydro 命令在确定性地形上做水位—连通分量分析：
// 一次按高度扫描预测各水位分量数，独立 BFS 逐档核对，并导出 CSV、Markdown 报告与水陆图。
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"goproc/api"
	"goproc/internal/hydro"
	"goproc/internal/render"
)

func main() {
	fs := flag.NewFlagSet("hydro", flag.ContinueOnError)
	size := fs.Int("size", 128, "地形边长")
	seed := fs.Uint64("seed", 7, "噪声种子")
	octaves := fs.Int("octaves", 4, "分形层数")
	bands := fs.Int("bands", 21, "展示水位档数（0 到 1 等距）")
	cell := fs.Int("cell", 6, "水陆图每格像素边长")
	out := fs.String("out", "artifacts/hydro_run", "输出目录")
	fontPath := fs.String("font", "/home/ys/.local/share/fonts/NotoSansCJKsc-Regular.ttf", "TTF 字体路径")
	if err := fs.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	if err := run(*size, *seed, *octaves, *bands, *cell, *out, *fontPath); err != nil {
		fmt.Fprintln(os.Stderr, "hydro failed:", err)
		os.Exit(1)
	}
}

type bandRow struct {
	index                   int
	level                   float64
	predicted, measured     int
	births, merges, regular int
	net                     int
}

func run(size int, seed uint64, octaves, bands, cellPx int, outDir, fontPath string) error {
	if err := os.MkdirAll(filepath.Join(outDir, "maps"), 0o755); err != nil {
		return err
	}

	// 1) 确定性地形（与 gen 同一管线）。
	hm, err := api.Generate(api.Options{
		Size: size, Octaves: octaves, Lacunarity: 2, Gain: 0.5,
		Seed: seed, Frequency: 0.02,
	})
	if err != nil {
		return err
	}
	h := hm.Data

	// 2) 一次扫描：事件表 + 各高度档预测分量数。
	t0 := time.Now()
	an := hydro.Analyze(size, h)
	scanDur := time.Since(t0)

	// 每个事件所属的展示档（高度升序第几个不同高度）。
	eventBand := make([]int, len(an.Events))
	evIdx := 0
	for bi, rec := range an.Records {
		for range rec.Events {
			eventBand[evIdx] = bi + 1
			evIdx++
		}
	}

	// 3) 全不同高度档独立 BFS 实测（只用于核对，不参与预测），并计时对照。
	type bfsResult struct {
		level float64
		comp  int
	}
	t1 := time.Now()
	allBFS := make([]bfsResult, len(an.Records))
	mismatchAll := 0
	for i, rec := range an.Records {
		c := hydro.CountComponentsBFS(size, h, rec.Height)
		allBFS[i] = bfsResult{rec.Height, c}
		if c != rec.Components {
			mismatchAll++
		}
	}
	bfsDur := time.Since(t1)

	// 4) 展示水位档（0..1 等距）：预测、实测、本档事件统计。
	rows := make([]bandRow, 0, bands)
	prevComp := 0
	mismatchShow := 0
	prevBandIdx := -1 // 上一展示档覆盖到的最后一条高度记录
	for k := 0; k < bands; k++ {
		w := float64(k) / float64(bands-1)
		pred := an.ComponentsAt(w)
		meas := hydro.CountComponentsBFS(size, h, w)
		if pred != meas {
			mismatchShow++
		}
		bi := an.Band(w)
		// 本档跨过的事件 = 高度落在 (上一档水位, 本档水位] 的记录，
		// 即高度记录下标 (prevBandIdx, bi]。
		births, merges, regular := 0, 0, 0
		for ri := prevBandIdx + 1; ri <= bi; ri++ {
			r := an.Records[ri]
			births += r.Births
			merges += r.Merges
			regular += r.Regular
		}
		rows = append(rows, bandRow{
			index: k + 1, level: w, predicted: pred, measured: meas,
			births: births, merges: merges, regular: regular, net: pred - prevComp,
		})
		prevComp = pred
		prevBandIdx = bi
	}

	// 5) 导出 levels.csv。
	lf, err := os.Create(filepath.Join(outDir, "levels.csv"))
	if err != nil {
		return err
	}
	lw := csv.NewWriter(lf)
	if err := lw.Write([]string{
		"index", "level", "predicted", "measured", "match",
		"new_minima", "new_saddles", "net_change",
	}); err != nil {
		return err
	}
	for _, r := range rows {
		if err := lw.Write([]string{
			strconv.Itoa(r.index),
			strconv.FormatFloat(r.level, 'f', 6, 64),
			strconv.Itoa(r.predicted),
			strconv.Itoa(r.measured),
			strconv.FormatBool(r.predicted == r.measured),
			strconv.Itoa(r.births),
			strconv.Itoa(r.merges),
			fmt.Sprintf("%+d", r.net),
		}); err != nil {
			return err
		}
	}
	lw.Flush()
	if err := lw.Error(); err != nil {
		return err
	}
	lf.Close()

	// 6) 导出 jumps.csv（每个临界台地一行）。
	jf, err := os.Create(filepath.Join(outDir, "jumps.csv"))
	if err != nil {
		return err
	}
	jw := csv.NewWriter(jf)
	if err := jw.Write([]string{
		"order", "height_band", "y", "x", "height", "class", "cells",
		"k", "delta", "pattern", "geom_saddle", "local_max",
	}); err != nil {
		return err
	}
	for i, e := range an.Events {
		if err := jw.Write([]string{
			strconv.Itoa(i + 1),
			strconv.Itoa(eventBand[i]),
			strconv.Itoa(e.Y),
			strconv.Itoa(e.X),
			strconv.FormatFloat(e.Height, 'f', 17, 64),
			e.Kind.String(),
			strconv.Itoa(len(e.Cells)),
			strconv.Itoa(e.K),
			strconv.Itoa(e.Delta),
			e.Pattern,
			strconv.FormatBool(e.GeomSaddle),
			strconv.FormatBool(e.LocalMax),
		}); err != nil {
			return err
		}
	}
	jw.Flush()
	if err := jw.Error(); err != nil {
		return err
	}
	jf.Close()

	// 7) 每档一张水陆图。
	fonts, err := render.LoadFont(fontPath, 15)
	if err != nil {
		return fmt.Errorf("加载字体 %s: %w", fontPath, err)
	}
	defer fonts.Close()

	for _, r := range rows {
		// 只渲染真正引起分量变化的临界台地（极小/鞍点）；普通点 Δ=0 不标记。
		marks := make([]render.EventMark, 0)
		// 本档刚跨过的事件：高度落在 (上一档水位, 本档水位]。
		lo := r.level - 1.0/float64(bands-1)
		if r.index == 1 {
			lo = -1
		}
		for _, e := range an.Events {
			if e.Kind == hydro.KindRegular || e.Height > r.level {
				continue
			}
			marks = append(marks, render.EventMark{
				Y: e.Y, X: e.X, IsMinima: e.Kind == hydro.KindMinimum,
				IsCurrent: e.Height > lo,
			})
		}
		md := render.MapData{
			Size: size, Height: h, WaterLevel: r.level,
			BandIndex: r.index, BandCount: bands,
			Predicted: r.predicted, Measured: r.measured, NetChange: r.net,
			BirthsThisBand: r.births, MergesThisBand: r.merges,
			Events: marks,
		}
		name := fmt.Sprintf("level_%02d.png", r.index)
		f, err := os.Create(filepath.Join(outDir, "maps", name))
		if err != nil {
			return err
		}
		if err := render.Render(f, md, render.Options{
			Cell: cellPx, PanelW: 330, Margin: 24, Font: fonts,
			Palette: render.DefaultPalette(),
		}); err != nil {
			f.Close()
			return err
		}
		f.Close()
	}

	// 8) 结构统计与 Markdown 报告。
	boundaryMinima := 0
	geomSaddles := 0
	saddleLocalMax := 0
	maxK := 0
	sumDelta := 0
	sumKM1 := 0
	for _, e := range an.Events {
		if e.Kind == hydro.KindMinimum &&
			(e.Y == 0 || e.X == 0 || e.Y == size-1 || e.X == size-1) {
			boundaryMinima++
		}
		if e.GeomSaddle {
			geomSaddles++
		}
		if e.Kind == hydro.KindSaddle {
			if e.LocalMax {
				saddleLocalMax++
			}
			if e.K > maxK {
				maxK = e.K
			}
			sumKM1 += e.K - 1
		}
		sumDelta += e.Delta
	}
	changedBands := 0
	for _, rec := range an.Records {
		if rec.Delta != 0 {
			changedBands++
		}
	}

	if err := writeReport(filepath.Join(outDir, "report.md"), reportInput{
		size: size, seed: seed, octaves: octaves, bands: bands,
		an: an, rows: rows, eventBand: eventBand,
		scanDur: scanDur, bfsDur: bfsDur,
		mismatchAll: mismatchAll, mismatchShow: mismatchShow,
		boundaryMinima: boundaryMinima, geomSaddles: geomSaddles,
		saddleLocalMax: saddleLocalMax, maxK: maxK,
		sumDelta: sumDelta, sumKM1: sumKM1, changedBands: changedBands,
	}); err != nil {
		return err
	}

	fmt.Printf("done: %d 格点, %d 个不同高度, 事件 %d (极小 %d/鞍点 %d/普通 %d)\n",
		size*size, len(an.Records), len(an.Events), an.Minima, an.Saddles, an.Regular)
	fmt.Printf("全档 BFS 不一致 %d 处；展示档不一致 %d 处\n", mismatchAll, mismatchShow)
	fmt.Printf("扫描 %v；全档 BFS %v\n", scanDur, bfsDur)
	fmt.Printf("输出目录 %s\n", outDir)
	return nil
}
