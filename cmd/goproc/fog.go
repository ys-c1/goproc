package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strconv"

	"goproc/api"
	"goproc/internal/fog"
)

type levelReport struct {
	Label            string  `json:"label"`
	BetaPerKM        float64 `json:"beta_per_km"`
	ReliefKM         float64 `json:"relief_km"`
	Chi              float64 `json:"chi"`
	KoschmiederVisKM float64 `json:"koschmieder_vis_km"`
	PredCDF          fog.Count
	PredViaCDF       fog.Count `json:"pred_via_cdf_lookup"`
	PredHistogram32  fog.Count
	PredUniform      float64 `json:"pred_uniform_fraction"`
	PredGaussian     float64 `json:"pred_gaussian_fraction"`
	RenderFloat      fog.Count
	Readback8Bit     fog.Count
	CDFMinusRenderPP float64 `json:"cdf_minus_render_pp"`
	CDFRelDiff       float64 `json:"cdf_relative_diff"`
	ReadbackRelDiff  float64 `json:"readback8_relative_diff_vs_float"`
	UniformRelDiff   float64 `json:"uniform_relative_diff_vs_render"`
	GaussianRelDiff  float64 `json:"gaussian_relative_diff_vs_render"`
	ImageFogPNG      string  `json:"image_fog_png"`
	ImageContrastPNG string  `json:"image_contrast_png"`
	ImageMaskPNG     string  `json:"image_mask_png"`
}

type scalingRow struct {
	Chi      float64 `json:"chi"`
	ReliefKM float64 `json:"relief_km"`
	Beta     float64 `json:"beta_per_km"`
	Fraction float64 `json:"fraction"`
}

type shapeRow struct {
	Seed     uint64  `json:"seed"`
	Chi      float64 `json:"chi"`
	Mean     float64 `json:"mean"`
	Std      float64 `json:"std"`
	Fraction float64 `json:"fraction"`
}

type experimentReport struct {
	Options       api.Options        `json:"terrain_options"`
	Epsilon       float64            `json:"epsilon"`
	TauStar       float64            `json:"tau_star"`
	Relief        fog.HeightStats    `json:"height_stats"`
	Levels        []levelReport      `json:"levels"`
	Scaling       []scalingRow       `json:"scaling_product_check"`
	ShapeCheck    []shapeRow         `json:"cross_seed_shape_check"`
	ScalingSpread map[string]float64 `json:"scaling_max_spread_by_chi"`
}

func runFog(args []string) error {
	fs := flag.NewFlagSet("fog", flag.ContinueOnError)
	size := fs.Int("size", 512, "地形边长")
	seed := fs.Uint64("seed", 7, "种子")
	octaves := fs.Int("octaves", 5, "分形层数")
	frequency := fs.Float64("frequency", 0.02, "基础频率")
	outdir := fs.String("outdir", "artifacts/fog", "产物目录")
	relief := fs.Float64("relief", 1.0, "地形起伏 R（km）")
	beta1 := fs.Float64("beta1", 5.0, "第一档雾浓度（km^-1）")
	beta2 := fs.Float64("beta2", 10.0, "第二档雾浓度（km^-1）")
	eps := fs.Float64("eps", fog.DefaultEpsilon, "对比度阈值")
	if err := fs.Parse(args); err != nil {
		return err
	}

	opt := api.Options{
		Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5,
		Seed: *seed, Frequency: *frequency,
	}
	hmap, err := api.Generate(opt)
	if err != nil {
		return fmt.Errorf("generate: %w", err)
	}
	if err := os.MkdirAll(*outdir, 0o755); err != nil {
		return err
	}

	stats := fog.Stats(hmap.Data)
	hist32 := fog.BuildHistogram(hmap.Data, 32)
	tau := fog.TauThreshold(*eps)

	saveGray := func(name string, img *image.Gray) (string, error) {
		p := filepath.Join(*outdir, name)
		f, err := os.Create(p)
		if err != nil {
			return "", err
		}
		defer f.Close()
		if err := png.Encode(f, img); err != nil {
			return "", err
		}
		return p, nil
	}
	readGray := func(p string) (*image.Gray, error) {
		f, err := os.Open(p)
		if err != nil {
			return nil, err
		}
		defer f.Close()
		im, err := png.Decode(f)
		if err != nil {
			return nil, err
		}
		g, ok := im.(*image.Gray)
		if !ok {
			return nil, fmt.Errorf("%s 不是 8-bit 灰度 PNG", p)
		}
		return g, nil
	}

	// 无雾底图只存一次。
	baseRes := fog.Render(hmap.Data, hmap.Size, fog.DefaultRenderOptions(fog.Chi(*beta1, *relief)))
	basePath, err := saveGray("terrain_base.png", baseRes.BaseGray)
	if err != nil {
		return err
	}
	_ = basePath

	betas := []struct {
		label string
		beta  float64
	}{{"weak", *beta1}, {"strong", *beta2}}

	var levels []levelReport
	var renders []*fog.Result
	for _, lv := range betas {
		chi := fog.Chi(lv.beta, *relief)
		pred := fog.Predict(hmap.Data, chi, *eps)
		predCDF := fog.PredictViaCDF(hmap.Data, chi, *eps)
		predHist := fog.PredictHistogramStep(hist32, chi, *eps)
		uni := fog.PredictUniform(chi, *eps)
		gauss := fog.PredictGaussian(stats.Mean, stats.Std, chi, *eps)

		ro := fog.DefaultRenderOptions(chi)
		ro.Epsilon = *eps
		res := fog.Render(hmap.Data, hmap.Size, ro)
		renders = append(renders, res)

		tag := fmt.Sprintf("%s_beta%g_chi%g", lv.label, lv.beta, chi)
		fogName := "fog_" + tag + ".png"
		conName := "contrast_" + tag + ".png"
		maskName := "mask_" + tag + ".png"
		fogPath, err := saveGray(fogName, res.Gray)
		if err != nil {
			return err
		}
		conPath, err := saveGray(conName, res.Contrast8)
		if err != nil {
			return err
		}
		if _, err := saveGray(maskName, res.Mask); err != nil {
			return err
		}

		// 从磁盘回读刚保存的 8-bit 对比度图，逐像素统计。
		g8, err := readGray(conPath)
		if err != nil {
			return err
		}
		rb := fog.CountContrastGray8(g8, chi, *eps)

		meas := res.FloatCount.Fraction
		levels = append(levels, levelReport{
			Label:            lv.label,
			BetaPerKM:        lv.beta,
			ReliefKM:         *relief,
			Chi:              chi,
			KoschmiederVisKM: tau / lv.beta,
			PredCDF:          pred,
			PredViaCDF:       predCDF,
			PredHistogram32:  predHist,
			PredUniform:      uni,
			PredGaussian:     gauss,
			RenderFloat:      res.FloatCount,
			Readback8Bit:     rb,
			CDFMinusRenderPP: (pred.Fraction - meas) * 100,
			CDFRelDiff:       fog.RelativeDiff(pred.Fraction, meas),
			ReadbackRelDiff:  fog.RelativeDiff(rb.Fraction, meas),
			UniformRelDiff:   fog.RelativeDiff(uni, meas),
			GaussianRelDiff:  fog.RelativeDiff(gauss, meas),
			ImageFogPNG:      fogPath,
			ImageContrastPNG: conPath,
			ImageMaskPNG:     filepath.Join(*outdir, maskName),
		})
	}

	// 总览面板：每行一个浓度，三列依次为 带雾图 / 对比度图 / 可见掩膜。
	n := hmap.Size
	panel := image.NewGray(image.Rect(0, 0, 3*n, 2*n))
	for row, res := range renders {
		imgs := []*image.Gray{res.Gray, res.Contrast8, res.Mask}
		for col, src := range imgs {
			for y := 0; y < n; y++ {
				for x := 0; x < n; x++ {
					panel.Pix[(row*n+y)*panel.Stride+col*n+x] = src.Pix[y*src.Stride+x]
				}
			}
		}
	}
	if _, err := saveGray("overview_panel.png", panel); err != nil {
		return err
	}

	// —— 乘积决定论验证：同一 chi 用不同 (beta,R) 组合，比例必须一致 ——
	chis := []float64{4, 5, 6, 7, 8, 9, 10, 12}
	reliefs := []float64{0.5, 1.0, 2.0}
	var scaling []scalingRow
	spread := map[string]float64{}
	for _, chi := range chis {
		var fs2 []float64
		for _, R := range reliefs {
			beta := chi / R
			f := fog.Predict(hmap.Data, fog.Chi(beta, R), *eps).Fraction
			scaling = append(scaling, scalingRow{Chi: chi, ReliefKM: R, Beta: beta, Fraction: f})
			fs2 = append(fs2, f)
		}
		lo, hi := fs2[0], fs2[0]
		for _, v := range fs2[1:] {
			if v < lo {
				lo = v
			}
			if v > hi {
				hi = v
			}
		}
		spread[strconv.FormatFloat(chi, 'f', -1, 64)] = hi - lo
	}

	// —— 跨种子：同一 chi 下归一化高度分布形状不同，比例有散布 ——
	var shape []shapeRow
	for _, sd := range []uint64{1, 7, 42, 2024} {
		o2 := opt
		o2.Seed = sd
		o2.Size = 256
		hm2, err := api.Generate(o2)
		if err != nil {
			return err
		}
		st2 := fog.Stats(hm2.Data)
		for _, chi := range []float64{5, 9} {
			shape = append(shape, shapeRow{
				Seed: sd, Chi: chi, Mean: st2.Mean, Std: st2.Std,
				Fraction: fog.Predict(hm2.Data, chi, *eps).Fraction,
			})
		}
	}

	rep := experimentReport{
		Options:       opt,
		Epsilon:       *eps,
		TauStar:       tau,
		Relief:        stats,
		Levels:        levels,
		Scaling:       scaling,
		ShapeCheck:    shape,
		ScalingSpread: spread,
	}

	jb, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(*outdir, "results.json"), jb, 0o644); err != nil {
		return err
	}

	cf, err := os.Create(filepath.Join(*outdir, "scaling.csv"))
	if err != nil {
		return err
	}
	defer cf.Close()
	cw := csv.NewWriter(cf)
	_ = cw.Write([]string{"chi", "relief_km", "beta_per_km", "predicted_visible_fraction"})
	for _, r := range scaling {
		_ = cw.Write([]string{
			strconv.FormatFloat(r.Chi, 'f', -1, 64),
			strconv.FormatFloat(r.ReliefKM, 'f', 1, 64),
			strconv.FormatFloat(r.Beta, 'f', 2, 64),
			strconv.FormatFloat(r.Fraction, 'f', 6, 64),
		})
	}
	cw.Flush()

	printMarkdown(rep)
	return nil
}

func printMarkdown(rep experimentReport) {
	fmt.Printf("# fog experiment\n\n")
	fmt.Printf("seed=%d size=%d octaves=%d eps=%.2f tau*=%.4f\n\n",
		rep.Options.Seed, rep.Options.Size, rep.Options.Octaves, rep.Epsilon, rep.TauStar)
	fmt.Printf("height: mean=%.4f std=%.4f p05=%.3f p50=%.3f p95=%.3f\n\n",
		rep.Relief.Mean, rep.Relief.Std,
		rep.Relief.Quantile["p05"], rep.Relief.Quantile["p50"], rep.Relief.Quantile["p95"])

	fmt.Println("| 项目 | 弱雾档 | 浓雾档 |")
	fmt.Println("|---|---:|---:|")
	l := rep.Levels
	fmt.Printf("| beta (km^-1) | %.3g | %.3g |\n", l[0].BetaPerKM, l[1].BetaPerKM)
	fmt.Printf("| chi=beta*R | %.3g | %.3g |\n", l[0].Chi, l[1].Chi)
	fmt.Printf("| Koschmieder 能见度 (km) | %.3f | %.3f |\n", l[0].KoschmiederVisKM, l[1].KoschmiederVisKM)
	fmt.Printf("| CDF 预判比例 | %.4f | %.4f |\n", l[0].PredCDF.Fraction, l[1].PredCDF.Fraction)
	fmt.Printf("| CDF 查表预判 | %.4f | %.4f |\n", l[0].PredViaCDF.Fraction, l[1].PredViaCDF.Fraction)
	fmt.Printf("| 32 柱直方图预判 | %.4f | %.4f |\n", l[0].PredHistogram32.Fraction, l[1].PredHistogram32.Fraction)
	fmt.Printf("| 高斯参数预判 | %.4f | %.4f |\n", l[0].PredGaussian, l[1].PredGaussian)
	fmt.Printf("| 均匀分布乘积速判 | %.4f | %.4f |\n", l[0].PredUniform, l[1].PredUniform)
	fmt.Printf("| 渲染浮点逐像素实测 | %.4f (%d/%d) | %.4f (%d/%d) |\n",
		l[0].RenderFloat.Fraction, l[0].RenderFloat.Visible, l[0].RenderFloat.Total,
		l[1].RenderFloat.Fraction, l[1].RenderFloat.Visible, l[1].RenderFloat.Total)
	fmt.Printf("| 8-bit 对比度图回读 | %.4f (%d/%d) | %.4f (%d/%d) |\n",
		l[0].Readback8Bit.Fraction, l[0].Readback8Bit.Visible, l[0].Readback8Bit.Total,
		l[1].Readback8Bit.Fraction, l[1].Readback8Bit.Visible, l[1].Readback8Bit.Total)
	fmt.Printf("| CDF-渲染 偏差 (百分点) | %+.3f pp | %+.3f pp |\n", l[0].CDFMinusRenderPP, l[1].CDFMinusRenderPP)
	fmt.Printf("| CDF 相对偏差 | %.3f%% | %.3f%% |\n", l[0].CDFRelDiff*100, l[1].CDFRelDiff*100)
	fmt.Printf("| 8-bit 回读相对偏差 | %.3f%% | %.3f%% |\n", l[0].ReadbackRelDiff*100, l[1].ReadbackRelDiff*100)
	fmt.Printf("| 均匀速判相对偏差 | %.1f%% | %.1f%% |\n", l[0].UniformRelDiff*100, l[1].UniformRelDiff*100)
	fmt.Printf("| 高斯预判相对偏差 | %.1f%% | %.1f%% |\n", l[0].GaussianRelDiff*100, l[1].GaussianRelDiff*100)

	fmt.Println("\nproduct collapse (max fraction spread across R=0.5/1/2 km at fixed chi):")
	for _, chi := range []float64{4, 5, 6, 7, 8, 9, 10, 12} {
		k := strconv.FormatFloat(chi, 'f', -1, 64)
		fmt.Printf("  chi=%-2g spread=%.2e\n", chi, rep.ScalingSpread[k])
	}
}
