package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"goproc/api"
	"goproc/internal/normalmap"
)

// runNormalReport 生成“位深—角度误差—台阶数”整组证据并分别存档法线贴图。
func runNormalReport(args []string) int {
	fs := flag.NewFlagSet("normalreport", flag.ContinueOnError)
	out := fs.String("out", "artifacts/normal_quant", "归档目录")
	bitsCSV := fs.String("bits", "6,8,10", "位深列表，逗号分隔")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	bitsList := []int{6, 8, 10}
	if *bitsCSV != "6,8,10" {
		bitsList = bitsList[:0]
		for _, b := range splitCSV(*bitsCSV) {
			bitsList = append(bitsList, b)
		}
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir failed:", err)
		return 1
	}

	// ---------- 标准斜坡靶 ----------
	cfg := normalmap.DefaultSlopeConfig()
	target := normalmap.BuildSlopeTarget(cfg)
	nm := target.NormalMap()

	if err := os.WriteFile(filepath.Join(*out, "slope_height_16bit.pgm"),
		normalmap.EncodeHeightPGM16(target.Width, target.Height, target.HeightData), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write height failed:", err)
		return 1
	}
	// 同步存一份归一化 8 位 PNG 预览（*.pgm 在 .gitignore 内，PNG 便于归档/查看）。
	hprev := append([]float64(nil), target.HeightData...)
	lo, hi := hprev[0], hprev[0]
	for _, v := range hprev {
		if v < lo {
			lo = v
		}
		if v > hi {
			hi = v
		}
	}
	if hi > lo {
		for i := range hprev {
			hprev[i] = (hprev[i] - lo) / (hi - lo)
		}
	}
	if hp, err := normalmap.EncodeGrayPNG(target.Width, target.Height, hprev); err == nil {
		_ = os.WriteFile(filepath.Join(*out, "slope_height.png"), hp, 0o644)
	}

	type fileRefs struct {
		NormalPPM string `json:"normal_map_ppm"`
		NormalPNG string `json:"normal_map_png"`
		ShadedPNG string `json:"shaded_png"`
	}
	type group struct {
		normalmap.StepReport
		Files fileRefs `json:"files"`
	}

	var groups []group
	var profiles [][]float64
	// 参考：未量化 float 法线的着色剖面（也存一张图）。
	refShaded := normalmap.Shade(nm.Normals, target.Light, 0)
	refProfile := target.SlopeProfile(refShaded)
	profiles = append(profiles, refProfile)
	if b, err := normalmap.EncodeGrayPNG(target.Width, target.Height, refShaded); err == nil {
		_ = os.WriteFile(filepath.Join(*out, "shaded_float_reference.png"), b, 0o644)
	}

	for _, bits := range bitsList {
		q := normalmap.Quantize(nm, bits)
		rep := normalmap.EvaluateBits(target, bits)

		ppmName := fmt.Sprintf("normal_b%d.ppm", bits)
		pngName := fmt.Sprintf("normal_b%d.png", bits)
		shadeName := fmt.Sprintf("shaded_b%d.png", bits)

		if err := os.WriteFile(filepath.Join(*out, ppmName),
			normalmap.EncodePPM(target.Width, target.Height, bits, q.Codes), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write ppm failed:", err)
			return 1
		}
		pngBytes, err := normalmap.EncodeNormalPNG(target.Width, target.Height, bits, q.Codes)
		if err != nil {
			fmt.Fprintln(os.Stderr, "encode normal png failed:", err)
			return 1
		}
		if err := os.WriteFile(filepath.Join(*out, pngName), pngBytes, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write normal png failed:", err)
			return 1
		}
		shaded := normalmap.Shade(q.Normals, target.Light, 0)
		shadePNG, err := normalmap.EncodeGrayPNG(target.Width, target.Height, shaded)
		if err != nil {
			fmt.Fprintln(os.Stderr, "encode shade failed:", err)
			return 1
		}
		if err := os.WriteFile(filepath.Join(*out, shadeName), shadePNG, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write shade failed:", err)
			return 1
		}
		profiles = append(profiles, target.SlopeProfile(shaded))
		groups = append(groups, group{StepReport: rep, Files: fileRefs{
			NormalPPM: ppmName, NormalPNG: pngName, ShadedPNG: shadeName,
		}})
	}

	// 剖面图宽度取斜坡统计区间长度，堆叠对比图。
	profW := target.X1 - target.X0
	if cmp, err := normalmap.StackProfiles(profW, profiles, 6); err == nil {
		_ = os.WriteFile(filepath.Join(*out, "shaded_compare.png"), cmp, 0o644)
	}

	// 台阶局部放大：取剖面第 120..320 个样本，最近邻横向 ×5，纵向 14 行。
	const z0, z1, zx = 120, 320, 5
	zoom := make([][]float64, len(profiles))
	for i, p := range profiles {
		seg := p[z0:z1]
		wide := make([]float64, 0, len(seg)*zx)
		for _, v := range seg {
			for k := 0; k < zx; k++ {
				wide = append(wide, v)
			}
		}
		zoom[i] = wide
	}
	if zp, err := normalmap.StackProfiles((z1-z0)*zx, zoom, 14); err == nil {
		_ = os.WriteFile(filepath.Join(*out, "banding_zoom.png"), zp, 0o644)
	}

	// 梯度分箱误差：RGB 三通道与 XY+重构造 z 两种编码。
	bins := normalmap.DefaultGradientBins()
	type binsByBits struct {
		Bits int                     `json:"bits"`
		Bins []normalmap.GradientBin `json:"bins"`
	}
	var rgbBins, xyBins []binsByBits
	for _, bits := range bitsList {
		rgbBins = append(rgbBins, binsByBits{Bits: bits, Bins: normalmap.BinErrorsByGradient(nm, normalmap.Quantize(nm, bits), target.Mask, bins)})
		xyBins = append(xyBins, binsByBits{Bits: bits, Bins: normalmap.BinErrorsByGradient(nm, normalmap.QuantizeXY(nm, bits), target.Mask, bins)})
	}

	// ---------- 反解最小位深样例 ----------
	gmax := 0.0
	for i, ok := range target.Mask {
		if ok && nm.Gradient[i] > gmax {
			gmax = nm.Gradient[i]
		}
	}
	type inverseRow struct {
		TargetAngleDeg      float64 `json:"target_angle_deg"`
		Gradient            float64 `json:"gradient_ref"`
		MinBits             int     `json:"min_bits_gradient_bound"`
		BoundAtMinDeg       float64 `json:"bound_at_min_deg"`
		BoundAtMinMinus1Deg float64 `json:"bound_at_minus1_deg"`
		MinBitsRigorous     int     `json:"min_bits_rigorous_ball"`
	}
	var inverse []inverseRow
	for _, eps := range []float64{2.0, 1.0, 0.5, 0.2, 0.1} {
		b := api.MinNormalBits(gmax, eps)
		inverse = append(inverse, inverseRow{
			TargetAngleDeg:      eps,
			Gradient:            gmax,
			MinBits:             b,
			BoundAtMinDeg:       api.NormalAngleBoundDeg(b, gmax),
			BoundAtMinMinus1Deg: api.NormalAngleBoundDeg(b-1, gmax),
			MinBitsRigorous:     api.MinNormalBitsRigorous(eps),
		})
	}

	allWithin := true
	for _, g := range groups {
		if !g.WithinBound {
			allWithin = false
		}
	}
	report := map[string]any{
		"task":             "heightmap -> normal map: VRAM bit depth vs normal-angle quantization error and slope banding steps",
		"bit_depths":       bitsList,
		"angle_unit":       "degree",
		"all_within_bound": allWithin,
		"artifact_dir":     "artifacts/normal_quant",
		"convention": map[string]any{
			"unorm_max_code":      "M=2^b-1",
			"component_half_step": "h=1/M",
			"angle_step_first":    "dtheta ~= 2*sqrt(1+g^2)/M rad",
			"gradient_bound":      "eps <= 2*sqrt(1+g^2)/M rad (strict for b>=3 via asin(sqrt(3)/M) <= 2/M)",
			"rigorous_ball_bound": "eps <= asin(sqrt(3)/M) rad",
			"inverse_min_bits":    "b_min = ceil(log2(1 + 2*sqrt(1+g^2)/eps_rad))",
			"inverse_rigorous":    "b_min = ceil(log2(1 + sqrt(3)/sin(eps_rad)))",
			"step_count":          "maximal constant-shade runs on the slope interior row; plateaus excluded by mask",
		},
		"target": map[string]any{
			"width": target.Width, "height": target.Height,
			"theta0_deg": cfg.Theta0Deg, "theta1_deg": cfg.Theta1Deg,
			"light_tilt_deg": cfg.LightDeg, "gradient_max": gmax,
			"slope_columns": []int{target.X0, target.X1},
		},
		"groups":                         groups,
		"inverse":                        inverse,
		"gradient_bins_rgb":              rgbBins,
		"gradient_bins_xy_reconstruct_z": xyBins,
	}
	writeJSON := func(name string, v any) error {
		b, err := json.MarshalIndent(v, "", "  ")
		if err != nil {
			return err
		}
		b = append(b, '\n')
		return os.WriteFile(filepath.Join(*out, name), b, 0o644)
	}
	if err := writeJSON("report.json", report); err != nil {
		fmt.Fprintln(os.Stderr, "write report failed:", err)
		return 1
	}
	// 单独的“一组”文件：只含三个位深的误差与台阶数。
	if err := writeJSON("normal_groups.json", groups); err != nil {
		fmt.Fprintln(os.Stderr, "write groups failed:", err)
		return 1
	}
	// artifacts 根目录再放一份分组结果，方便固定路径检查。
	if err := writeJSON("../normal_groups.json", groups); err != nil {
		fmt.Fprintln(os.Stderr, "write root groups failed:", err)
	}

	// 在 artifacts 根再放一份，方便固定路径检查。
	if err := writeJSON("../normal_report.json", report); err != nil {
		fmt.Fprintln(os.Stderr, "write root report failed:", err)
	}

	// ---------- 真实程序化地形也跑一遍并分别存档 ----------
	generateTerrainArchives(*out)

	// ---------- 控制台摘要 ----------
	fmt.Printf("normal report -> %s\n", *out)
	fmt.Printf("target %dx%d theta %.0f..%.0f deg gmax=%.4f\n", target.Width, target.Height,
		cfg.Theta0Deg, cfg.Theta1Deg, gmax)
	fmt.Printf("%-5s %-8s %-12s %-12s %-12s %-8s %-8s\n", "bits", "M", "maxErr°", "meanErr°", "bound°", "steps", "st8bit")
	for _, g := range groups {
		fmt.Printf("%-5d %-8d %-12.5f %-12.5f %-12.5f %-8d %-8d viol=%d\n",
			g.Bits, g.MaxCode, g.MaxAngleErrorDeg, g.MeanAngleErrorDeg, g.BoundDeg,
			g.Steps, g.Steps8Bit, g.BoundViolations)
	}
	for _, row := range inverse {
		fmt.Printf("inverse eps=%.2f° g=%.3f -> b_min=%d (bound %.4f°, b-1 bound %.4f°), rigorous b=%d\n",
			row.TargetAngleDeg, row.Gradient, row.MinBits, row.BoundAtMinDeg, row.BoundAtMinMinus1Deg, row.MinBitsRigorous)
	}
	return 0
}

func splitCSV(s string) []int {
	var out []int
	n := 0
	neg := false
	has := false
	flush := func() {
		if has {
			if neg {
				n = -n
			}
			out = append(out, n)
		}
		n, neg, has = 0, false, false
	}
	for _, r := range s {
		switch {
		case r == ' ':
			continue
		case r == ',':
			flush()
		case r == '-':
			neg = true
		case r >= '0' && r <= '9':
			n = n*10 + int(r-'0')
			has = true
		}
	}
	flush()
	return out
}

// generateTerrainArchives 用仓库自己的确定性地形生成器出图，转换并归档
// 6/8/10 位法线贴图与着色图，回应“高度图转法线贴图后斜坡台阶”的原始场景。
func generateTerrainArchives(outDir string) {
	h, err := api.Generate(api.Options{Size: 256, Octaves: 4, Lacunarity: 2, Gain: 0.5, Seed: 7, Frequency: 0.02})
	if err != nil {
		fmt.Fprintln(os.Stderr, "terrain generate failed:", err)
		return
	}
	const heightScale = 24.0
	nm := api.HeightmapNormals(h, heightScale)
	// 用 internal 包重算一份用于编码/着色。
	full := normalmap.FromHeight(h.Size, h.Size, h.Data, heightScale)
	light := normalmap.Vec3{X: -0.45, Y: -0.25, Z: 0.86}.Normalize()

	// 地形坡度统计：只在斜坡像素（g>=0.15）上评误差。
	mask := make([]bool, h.Size*h.Size)
	slopePixels := 0
	for i, g := range full.Gradient {
		if g >= 0.15 {
			mask[i] = true
			slopePixels++
		}
	}
	type terrainGroup struct {
		Bits              int     `json:"bits"`
		SlopePixels       int     `json:"slope_pixels"`
		MaxAngleErrorDeg  float64 `json:"max_angle_error_deg"`
		MeanAngleErrorDeg float64 `json:"mean_angle_error_deg"`
		BoundMaxDeg       float64 `json:"bound_max_deg"`
		Violations        int     `json:"bound_violations"`
		NormalMapPNG      string  `json:"normal_map_png"`
		NormalMapPPM      string  `json:"normal_map_ppm"`
		ShadedPNG         string  `json:"shaded_png"`
	}
	var tgroups []terrainGroup
	for _, bits := range []int{6, 8, 10} {
		q := normalmap.Quantize(full, bits)
		st := normalmap.MeasureErrors(full, q, mask)
		pngName := fmt.Sprintf("terrain_normal_b%d.png", bits)
		ppmName := fmt.Sprintf("terrain_normal_b%d.ppm", bits)
		shadeName := fmt.Sprintf("terrain_shaded_b%d.png", bits)
		if b, err := normalmap.EncodeNormalPNG(h.Size, h.Size, bits, q.Codes); err == nil {
			_ = os.WriteFile(filepath.Join(outDir, pngName), b, 0o644)
		}
		_ = os.WriteFile(filepath.Join(outDir, ppmName),
			normalmap.EncodePPM(h.Size, h.Size, bits, q.Codes), 0o644)
		shaded := normalmap.Shade(q.Normals, light, 0.05)
		if b, err := normalmap.EncodeGrayPNG(h.Size, h.Size, shaded); err == nil {
			_ = os.WriteFile(filepath.Join(outDir, shadeName), b, 0o644)
		}
		tgroups = append(tgroups, terrainGroup{
			Bits: bits, SlopePixels: slopePixels,
			MaxAngleErrorDeg: st.MaxDeg, MeanAngleErrorDeg: st.MeanDeg,
			BoundMaxDeg: st.BoundMaxDeg, Violations: st.Violations,
			NormalMapPNG: pngName, NormalMapPPM: ppmName, ShadedPNG: shadeName,
		})
	}
	// 原始高度图也存档（8 位 PGM 仅预览，PNG16 更精确）。
	_ = os.WriteFile(filepath.Join(outDir, "terrain_height.pgm"), h.ToPGM(), 0o644)
	if b, err := normalmap.EncodeGrayPNG(h.Size, h.Size, h.Data); err == nil {
		_ = os.WriteFile(filepath.Join(outDir, "terrain_height.png"), b, 0o644)
	}
	b, _ := json.MarshalIndent(map[string]any{
		"height_scale": heightScale, "slope_gradient_threshold": 0.15,
		"groups": tgroups, "normal_info_size": nm.Size,
	}, "", "  ")
	_ = os.WriteFile(filepath.Join(outDir, "terrain_report.json"), append(b, '\n'), 0o644)
}
