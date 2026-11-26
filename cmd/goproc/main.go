package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"goproc/api"
	"goproc/internal/calibrate"
	"goproc/internal/water"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "gen":
		runGen(os.Args[2:])
	case "highlight":
		runHighlight(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage:")
	fmt.Fprintln(os.Stderr, "  goproc gen --size N --seed S --octaves K --out F")
	fmt.Fprintln(os.Stderr, "  goproc highlight --size N --seed S --octaves K --calib d1,d2,d3 --check e1,e2 --outdir D")
}

func runGen(args []string) {
	fs := flag.NewFlagSet("gen", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 1, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	out := fs.String("out", "map.pgm", "输出文件")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	h, err := api.Generate(api.Options{Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5, Seed: *seed, Frequency: 0.02})
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		os.Exit(1)
	}
	if err := os.WriteFile(*out, h.ToPGM(), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write failed:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s size=%d\n", *out, h.Size)
}

// calibReport 是留档 JSON 的结构。
type calibReport struct {
	Heightmap struct {
		Size    int    `json:"size"`
		Seed    uint64 `json:"seed"`
		Octaves int    `json:"octaves"`
	} `json:"heightmap"`
	Shader water.Shader `json:"shader"`
	Calibration []struct {
		LightElevDeg float64 `json:"lightElevDeg"`
		Threshold     float64 `json:"threshold"`
		WhiteCount    int     `json:"whiteCount"`
		MaskFile      string  `json:"maskFile"`
		RenderFile    string  `json:"renderFile"`
	} `json:"calibration"`
	Curve struct {
		A float64 `json:"a"`
		B float64 `json:"b"`
		C float64 `json:"c"`
	} `json:"quadraticCurve"`
	Validation []struct {
		LightElevDeg       float64 `json:"lightElevDeg"`
		ThresholdFromCurve  float64 `json:"thresholdFromCurve"`
		CriticalAngleDeg    float64 `json:"criticalAngleDeg"`
		PredictedCount      int     `json:"predictedCount"`
		RenderedWhiteCount  int     `json:"renderedWhiteCount"`
		OnlyPredicted       int     `json:"onlyPredicted"`
		OnlyRendered        int     `json:"onlyRendered"`
		MaxOffsetCells      float64 `json:"maxOffsetCells"`
		OverTwoCells        int     `json:"overTwoCells"`
		PredMaskFile        string  `json:"predMaskFile"`
		RenderMaskFile      string  `json:"renderMaskFile"`
		RenderFile          string  `json:"renderFile"`
		DeviationFile       string  `json:"deviationMaskFile"`
	} `json:"validation"`
}

func runHighlight(args []string) {
	fs := flag.NewFlagSet("highlight", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 1, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	calibAngles := fs.String("calib", "20,40,60", "三个标定光照仰角（度，逗号分隔）")
	checkAngles := fs.String("check", "30,50", "两个验证光照仰角（度，逗号分隔）")
	outdir := fs.String("outdir", "artifacts/highlight", "留档目录")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	calibDeg, err := parseAngles(*calibAngles, 3)
	if err != nil {
		fmt.Fprintln(os.Stderr, "calib 参数非法:", err)
		os.Exit(2)
	}
	checkDeg, err := parseAngles(*checkAngles, 2)
	if err != nil {
		fmt.Fprintln(os.Stderr, "check 参数非法:", err)
		os.Exit(2)
	}
	if err := os.MkdirAll(*outdir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "建目录失败:", err)
		os.Exit(1)
	}

	h, err := api.Generate(api.Options{Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5, Seed: *seed, Frequency: 0.02})
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		os.Exit(1)
	}
	sh := water.DefaultShader()
	nf := water.NewNormalField(h.Size, h.Data, sh.GridStep)

	// 离线标定：三个已知光照角渲染并测阈值。
	var points [3]water.CalibrationPoint
	var report calibReport
	report.Heightmap.Size = h.Size
	report.Heightmap.Seed = *seed
	report.Heightmap.Octaves = *octaves
	report.Shader = sh
	for i, deg := range calibDeg {
		p, err := sh.MeasureThreshold(nf, deg*math.Pi/180)
		if err != nil {
			fmt.Fprintf(os.Stderr, "标定角 %g 度测量失败: %v\n", deg, err)
			os.Exit(1)
		}
		points[i] = p
		frame := sh.Render(nf, deg*math.Pi/180)
		maskName := fmt.Sprintf("calib_angle%03d_white.pgm", int(math.Round(deg)))
		renderName := fmt.Sprintf("calib_angle%03d_render.pgm", int(math.Round(deg)))
		writeMaskPGM(filepath.Join(*outdir, maskName), h.Size, frame.White)
		writeGrayPGM(filepath.Join(*outdir, renderName), h.Size, frame.Intensity)
		entry := struct {
			LightElevDeg float64 `json:"lightElevDeg"`
			Threshold     float64 `json:"threshold"`
			WhiteCount    int     `json:"whiteCount"`
			MaskFile      string  `json:"maskFile"`
			RenderFile    string  `json:"renderFile"`
		}{deg, p.Threshold, p.WhiteCount, maskName, renderName}
		report.Calibration = append(report.Calibration, entry)
		fmt.Printf("[标定] 光照角 %3g° 阈值=%.6f 白像素=%d\n", deg, p.Threshold, p.WhiteCount)
	}

	// 二次曲线拟合：三个标定点唯一求解。
	curve, err := calibrate.FitQuadratic(points)
	if err != nil {
		fmt.Fprintln(os.Stderr, "曲线拟合失败:", err)
		os.Exit(1)
	}
	report.Curve.A, report.Curve.B, report.Curve.C = curve.A, curve.B, curve.C
	fmt.Printf("[拟合] T(t) = %.8e*t^2 %+.8e*t %+.8f\n", curve.A, curve.B, curve.C)

	// 在线预判 + 渲染对照：预判只输入光照角，不跑渲染。
	for _, deg := range checkDeg {
		pred, err := calibrate.Predict(curve, sh, nf, deg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "预判角 %g 度失败: %v\n", deg, err)
			os.Exit(1)
		}
		frame := sh.Render(nf, deg*math.Pi/180)
		dev := calibrate.Compare(pred.Mask, frame, h.Size)

		tag := fmt.Sprintf("check_angle%03d", int(math.Round(deg)))
		predName := tag + "_pred.pgm"
		renderMaskName := tag + "_render_white.pgm"
		renderName := tag + "_render.pgm"
		devName := tag + "_deviation.pgm"
		writeMaskPGM(filepath.Join(*outdir, predName), h.Size, pred.Mask)
		writeMaskPGM(filepath.Join(*outdir, renderMaskName), h.Size, frame.White)
		writeGrayPGM(filepath.Join(*outdir, renderName), h.Size, frame.Intensity)
		writeDeviationPGM(filepath.Join(*outdir, devName), h.Size, pred.Mask, frame.White, dev)

		report.Validation = append(report.Validation, struct {
			LightElevDeg       float64 `json:"lightElevDeg"`
			ThresholdFromCurve  float64 `json:"thresholdFromCurve"`
			CriticalAngleDeg    float64 `json:"criticalAngleDeg"`
			PredictedCount      int     `json:"predictedCount"`
			RenderedWhiteCount  int     `json:"renderedWhiteCount"`
			OnlyPredicted       int     `json:"onlyPredicted"`
			OnlyRendered        int     `json:"onlyRendered"`
			MaxOffsetCells      float64 `json:"maxOffsetCells"`
			OverTwoCells        int     `json:"overTwoCells"`
			PredMaskFile        string  `json:"predMaskFile"`
			RenderMaskFile      string  `json:"renderMaskFile"`
			RenderFile          string  `json:"renderFile"`
			DeviationFile       string  `json:"deviationMaskFile"`
		}{
			LightElevDeg:      deg,
			ThresholdFromCurve: pred.Threshold,
			CriticalAngleDeg:   pred.CriticalAngle * 180 / math.Pi,
			PredictedCount:     pred.Count,
			RenderedWhiteCount: frame.WhiteCount,
			OnlyPredicted:      dev.OnlyPredicted,
			OnlyRendered:       dev.OnlyRendered,
			MaxOffsetCells:     dev.MaxOffsetCells,
			OverTwoCells:       dev.OverTwoCells,
			PredMaskFile:       predName,
			RenderMaskFile:     renderMaskName,
			RenderFile:         renderName,
			DeviationFile:      devName,
		})
		fmt.Printf("[对照] 光照角 %3g° 预判=%d 渲染白像素=%d 临界夹角=%.4f° 偏差>2格=%d 最大偏差=%.2f格\n",
			deg, pred.Count, frame.WhiteCount, pred.CriticalAngle*180/math.Pi, dev.OverTwoCells, dev.MaxOffsetCells)
	}

	reportBytes, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, "报告序列化失败:", err)
		os.Exit(1)
	}
	reportPath := filepath.Join(*outdir, "report.json")
	if err := os.WriteFile(reportPath, append(reportBytes, '\n'), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "报告写入失败:", err)
		os.Exit(1)
	}
	fmt.Printf("留档目录 %s 报告 %s\n", *outdir, reportPath)
}

func parseAngles(s string, want int) ([]float64, error) {
	parts := strings.Split(s, ",")
	if len(parts) != want {
		return nil, fmt.Errorf("需要 %d 个角度，实际 %d 个", want, len(parts))
	}
	out := make([]float64, 0, want)
	for _, p := range parts {
		v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
		if err != nil {
			return nil, err
		}
		if v <= 0 || v >= 90 {
			return nil, fmt.Errorf("光照仰角必须在零到九十度之间: %v", v)
		}
		out = append(out, v)
	}
	return out, nil
}

func pgmHeader(size int) []byte {
	return []byte(fmt.Sprintf("P5\n%d %d\n255\n", size, size))
}

// writeMaskPGM 写布尔掩码：真为二五五，假为零。
func writeMaskPGM(path string, size int, mask []bool) {
	out := pgmHeader(size)
	for _, v := range mask {
		if v {
			out = append(out, 255)
		} else {
			out = append(out, 0)
		}
	}
	if err := os.WriteFile(path, out, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "写掩码失败:", err)
		os.Exit(1)
	}
}

// writeGrayPGM 写浮点强度灰度图，顶格二五五即白像素。
func writeGrayPGM(path string, size int, intensity []float64) {
	out := pgmHeader(size)
	for _, v := range intensity {
		g := int(v*255 + 0.5)
		if g < 0 {
			g = 0
		}
		if g > 255 {
			g = 255
		}
		out = append(out, byte(g))
	}
	if err := os.WriteFile(path, out, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "写灰度图失败:", err)
		os.Exit(1)
	}
}

// writeDeviationPGM 写偏差标注图：偏差超两格二五五，两格内不一致一二八，一致零。
// 对方集合为空（无最近像素）时按超两格处理并同样标二五五。
func writeDeviationPGM(path string, size int, pred, rendered []bool, dev calibrate.Deviation) {
	out := pgmHeader(size)
	for i := range pred {
		switch {
		case dev.OverMask[i]:
			out = append(out, 255)
		case pred[i] != rendered[i]:
			out = append(out, 128)
		default:
			out = append(out, 0)
		}
	}
	if err := os.WriteFile(path, out, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "写偏差图失败:", err)
		os.Exit(1)
	}
}
