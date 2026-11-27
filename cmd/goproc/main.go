package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"

	"goproc/api"
	"goproc/internal/calibration"
	"goproc/internal/hmap"
	"goproc/internal/resample"
)

// CalibFile 是标定落盘文件，predict 子命令只读它即可在线预判。
type CalibFile struct {
	Source   string              `json:"source"`
	Size     int                 `json:"size"`
	Window   int                 `json:"localVarianceWindow"`
	Scales   []float64           `json:"calibrationScales"`
	Curves   []calibration.Curve `json:"curves"`
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "gen":
		cmdGen(os.Args[2:])
	case "calibrate":
		cmdCalibrate(os.Args[2:])
	case "predict", "invert":
		cmdPredict(os.Args[2:])
	default:
		usage()
		os.Exit(2)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage:")
	fmt.Fprintln(os.Stderr, "  goproc gen --size N --seed S --octaves K --out F")
	fmt.Fprintln(os.Stderr, "  goproc calibrate [--in F.pgm | --size 512 --seed 7 --octaves 4] \\")
	fmt.Fprintln(os.Stderr, "                   --window 9 --outdir D")
	fmt.Fprintln(os.Stderr, "  goproc predict --calib D/calibration.json --kernel box|triangle --scale S")
	fmt.Fprintln(os.Stderr, "  goproc invert  --calib D/calibration.json --kernel box|triangle --fog F")
}

func cmdGen(args []string) {
	fs := flag.NewFlagSet("gen", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 1, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	out := fs.String("out", "map.pgm", "输出文件")
	must(fs.Parse(args))
	h, err := api.Generate(api.Options{Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5, Seed: *seed, Frequency: 0.02})
	must(err)
	must(os.WriteFile(*out, h.ToPGM(), 0o644))
	fmt.Printf("wrote %s size=%d\n", *out, h.Size)
}

func loadOrig(in string, size int, seed uint64, octaves int) (int, []float64, string) {
	if in != "" {
		raw, err := os.ReadFile(in)
		must(err)
		n, data, err := hmap.DecodePGM(raw)
		must(err)
		return n, data, in
	}
	h, err := api.Generate(api.Options{Size: size, Octaves: octaves, Lacunarity: 2, Gain: 0.5, Seed: seed, Frequency: 0.02})
	must(err)
	return h.Size, h.Data, fmt.Sprintf("generated:size=%d,seed=%d,octaves=%d", size, seed, octaves)
}

func cmdCalibrate(args []string) {
	fs := flag.NewFlagSet("calibrate", flag.ContinueOnError)
	in := fs.String("in", "", "输入 PGM；缺省时现场生成")
	size := fs.Int("size", 512, "现场生成边长")
	seed := fs.Uint64("seed", 7, "现场生成种子")
	octaves := fs.Int("octaves", 4, "现场生成层数")
	window := fs.Int("window", 17, "原图局部方差窗口边长（奇数）")
	outdir := fs.String("outdir", "artifacts/calib", "输出目录")
	must(fs.Parse(args))
	if *window <= 0 || *window%2 == 0 {
		fatal("--window 必须为正奇数")
	}

	n, orig, source := loadOrig(*in, *size, *seed, *octaves)
	if err := os.MkdirAll(*outdir, 0o755); err != nil {
		fatal(err.Error())
	}

	kernels := []resample.Kind{resample.Box, resample.Triangle}
	curves := make([]calibration.Curve, 0, 2)
	for _, k := range kernels {
		curves = append(curves, calibration.Calibrate(orig, n, k, *window))
	}

	calib := CalibFile{
		Source: source, Size: n, Window: *window,
		Scales: calibration.CalScales(), Curves: curves,
	}
	writeJSON(*outdir+"/calibration.json", calib)
	writeCalibrationCSV(*outdir+"/calibration_points.csv", curves)

	// 在线验证：预测只吃缩放比，另跑实测逐档对照。
	valRows := make([][]string, 0)
	valRows = append(valRows, []string{"kernel", "scale", "thumb_size", "in_calib_range", "R_pred", "R_meas", "F_pred", "F_meas", "rel_dev_R", "rel_dev_F", "flag"})
	for _, k := range kernels {
		c := curveOf(curves, k.String())
		for _, s := range calibration.ValScales() {
			mp := calibration.MeasureAt(orig, n, s, k, *window)
			rPred, fPred := c.PredictR(mp.Scale), c.PredictF(mp.Scale)
			relR := relErr(rPred, mp.R)
			relF := relErr(fPred, mp.F)
			inRange := "yes"
			flag := ""
			if !c.InCalibratedRange(mp.Scale) {
				inRange = "no(extrapolation)"
			}
			if relR > 0.10 || relF > 0.10 {
				if inRange == "yes" {
					flag = "DEV_OVER_10PCT"
				} else {
					flag = "DEV_OVER_10PCT:标定区间外外推，低缩放比下局部方差存在度量下限，曲线在此区变平"
				}
			}
			valRows = append(valRows, []string{
				k.String(), f4(mp.Scale), strconv.Itoa(mp.ThumbSize), inRange,
				f4(rPred), f4(mp.R), f4(fPred), f4(mp.F),
				f4(relR), f4(relF), flag,
			})
		}
	}
	writeCSV(*outdir+"/validation.csv", valRows)
	fmt.Printf("calibrated source=%s size=%d window=%d outdir=%s\n", source, n, *window, *outdir)
	for _, c := range curves {
		fmt.Printf("  %-8s A=%.6f gamma=%.6f R2=%.6f\n", c.Kernel, c.A, c.Gamma, c.RSq)
	}
}

func cmdPredict(args []string) {
	fs := flag.NewFlagSet("predict", flag.ContinueOnError)
	calibPath := fs.String("calib", "artifacts/calib/calibration.json", "标定文件")
	kernel := fs.String("kernel", "box", "采样核: box|triangle")
	scale := fs.Float64("scale", 0, "缩放比 (0,1]")
	fog := fs.Float64("fog", -1, "反解模式：给定雾化程度，输出对应缩放比；与 --scale 二选一")
	must(fs.Parse(args))

	raw, err := os.ReadFile(*calibPath)
	must(err)
	var calib CalibFile
	must(json.Unmarshal(raw, &calib))
	c := curveOf(calib.Curves, *kernel)
	if *fog >= 0 {
		if *fog < 0 || *fog >= 1 {
			fatal("--fog 必须在 [0,1)")
		}
		fmt.Printf("kernel=%s fog=%.4f -> scale=%.6f (R=%.6f)\n", *kernel, *fog, c.InvertScaleF(*fog), 1-*fog)
		return
	}
	if *scale <= 0 || *scale > 1 {
		fatal("--scale 必须在 (0,1]")
	}
	note := ""
	if !c.InCalibratedRange(*scale) {
		note = " [注意: 标定区间外，按端段斜率外推]"
	}
	fmt.Printf("kernel=%s scale=%.4f -> R=%.6f fog=%.6f%s\n", *kernel, *scale, c.PredictR(*scale), c.PredictF(*scale), note)
	fmt.Printf("(紧凑幂律模型: R=%.6f fog=%.6f, gamma=%.4f)\n", c.PowerLawR(*scale), 1-c.PowerLawR(*scale), c.Gamma)
}

func curveOf(cs []calibration.Curve, name string) calibration.Curve {
	for _, c := range cs {
		if c.Kernel == name {
			return c
		}
	}
	fatal("未知采样核: " + name)
	return calibration.Curve{}
}

func writeCalibrationCSV(path string, curves []calibration.Curve) {
	rows := [][]string{{"kernel", "k_index", "nominal_scale", "actual_scale", "thumb_size", "R_meas", "F_meas"}}
	nominal := calibration.CalScales()
	for _, c := range curves {
		for i, p := range c.Points {
			rows = append(rows, []string{
				c.Kernel, strconv.Itoa(i), f4(nominal[i]), f4(p.Scale),
				strconv.Itoa(p.ThumbSize), f4(p.R), f4(p.F),
			})
		}
	}
	writeCSV(path, rows)
}

func relErr(pred, meas float64) float64 {
	if meas == 0 {
		if pred == 0 {
			return 0
		}
		return 1
	}
	return abs(pred-meas) / meas
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func f4(x float64) string { return strconv.FormatFloat(x, 'f', 6, 64) }

func writeJSON(path string, v any) {
	b, err := json.MarshalIndent(v, "", "  ")
	must(err)
	must(os.WriteFile(path, append(b, '\n'), 0o644))
}

func writeCSV(path string, rows [][]string) {
	f, err := os.Create(path)
	must(err)
	defer f.Close()
	w := csv.NewWriter(f)
	must(w.WriteAll(rows))
	w.Flush()
	must(w.Error())
}

func must(err error) {
	if err != nil {
		fatal(err.Error())
	}
}

func fatal(msg string) {
	fmt.Fprintln(os.Stderr, "goproc:", msg)
	os.Exit(1)
}
