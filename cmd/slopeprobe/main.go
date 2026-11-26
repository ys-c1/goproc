// slopeprobe 对两个画幅生成坡度图，逐像素比对边界延拓补偿前后的差异。
//
// 用法：
//
//	go run ./cmd/slopeprobe            # 默认输出到 artifacts/slope
//	go run ./cmd/slopeprobe -out DIR
//
// 每个画幅输出：
//   - <name>_before.png   补偿前的坡度图
//   - <name>_after.png    补偿后的坡度图
//   - <name>_compare.png  两版并置
//   - <name>_diff.png     色阶改变的像素（橙色），其余保留补偿后色阶
//   - <name>_ring.png     只保留边缘一圈的两版对照
//
// 两画幅的关键数字另写入 report.json。
//
// 真值口径：底层 fbm 是连续函数，画幅外也有定义。用很小步长的中心差
// （hh=1e-4）采样其导数作为「解析真值」，再把离散中心差与之相比。
// 内部像素与环内像素都按同一真值口径计误差，二者可直接比较。
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"strings"

	"goproc/api"
	"goproc/internal/fbm"
	"goproc/internal/noise"
	"goproc/internal/slope"
)

const levels = 256

// analyticMag 返回连续 fbm 在整数格点处的真实坡度模长。
// 归一化与 api.Generate 一致：用画幅内的极值做线性缩放。
func analyticMag(opt api.Options) []float64 {
	src := noise.New(opt.Seed)
	n := opt.Size
	eval := func(x, y float64) float64 {
		return fbm.Eval(src, x*opt.Frequency, y*opt.Frequency,
			opt.Octaves, opt.Lacunarity, opt.Gain)
	}
	lo, hi := math.Inf(1), math.Inf(-1)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			v := eval(float64(x), float64(y))
			lo, hi = math.Min(lo, v), math.Max(hi, v)
		}
	}
	scale := 1.0
	if hi > lo {
		scale = 1 / (hi - lo)
	}
	const hh = 1e-4
	out := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			xf, yf := float64(x), float64(y)
			gx := (eval(xf+hh, yf) - eval(xf-hh, yf)) / (2 * hh) * scale
			gy := (eval(xf, yf+hh) - eval(xf, yf-hh)) / (2 * hh) * scale
			out[y*n+x] = math.Hypot(gx, gy)
		}
	}
	return out
}

// errStats 返回一组相对解析真值的误差统计。
type errStats struct {
	rms      float64 // 均方根误差
	meanSign float64 // 有符号平均误差（>0 表示估计偏陡）
}

func errStatsFor(got, truth []float64, n int, include func(x, y int) bool) errStats {
	var ss, sign float64
	var c int
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			if !include(x, y) {
				continue
			}
			d := got[y*n+x] - truth[y*n+x]
			ss += d * d
			sign += d
			c++
		}
	}
	if c == 0 {
		return errStats{}
	}
	return errStats{rms: math.Sqrt(ss / float64(c)), meanSign: sign / float64(c)}
}

func mags(f *slope.Field) []float64 {
	out := make([]float64, len(f.Gx))
	for i := range out {
		out[i] = f.Mag2(i)
	}
	return out
}

func encodeMag(mag []float64, maxMag float64) []uint8 {
	out := make([]uint8, len(mag))
	if maxMag <= 0 {
		return out
	}
	for i, m := range mag {
		v := m / maxMag
		if v < 0 {
			v = 0
		}
		if v > 1 {
			v = 1
		}
		out[i] = uint8(v*float64(levels-1) + 0.5)
	}
	return out
}

// blueRamp 是 dataviz 参考调色板的顺序蓝阶（100->700）：浅=缓，深=陡。
var blueRamp = [][3]uint8{
	{0xcd, 0xe2, 0xfb}, {0xb7, 0xd3, 0xf6}, {0x9e, 0xc5, 0xf4},
	{0x86, 0xb6, 0xef}, {0x6d, 0xa7, 0xec}, {0x55, 0x98, 0xe7},
	{0x39, 0x87, 0xe5}, {0x2a, 0x78, 0xd6}, {0x25, 0x6a, 0xbf},
	{0x1c, 0x5c, 0xab}, {0x18, 0x4f, 0x95}, {0x10, 0x42, 0x81}, {0x0d, 0x36, 0x6b},
}

func colormap(v uint8) color.RGBA {
	t := float64(v) / 255 * float64(len(blueRamp)-1)
	i := int(t)
	if i >= len(blueRamp)-1 {
		i = len(blueRamp) - 2
	}
	f := t - float64(i)
	a, b := blueRamp[i], blueRamp[i+1]
	mix := func(x, y uint8) uint8 { return uint8(float64(x) + (float64(y)-float64(x))*f) }
	return color.RGBA{R: mix(a[0], b[0]), G: mix(a[1], b[1]), B: mix(a[2], b[2]), A: 255}
}

func colorImage(size int, px []uint8) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			img.SetRGBA(x, y, colormap(px[y*size+x]))
		}
	}
	return img
}

var surface = color.RGBA{0xfc, 0xfc, 0xfb, 255}
var border = color.RGBA{0x89, 0x87, 0x81, 255}

// zoom 用最近邻放大 img，四周留 margin px 底并画一条细边框。
// 边框画在留白里，不压任何像素，边缘一圈因此保持可见。
func zoom(img *image.RGBA, z, margin int) *image.RGBA {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	W, H := w*z+margin*2, h*z+margin*2
	out := image.NewRGBA(image.Rect(0, 0, W, H))
	fill(out, surface)
	for y := 0; y < h*z; y++ {
		for x := 0; x < w*z; x++ {
			out.SetRGBA(margin+x, margin+y, img.RGBAAt(x/z, y/z))
		}
	}
	x0, y0, x1, y1 := margin-2, margin-2, margin+w*z+1, margin+h*z+1
	for x := x0; x <= x1; x++ {
		out.SetRGBA(x, y0, border)
		out.SetRGBA(x, y1, border)
	}
	for y := y0; y <= y1; y++ {
		out.SetRGBA(x0, y, border)
		out.SetRGBA(x1, y, border)
	}
	return out
}

func fill(img *image.RGBA, c color.RGBA) {
	b := img.Bounds()
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			img.SetRGBA(x, y, c)
		}
	}
}

func sideBySide(a, b *image.RGBA, gap int) *image.RGBA {
	w, h := a.Bounds().Dx(), a.Bounds().Dy()
	out := image.NewRGBA(image.Rect(0, 0, w*2+gap, h))
	fill(out, surface)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			out.SetRGBA(x, y, a.RGBAAt(x, y))
			out.SetRGBA(w+gap+x, y, b.RGBAAt(x, y))
		}
	}
	return out
}

var changeColor = color.RGBA{0xeb, 0x68, 0x34, 255} // 分类色槽 2 橙

func diffImage(size int, changed []bool, base []uint8) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			i := y*size + x
			if changed[i] {
				img.SetRGBA(x, y, changeColor)
			} else {
				img.SetRGBA(x, y, colormap(base[i]))
			}
		}
	}
	return img
}

// ringOnly 把内部置为底色，只保留边缘一圈（一格宽）。
func ringOnly(px []uint8, n int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, n, n))
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			if slope.IsRing(n, x, y) {
				img.SetRGBA(x, y, colormap(px[y*n+x]))
			} else {
				img.SetRGBA(x, y, surface)
			}
		}
	}
	return img
}

type result struct {
	Name           string    `json:"name"`
	Size           int       `json:"size"`
	Ring           int       `json:"ring_pixels"`
	RawChanged     int       `json:"raw_changed_pixels"`
	LevelChanged   int       `json:"level_changed_pixels"`
	OutsideRing    int       `json:"changed_outside_ring"`
	MaxDev         float64   `json:"max_dev_after"` // 补偿前后逐像素最大偏差
	LevelMaxDiff   int       `json:"level_max_diff"`
	RingRMSBefore  float64   `json:"ring_rms_before"`
	RingRMSAfter   float64   `json:"ring_rms_after"`
	InnerRMS       float64   `json:"inner_rms"`
	RingBiasBefore float64   `json:"ring_bias_before"`
	RingBiasAfter  float64   `json:"ring_bias_after"`
	ProfileBefore  []float64 `json:"profile_before"` // 左上边缘向内 8 格的平均坡度
	ProfileAfter   []float64 `json:"profile_after"`
	ProfileInner   float64   `json:"profile_inner"` // 远离边缘的基线
}

func runFrame(name string, opt api.Options, outDir string) (result, error) {
	h, err := api.Generate(opt)
	if err != nil {
		return result{}, err
	}
	n := h.Size
	field := slope.NewScalarField(n, h.Data)
	raw := slope.Gradient(field)
	comp := slope.GradientCompensated(field)
	rm := mags(raw)
	cm := mags(comp)
	truth := analyticMag(opt)

	res := result{Name: name, Size: n, Ring: slope.RingSize(n)}

	// 两版共用同一归一化上限，色阶可直接比对。
	maxMag, _, _ := comp.MaxMag()
	if m, _, _ := raw.MaxMag(); m > maxMag {
		maxMag = m
	}
	rawPx := encodeMag(rm, maxMag)
	compPx := encodeMag(cm, maxMag)

	changed := make([]bool, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			i := y*n + x
			if rm[i] != cm[i] {
				res.RawChanged++
				if !slope.IsRing(n, x, y) {
					res.OutsideRing++
				}
			}
			if rawPx[i] != compPx[i] {
				res.LevelChanged++
				changed[i] = true
				if d := absInt(int(rawPx[i]) - int(compPx[i])); d > res.LevelMaxDiff {
					res.LevelMaxDiff = d
				}
			}
			if d := math.Abs(rm[i] - cm[i]); d > res.MaxDev {
				res.MaxDev = d
			}
		}
	}

	ringPick := func(x, y int) bool { return slope.IsRing(n, x, y) }
	innerPick := func(x, y int) bool { return !slope.IsRing(n, x, y) }
	res.InnerRMS = errStatsFor(cm, truth, n, innerPick).rms
	ringBefore := errStatsFor(rm, truth, n, ringPick)
	ringAfter := errStatsFor(cm, truth, n, ringPick)
	res.RingRMSBefore = ringBefore.rms
	res.RingRMSAfter = ringAfter.rms
	res.RingBiasBefore = ringBefore.meanSign
	res.RingBiasAfter = ringAfter.meanSign
	res.ProfileBefore, res.ProfileAfter, res.ProfileInner = profiles(cm, rm, n)

	const z = 3
	before := zoom(colorImage(n, rawPx), z, 4)
	after := zoom(colorImage(n, compPx), z, 4)
	write := func(suffix string, img image.Image) error {
		return writePNG(filepath.Join(outDir, name+"_"+suffix+".png"), img)
	}
	if err := write("before", before); err != nil {
		return res, err
	}
	if err := write("after", after); err != nil {
		return res, err
	}
	if err := write("compare", sideBySide(before, after, 10)); err != nil {
		return res, err
	}
	if err := write("diff", diffImage(n, changed, compPx)); err != nil {
		return res, err
	}
	if err := write("ring", sideBySide(zoom(ringOnly(rawPx, n), 4, 4), zoom(ringOnly(compPx, n), 4, 4), 10)); err != nil {
		return res, err
	}
	return res, nil
}

// profiles 返回从左边 x=0 与上边 y=0 向内前 8 格、及右边/下边向内的平均坡度，
// 对两个方向取平均；ProfileInner 是远离边缘（第 12 格起）的基线。
func profiles(after, before []float64, n int) (pb, pa []float64, baseline float64) {
	const depth = 8
	pb = make([]float64, depth)
	pa = make([]float64, depth)
	for k := 0; k < depth; k++ {
		var sb, sa float64
		var c int
		for y := 0; y < n; y++ {
			sb += before[y*n+k]
			sa += after[y*n+k]
			sb += before[y*n+(n-1-k)]
			sa += after[y*n+(n-1-k)]
			c += 2
		}
		for x := 0; x < n; x++ {
			sb += before[k*n+x]
			sa += after[k*n+x]
			sb += before[(n-1-k)*n+x]
			sa += after[(n-1-k)*n+x]
			c += 2
		}
		pb[k], pa[k] = sb/float64(c), sa/float64(c)
	}
	var s float64
	var c int
	for y := 0; y < n; y++ {
		for x := 12; x < n-12; x++ {
			s += after[y*n+x] + after[x*n+y]
			c += 2
		}
	}
	baseline = s / float64(c)
	return pb, pa, baseline
}

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func writePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func main() {
	outDir := flag.String("out", "artifacts/slope", "输出目录")
	flag.Parse()
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	frames := []struct {
		name string
		opt  api.Options
	}{
		{"frame128", api.Options{Size: 128, Octaves: 4, Lacunarity: 2, Gain: 0.5, Seed: 7, Frequency: 0.02}},
		{"frame96", api.Options{Size: 96, Octaves: 4, Lacunarity: 2, Gain: 0.5, Seed: 7, Frequency: 0.02}},
	}
	var results []result
	for _, fr := range frames {
		res, err := runFrame(fr.name, fr.opt, *outDir)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		results = append(results, res)
	}

	// 两画幅并排成两栏：每栏列出该画幅的关键数字。
	fmt.Printf("%-20s %16s %16s\n", "", results[0].Name, results[1].Name)
	fmt.Printf("%-20s %16d %16d\n", "边长", results[0].Size, results[1].Size)
	fmt.Printf("%-20s %16d %16d\n", "边缘一圈(像素)", results[0].Ring, results[1].Ring)
	fmt.Printf("%-20s %16d %16d\n", "色阶改变像素数", results[0].LevelChanged, results[1].LevelChanged)
	fmt.Printf("%-20s %16d %16d\n", "  其中落在圈外", results[0].OutsideRing, results[1].OutsideRing)
	fmt.Printf("%-20s %16.4e %16.4e\n", "补偿后最大偏差", results[0].MaxDev, results[1].MaxDev)
	fmt.Printf("%-20s %16d %16d\n", "最大色阶差", results[0].LevelMaxDiff, results[1].LevelMaxDiff)
	fmt.Println()
	fmt.Printf("%-20s %16s %16s\n", "相对解析真值", results[0].Name, results[1].Name)
	fmt.Printf("%-20s %16.3e %16.3e\n", "内部 RMS", results[0].InnerRMS, results[1].InnerRMS)
	fmt.Printf("%-20s %16.3e %16.3e\n", "环 RMS 补偿前", results[0].RingRMSBefore, results[1].RingRMSBefore)
	fmt.Printf("%-20s %16.3e %16.3e\n", "环 RMS 补偿后", results[0].RingRMSAfter, results[1].RingRMSAfter)
	fmt.Printf("%-20s %+16.3e %+16.3e\n", "环有符号偏 补偿前", results[0].RingBiasBefore, results[1].RingBiasBefore)
	fmt.Printf("%-20s %+16.3e %+16.3e\n", "环有符号偏 补偿后", results[0].RingBiasAfter, results[1].RingBiasAfter)
	fmt.Println()
	// 边缘向内剖面：用 frame128 展示（两画幅趋势一致）。
	r0 := results[0]
	fmt.Printf("%s 边缘向内坡度剖面（左上右下四边平均）\n", r0.Name)
	fmt.Printf("%-8s %12s %12s\n", "向内格数", "补偿前", "补偿后")
	for k := range r0.ProfileBefore {
		fmt.Printf("%-8d %12.5f %12.5f\n", k, r0.ProfileBefore[k], r0.ProfileAfter[k])
	}
	fmt.Printf("%-8s %12.5f %12.5f\n", "内部基线", r0.ProfileInner, r0.ProfileInner)

	if err := writeJSON(filepath.Join(*outDir, "report.json"), results); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if err := writeCSV(filepath.Join(*outDir, "report.csv"), results); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	ok := true
	for _, r := range results {
		if r.OutsideRing != 0 || r.LevelChanged > r.Ring {
			ok = false
		}
	}
	if !ok {
		fmt.Fprintln(os.Stderr, "FAIL: 改变像素越出边缘一圈")
		os.Exit(1)
	}
	fmt.Println("\nOK: 全部改变像素均落在边缘一圈内")
}

// writeCSV 把关键两栏（改变像素数、补偿引入最大偏差）写成两画幅并排的 CSV。
func writeCSV(path string, results []result) error {
	var b strings.Builder
	b.WriteString("指标," + results[0].Name + "," + results[1].Name + "\n")
	row := func(k string, f func(r result) string) {
		b.WriteString(k)
		for _, r := range results {
			b.WriteString("," + f(r))
		}
		b.WriteString("\n")
	}
	row("边缘一圈像素数", func(r result) string { return fmt.Sprint(r.Ring) })
	row("色阶改变像素数", func(r result) string { return fmt.Sprint(r.LevelChanged) })
	row("改变像素落在圈外数", func(r result) string { return fmt.Sprint(r.OutsideRing) })
	row("补偿后最大偏差", func(r result) string { return fmt.Sprintf("%.6e", r.MaxDev) })
	row("最大色阶差", func(r result) string { return fmt.Sprint(r.LevelMaxDiff) })
	return os.WriteFile(path, []byte(b.String()), 0o644)
}

func writeJSON(path string, results []result) error {
	b, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}
