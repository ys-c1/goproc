// 饱和占比研究:对两档分布参数,在归一化前用上尾双对数拟合预判
// 灰度 255 档的像素占比,再与实际归一化量化结果逐档比对。
//
// 口径:饱和像素 = 量化后灰度为 255 的像素(顶档及以上,8 位下即 g=255)。
// 由 hmap.EncodePGM 的舍入规则 g=int(x*255+0.5) 可知
//
//	g = 255  <=>  x >= 1 - eps,  eps = 0.5/255 = 1/510
//
// 归一化是仿射变换 x=(V-lo)/(hi-lo),故等价地
//
//	饱和 <=>  V >= t,  t = hi - eps*(hi-lo)
//
// 上尾模型(有界分布的端点幂律): S(v)=P(V>v) ≈ C*(hi-v)^beta,
// 对 (ln(hi-v), ln S(v)) 在指定分位区间内做最小二乘(双对数拟合)。
// 每档用两个区间各拟合一次:
//
//	W(宽区间): 分位 [0.90, 0.999]   —— 远离阈值,检验浅区间外推
//	D(深区间): 分位 [0.999, 0.99995] —— 逼近阈值亏量尺度
//
// 预判: p_pred = C*(eps*R)^beta, R = hi-lo;
// 先验式: p_prior = (1/N)*(1 + eps*R*(C*N)^(1/beta))^beta。
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"goproc/internal/fbm"
	"goproc/internal/hmap"
	"goproc/internal/noise"
)

const (
	epsQuant = 0.5 / 255.0 // 量化口径:灰度 255 对应归一化值 >= 1-epsQuant
	baseFreq = 0.02
	lacun    = 2.0
	histBins = 256
)

type tierSpec struct {
	name    string
	octaves int
	gain    float64
}

type fitInterval struct {
	name string
	qLo  float64
	qHi  float64
}

var intervals = []fitInterval{
	{"W", 0.90, 0.999},
	{"D", 0.999, 0.99995},
}

type fitResult struct {
	interval fitInterval
	fitVLo   float64 // 拟合区间对应的原值下界
	fitVHi   float64 // 拟合区间对应的原值上界
	fitPts   int
	beta     float64 // 尾部指数(双对数斜率)
	c        float64 // 尾部系数
	r2       float64 // 双对数线性度
	pPred    float64 // 经验式预判 C*(eps*R)^beta
	pPrior   float64 // 先验式 (1/N)(1+eps*R*(C*N)^(1/beta))^beta
}

type tierResult struct {
	tier     tierSpec
	n        int
	lo, hi   float64
	r        float64
	t        float64 // 饱和临界原始值
	tQuant   float64 // t 的经验分位
	fits     []fitResult
	pMeas    float64 // 实测占比
	satCount int
}

func main() {
	outDir := flag.String("out", "docs/saturation", "结果与直方图存档目录")
	size := flag.Int("size", 512, "边长像素数")
	seed := flag.Uint64("seed", 7, "种子")
	flag.Parse()

	tiers := []tierSpec{
		{"A_oct1_g05", 1, 0.5},
		{"B_oct8_g08", 8, 0.8},
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fatal(err)
	}

	var results []tierResult
	for _, tier := range tiers {
		res, err := runTier(*outDir, tier, *size, *seed)
		if err != nil {
			fatal(err)
		}
		results = append(results, res)
	}
	if err := writeResultsTSV(filepath.Join(*outDir, "results.tsv"), results); err != nil {
		fatal(err)
	}
	printTable(results)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "satstudy:", err)
	os.Exit(1)
}

func runTier(outDir string, tier tierSpec, size int, seed uint64) (tierResult, error) {
	n := size * size
	src := noise.New(seed)
	raw := make([]float64, n)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			raw[y*size+x] = fbm.Eval(src, float64(x)*baseFreq, float64(y)*baseFreq,
				tier.octaves, lacun, tier.gain)
		}
	}

	sorted := make([]float64, n)
	copy(sorted, raw)
	sort.Float64s(sorted)
	lo, hi := sorted[0], sorted[n-1]
	r := hi - lo
	if r <= 0 {
		return tierResult{}, fmt.Errorf("档位 %s: 退化分布 hi==lo", tier.name)
	}
	t := hi - epsQuant*r
	tIdx := sort.SearchFloat64s(sorted, t)

	// ---- 双对数拟合: ln S(v) ~ ln C + beta*ln(hi-v), 按区间分别拟合 ----
	// 绘图点: 拟合点全收, 区间外按 stride 抽稀
	stride := 1
	if n/2 > 3000 {
		stride = n / 2 / 3000
	}
	type accum struct{ sx, sy, sxx, syy, sxy float64; pts int }
	acc := make([]accum, len(intervals))
	var plotX, plotY []float64
	var plotFlag []int // 0=区间外 1=W 2=D
	for i, v := range sorted {
		q := (float64(i) + 0.5) / float64(n)
		if q < 0.5 {
			continue
		}
		d := hi - v
		if d <= 0 {
			continue
		}
		ccdf := (float64(n-i) - 0.5) / float64(n)
		lx, ly := math.Log(d), math.Log(ccdf)
		flag := 0
		for k, iv := range intervals {
			if q >= iv.qLo && q <= iv.qHi {
				a := &acc[k]
				a.sx += lx
				a.sy += ly
				a.sxx += lx * lx
				a.syy += ly * ly
				a.sxy += lx * ly
				a.pts++
				flag = k + 1 // D 覆盖 W 的标记(两区间不相交, 顺序无所谓)
			}
		}
		if flag > 0 || i%stride == 0 {
			plotX = append(plotX, lx)
			plotY = append(plotY, ly)
			plotFlag = append(plotFlag, flag)
		}
	}

	res := tierResult{
		tier: tier, n: n, lo: lo, hi: hi, r: r, t: t,
		tQuant: float64(tIdx) / float64(n),
	}
	for k, iv := range intervals {
		a := acc[k]
		if a.pts < 3 {
			return tierResult{}, fmt.Errorf("档位 %s 区间 %s: 拟合点不足 (%d)", tier.name, iv.name, a.pts)
		}
		nf := float64(a.pts)
		beta := (nf*a.sxy - a.sx*a.sy) / (nf*a.sxx - a.sx*a.sx)
		icept := (a.sy - beta*a.sx) / nf
		c := math.Exp(icept)
		r2 := (nf*a.sxy - a.sx*a.sy) * (nf*a.sxy - a.sx*a.sy) /
			((nf*a.sxx - a.sx*a.sx) * (nf*a.syy - a.sy*a.sy))
		fr := fitResult{
			interval: iv,
			fitVLo:   sorted[int(iv.qLo*float64(n))],
			fitVHi:   sorted[int(iv.qHi*float64(n))],
			fitPts:   a.pts,
			beta:     beta, c: c, r2: r2,
			pPred: c * math.Pow(epsQuant*r, beta),
			pPrior: (1.0 / float64(n)) *
				math.Pow(1+epsQuant*r*math.Pow(c*float64(n), 1.0/beta), beta),
		}
		res.fits = append(res.fits, fr)
		fmt.Printf("档位 %s 区间 %s: beta=%.4f C=%.4g R2=%.5f 拟合点数=%d 值域[%.4f, %.4f]\n",
			tier.name, iv.name, beta, c, r2, a.pts, fr.fitVLo, fr.fitVHi)
	}

	// ---- 实测:按生产代码同样的规则归一化 + 量化 ----
	norm := make([]float64, n)
	copy(norm, raw)
	hmap.Normalize(norm)
	var grayHist [256]int
	sat := 0
	for _, x := range norm {
		g := int(x*255 + 0.5)
		if g < 0 {
			g = 0
		}
		if g > 255 {
			g = 255
		}
		grayHist[g]++
		if g == 255 {
			sat++
		}
	}
	res.pMeas = float64(sat) / float64(n)
	res.satCount = sat

	// ---- 直方图存档 ----
	rawHist := make([]int, histBins)
	binW := r / histBins
	for _, v := range raw {
		b := int((v - lo) / binW)
		if b >= histBins {
			b = histBins - 1
		}
		if b < 0 {
			b = 0
		}
		rawHist[b]++
	}
	prefix := filepath.Join(outDir, "tier"+tier.name)
	if err := writeHistCSV(prefix+"_raw_hist.csv", lo, binW, rawHist); err != nil {
		return tierResult{}, err
	}
	if err := writeBarPNG(prefix+"_raw_hist.png", rawHist); err != nil {
		return tierResult{}, err
	}
	if err := writeGrayCSV(prefix+"_gray_hist.csv", grayHist); err != nil {
		return tierResult{}, err
	}
	if err := writeBarPNG(prefix+"_gray_hist.png", grayHist[:]); err != nil {
		return tierResult{}, err
	}
	if err := writeFitCSV(prefix+"_tailfit.csv", plotX, plotY, plotFlag); err != nil {
		return tierResult{}, err
	}
	if err := writeFitPNG(prefix+"_tailfit.png", plotX, plotY, plotFlag, res.fits); err != nil {
		return tierResult{}, err
	}
	return res, nil
}

func printTable(results []tierResult) {
	fmt.Println("\n档位\t区间\tbeta\tp_pred\tp_prior\tp_meas\t饱和像素\t相对偏差")
	for _, res := range results {
		for _, fr := range res.fits {
			gap := math.Abs(fr.pPred-res.pMeas) / res.pMeas
			fmt.Printf("%s\t%s\t%.3f\t%.3e\t%.3e\t%.3e\t%d\t%.1f%%\n",
				res.tier.name, fr.interval.name, fr.beta,
				fr.pPred, fr.pPrior, res.pMeas, res.satCount, gap*100)
		}
	}
}

func writeResultsTSV(path string, results []tierResult) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	w.Comma = '\t'
	head := []string{"tier", "octaves", "gain", "interval", "fit_qlo", "fit_qhi",
		"fit_vlo", "fit_vhi", "fit_points", "N", "lo", "hi", "range", "eps",
		"t", "t_quantile", "beta", "C", "r2", "p_pred", "p_prior",
		"p_meas", "sat_count", "rel_gap"}
	if err := w.Write(head); err != nil {
		return err
	}
	ff := func(v float64) string { return strconv.FormatFloat(v, 'g', 10, 64) }
	for _, res := range results {
		for _, fr := range res.fits {
			gap := math.Abs(fr.pPred-res.pMeas) / res.pMeas
			row := []string{
				res.tier.name, strconv.Itoa(res.tier.octaves),
				strconv.FormatFloat(res.tier.gain, 'g', 4, 64),
				fr.interval.name, ff(fr.interval.qLo), ff(fr.interval.qHi),
				ff(fr.fitVLo), ff(fr.fitVHi), strconv.Itoa(fr.fitPts),
				strconv.Itoa(res.n), ff(res.lo), ff(res.hi), ff(res.r), ff(epsQuant),
				ff(res.t), ff(res.tQuant), ff(fr.beta), ff(fr.c), ff(fr.r2),
				ff(fr.pPred), ff(fr.pPrior), ff(res.pMeas),
				strconv.Itoa(res.satCount), ff(gap),
			}
			if err := w.Write(row); err != nil {
				return err
			}
		}
	}
	w.Flush()
	return w.Error()
}

func writeHistCSV(path string, lo, width float64, counts []int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	if err := w.Write([]string{"bin_lo", "bin_hi", "count"}); err != nil {
		return err
	}
	for i, c := range counts {
		a := lo + float64(i)*width
		if err := w.Write([]string{
			strconv.FormatFloat(a, 'g', 8, 64),
			strconv.FormatFloat(a+width, 'g', 8, 64),
			strconv.Itoa(c),
		}); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

func writeGrayCSV(path string, counts [256]int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	if err := w.Write([]string{"gray", "count"}); err != nil {
		return err
	}
	for g, c := range counts {
		if err := w.Write([]string{strconv.Itoa(g), strconv.Itoa(c)}); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

func writeFitCSV(path string, xs, ys []float64, flags []int) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := csv.NewWriter(f)
	if err := w.Write([]string{"log_deficit", "log_ccdf", "interval"}); err != nil {
		return err
	}
	for i := range xs {
		if err := w.Write([]string{
			strconv.FormatFloat(xs[i], 'g', 10, 64),
			strconv.FormatFloat(ys[i], 'g', 10, 64),
			strconv.Itoa(flags[i]),
		}); err != nil {
			return err
		}
	}
	w.Flush()
	return w.Error()
}

var (
	colBlack = color.RGBA{0x11, 0x11, 0x11, 0xff}
	colGray  = color.RGBA{0xaa, 0xaa, 0xaa, 0xff}
	colBlue  = color.RGBA{0x22, 0x44, 0xcc, 0xff}
	colRed   = color.RGBA{0xcc, 0x22, 0x22, 0xff}
	colWhite = color.RGBA{0xff, 0xff, 0xff, 0xff}
)

func fill(img *image.RGBA, c color.RGBA) {
	for y := 0; y < img.Bounds().Dy(); y++ {
		for x := 0; x < img.Bounds().Dx(); x++ {
			img.SetRGBA(x, y, c)
		}
	}
}

func writeBarPNG(path string, counts []int) error {
	const (
		w, h           = 780, 320
		ml, mr, mt, mb = 40, 10, 10, 24
	)
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	fill(img, colWhite)
	maxc := 1
	for _, c := range counts {
		if c > maxc {
			maxc = c
		}
	}
	pw, ph := w-ml-mr, h-mt-mb
	for x := ml; x < ml+pw; x++ {
		img.SetRGBA(x, mt+ph, colGray)
	}
	for y := mt; y <= mt+ph; y++ {
		img.SetRGBA(ml, y, colGray)
	}
	n := len(counts)
	for i, c := range counts {
		x0 := ml + i*pw/n
		x1 := ml + (i+1)*pw/n
		bh := c * ph / maxc
		for x := x0; x < x1; x++ {
			for y := mt + ph - bh; y < mt+ph; y++ {
				img.SetRGBA(x, y, colBlack)
			}
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

// writeFitPNG 绘制双对数尾部分布与两条拟合线。
// flags: 0=区间外(灰) 1=W 区间(蓝) 2=D 区间(红); 拟合线同色。
func writeFitPNG(path string, xs, ys []float64, flags []int, fits []fitResult) error {
	const (
		w, h = 640, 640
		m    = 50
	)
	if len(xs) == 0 {
		return fmt.Errorf("无拟合点可绘")
	}
	minX, maxX := xs[0], xs[0]
	minY, maxY := ys[0], ys[0]
	for i := range xs {
		minX = math.Min(minX, xs[i])
		maxX = math.Max(maxX, xs[i])
		minY = math.Min(minY, ys[i])
		maxY = math.Max(maxY, ys[i])
	}
	// 让两条拟合线都落入画面
	for _, fr := range fits {
		icept := math.Log(fr.c)
		minY = math.Min(minY, fr.beta*maxX+icept)
		maxY = math.Max(maxY, fr.beta*minX+icept)
	}
	tx := func(x float64) int { return m + int((x-minX)/(maxX-minX)*float64(w-2*m)) }
	ty := func(y float64) int { return h - m - int((y-minY)/(maxY-minY)*float64(h-2*m)) }

	img := image.NewRGBA(image.Rect(0, 0, w, h))
	fill(img, colWhite)
	for x := m; x < w-m; x++ {
		img.SetRGBA(x, h-m, colGray)
	}
	for y := m; y < h-m; y++ {
		img.SetRGBA(m, y, colGray)
	}
	for i := range xs {
		c := colGray
		if flags[i] == 1 {
			c = colBlue
		}
		if flags[i] == 2 {
			c = colRed
		}
		px, py := tx(xs[i]), ty(ys[i])
		for dx := -1; dx <= 1; dx++ {
			for dy := -1; dy <= 1; dy++ {
				img.SetRGBA(px+dx, py+dy, c)
			}
		}
	}
	for k, fr := range fits {
		c := colBlue
		if k == 1 {
			c = colRed
		}
		icept := math.Log(fr.c)
		for px := m; px < w-m; px++ {
			x := minX + float64(px-m)/float64(w-2*m)*(maxX-minX)
			py := ty(fr.beta*x + icept)
			if py >= m && py < h-m {
				img.SetRGBA(px, py, c)
				img.SetRGBA(px, py+1, c)
			}
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}
