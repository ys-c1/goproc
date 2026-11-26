// Command repeataudit 量化“相邻层共用置换表 -> 尺度同源重复山脊”的缺陷。
//
// 它做四件确定的事：
//  1. 导出 shared / layered 两种模式下第 0..MaxOct-1 层的原始未加权网格（float64 LE）；
//  2. 按频率比 rho 计算全部同源重复片的整数坐标，写 repeat_coordinates.csv；
//  3. 在这些固定坐标上计算比例对齐的 Pearson 相关，写 repeat_metrics.json；
//  4. 向标准输出打印三档层数（2/4/6）的周期 / 重复片数 / 相关系数对照表。
//
// 标记图由 analysis/make_figures.py 读取这里导出的 raw/ 与 CSV 后绘制。
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"goproc/internal/noise"
)

const PermPeriod = 256 // noise 置换表 p 的长度（perm[512] 只是 p[i&255] 的加倍）

// 审计几何：size=1024 上的 3x3 锚点，锚片边长 256。
// 锚点与边长都是 256 的倍数，因此连除 rho=2 五次后坐标仍是整数、边长 >= 8。
var centers = []int{256, 512, 768}

const anchorW = 256

var tiers = []int{2, 4, 6}

type box struct {
	Tier     int
	AnchorID string
	CX0, CY0 int
	Level    int
	CenterX  int
	CenterY  int
	X0, Y0   int
	X1, Y1   int
	Side     int
}

func (b box) row() []string {
	return []string{
		fmt.Sprintf("%d", b.Tier), b.AnchorID,
		fmt.Sprintf("%d", b.CX0), fmt.Sprintf("%d", b.CY0),
		fmt.Sprintf("%d", b.Level), fmt.Sprintf("%d", b.Level),
		fmt.Sprintf("%d", b.CenterX), fmt.Sprintf("%d", b.CenterY),
		fmt.Sprintf("%d", b.X0), fmt.Sprintf("%d", b.Y0),
		fmt.Sprintf("%d", b.X1), fmt.Sprintf("%d", b.Y1),
		fmt.Sprintf("%d", b.Side),
	}
}

// levelBox 返回锚片 c0 在第 k 层对应的同源盒：
// center_k = c0 / rho^k, side_k = anchorW / rho^k。
func levelBox(cx0, cy0, k, rho int) (cx, cy, side int) {
	d := intPow(rho, k)
	cx, cy = cx0/d, cy0/d
	side = anchorW / d
	return
}

func intPow(a, n int) int {
	r := 1
	for i := 0; i < n; i++ {
		r *= a
	}
	return r
}

func gcd(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

// layerGrid 计算第 layer 层未加权值噪声网格，坐标口径与 api/fbm 完全一致：
// Value(pixel * Frequency * lacunarity^layer)。
func layerGrid(src *noise.Source, size int, freq, lac float64, layer int) []float64 {
	f := freq * math.Pow(lac, float64(layer))
	g := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			g[y*size+x] = src.Value(float64(x)*f, float64(y)*f)
		}
	}
	return g
}

func pearson(a, b []float64) float64 {
	n := float64(len(a))
	var ma, mb float64
	for i := range a {
		ma += a[i]
		mb += b[i]
	}
	ma /= n
	mb /= n
	var saa, sbb, sab float64
	for i := range a {
		x := a[i] - ma
		y := b[i] - mb
		saa += x * x
		sbb += y * y
		sab += x * y
	}
	den := math.Sqrt(saa * sbb)
	if den == 0 {
		return math.NaN()
	}
	return sab / den
}

// pairVectors 取出第 k 层盒内按 rho 抽稀的样本 B，与第 k+1 层的对应细盒 A。
func pairVectors(layers [][]float64, size, rho int, cx0, cy0, k int) (A, B []float64) {
	ckx, cky, wk := levelBox(cx0, cy0, k, rho)
	_, _, wf := levelBox(cx0, cy0, k+1, rho)
	hg := wk / 2
	x0k, y0k := ckx-hg, cky-hg
	cfx, cfy, _ := levelBox(cx0, cy0, k+1, rho)
	hf := wf / 2
	x0f, y0f := cfx-hf, cfy-hf

	A = make([]float64, 0, wf*wf)
	B = make([]float64, 0, wf*wf)
	for j := 0; j < wf; j++ {
		for i := 0; i < wf; i++ {
			xf, yf := x0f+i, y0f+j
			A = append(A, layers[k+1][yf*size+xf])
			xb, yb := x0k+rho*i, y0k+rho*j
			B = append(B, layers[k][yb*size+xb])
		}
	}
	return A, B
}

type tierMetric struct {
	Octaves          int       `json:"octaves"`
	Ratio            int       `json:"frequency_ratio"`
	PermPeriodCells  int       `json:"perm_period_cells"`
	CoincidenceCells int       `json:"coincidence_period_cells"`
	BaseFrequency    float64   `json:"base_frequency"`
	BasePeriodPx     float64   `json:"base_period_px"`
	TopLayer         int       `json:"top_layer"`
	TopPeriodPx      float64   `json:"top_period_px"`
	Anchors          int       `json:"anchors"`
	RepeatPatches    int       `json:"repeat_patches"`
	MarkedBoxes      int       `json:"marked_boxes"`
	Pairs            int       `json:"pairs"`
	SharedCorrMean   float64   `json:"shared_corr_mean"`
	SharedCorrMin    float64   `json:"shared_corr_min"`
	SharedCorrAllOne bool      `json:"shared_corr_all_exactly_one"`
	LayeredCorrMean  float64   `json:"layered_corr_mean"`
	LayeredCorrSD    float64   `json:"layered_corr_sd"`
	LayeredCorrMin   float64   `json:"layered_corr_min"`
	LayeredCorrMax   float64   `json:"layered_corr_max"`
	SharedPairCorr   []float64 `json:"shared_pair_corr,omitempty"`
	LayeredPairCorr  []float64 `json:"layered_pair_corr,omitempty"`
}

func meanSD(v []float64) (float64, float64) {
	if len(v) == 0 {
		return math.NaN(), math.NaN()
	}
	var m float64
	for _, x := range v {
		m += x
	}
	m /= float64(len(v))
	var s float64
	for _, x := range v {
		s += (x - m) * (x - m)
	}
	return m, math.Sqrt(s / float64(len(v)))
}

func main() {
	size := flag.Int("size", 1024, "边长（审计几何按 1024 标定）")
	seed := flag.Uint64("seed", 7, "种子")
	freq := flag.Float64("freq", 0.02, "基础频率")
	rho := flag.Int("rho", 2, "层间频率比（lacunarity）")
	outdir := flag.String("outdir", "artifacts/repetition", "产物目录")
	noDump := flag.Bool("no-dump", false, "不导出原始层网格")
	flag.Parse()

	if *size != 1024 {
		fmt.Fprintln(os.Stderr, "审计坐标几何仅在 size=1024 标定")
		os.Exit(2)
	}
	if err := os.MkdirAll(filepath.Join(*outdir, "raw"), 0o755); err != nil {
		panic(err)
	}

	maxK := tiers[len(tiers)-1]

	// 两种模式的各层网格。
	sharedSrc := noise.New(*seed)
	shared := make([][]float64, maxK)
	layered := make([][]float64, maxK)
	for i := 0; i < maxK; i++ {
		shared[i] = layerGrid(sharedSrc, *size, *freq, float64(*rho), i)
		layered[i] = layerGrid(noise.NewLayer(*seed, i), *size, *freq, float64(*rho), i)
	}

	if !*noDump {
		for i := 0; i < maxK; i++ {
			for mode, g := range map[string][][]float64{"shared": shared, "layered": layered} {
				p := filepath.Join(*outdir, "raw", fmt.Sprintf("layer_%s_%d.f64", mode, i))
				if err := os.WriteFile(p, float64Bytes(g[i]), 0o644); err != nil {
					panic(err)
				}
			}
		}
	}

	// 坐标 CSV（三档都写全 K 层盒）。
	csvPath := filepath.Join(*outdir, "repeat_coordinates.csv")
	f, err := os.Create(csvPath)
	if err != nil {
		panic(err)
	}
	w := csv.NewWriter(f)
	_ = w.Write([]string{"tier", "anchor_id", "cx0", "cy0", "level", "layer",
		"center_x", "center_y", "x0", "y0", "x1", "y1", "side"})
	var rows [][]string
	for _, K := range tiers {
		for ai, cx := range centers {
			for aj, cy := range centers {
				aid := fmt.Sprintf("a%d%d", ai+1, aj+1)
				for k := 0; k < K; k++ {
					cxK, cyK, side := levelBox(cx, cy, k, *rho)
					h := side / 2
					b := box{
						Tier: K, AnchorID: aid, CX0: cx, CY0: cy, Level: k,
						CenterX: cxK, CenterY: cyK,
						X0: cxK - h, Y0: cyK - h, X1: cxK + h, Y1: cyK + h, Side: side,
					}
					rows = append(rows, b.row())
				}
			}
		}
	}
	for _, r := range rows {
		_ = w.Write(r)
	}
	w.Flush()
	f.Close()

	// 指标。
	Pi := PermPeriod / gcd(PermPeriod, *rho-1)
	var metrics []tierMetric
	for _, K := range tiers {
		tm := tierMetric{
			Octaves: K, Ratio: *rho, PermPeriodCells: PermPeriod,
			CoincidenceCells: Pi, BaseFrequency: *freq,
			BasePeriodPx:  float64(PermPeriod) / *freq,
			TopLayer:      K - 1,
			TopPeriodPx:   float64(PermPeriod) / (*freq * math.Pow(float64(*rho), float64(K-1))),
			Anchors:       len(centers) * len(centers),
			MarkedBoxes:   len(centers) * len(centers) * K,
			RepeatPatches: len(centers) * len(centers) * (K - 1),
			Pairs:         len(centers) * len(centers) * (K - 1),
		}
		var cs, cl []float64
		for _, cx := range centers {
			for _, cy := range centers {
				for k := 0; k < K-1; k++ {
					A, B := pairVectors(shared, *size, *rho, cx, cy, k)
					cs = append(cs, pearson(A, B))
					A2, B2 := pairVectors(layered, *size, *rho, cx, cy, k)
					cl = append(cl, pearson(A2, B2))
				}
			}
		}
		tm.SharedPairCorr = cs
		tm.LayeredPairCorr = cl
		tm.SharedCorrMean, _ = meanSD(cs)
		tm.SharedCorrMin = minFloat(cs)
		tm.SharedCorrAllOne = true
		for _, v := range cs {
			if math.Abs(v-1) > 1e-12 {
				tm.SharedCorrAllOne = false
			}
		}
		tm.LayeredCorrMean, tm.LayeredCorrSD = meanSD(cl)
		tm.LayeredCorrMin = minFloat(cl)
		tm.LayeredCorrMax = maxFloat(cl)
		metrics = append(metrics, tm)
	}

	jb, _ := json.MarshalIndent(map[string]any{
		"size": *size, "seed": *seed, "rho": *rho, "frequency": *freq,
		"perm_period": PermPeriod, "tiers": metrics,
	}, "", "  ")
	if err := os.WriteFile(filepath.Join(*outdir, "repeat_metrics.json"), jb, 0o644); err != nil {
		panic(err)
	}

	fmt.Println("| 层数 K | 频率比 rho | 格点重合周期 Pi (cells) | 顶层像素周期 T_(K-1) (px) | 重复片/配对数 | shared r (mean/min) | layered r (mean±sd) |")
	fmt.Println("|---:|---:|---:|---:|---:|---:|---:|")
	for _, m := range metrics {
		fmt.Printf("| %d | %d | %d | %.1f | %d | %.6f / %.6f (allOne=%v) | %+.4f ± %.4f |\n",
			m.Octaves, m.Ratio, m.CoincidenceCells, m.TopPeriodPx, m.RepeatPatches,
			m.SharedCorrMean, m.SharedCorrMin, m.SharedCorrAllOne,
			m.LayeredCorrMean, m.LayeredCorrSD)
	}
	fmt.Println("\nwrote", csvPath)
	fmt.Println("wrote", filepath.Join(*outdir, "repeat_metrics.json"))
}

func float64Bytes(g []float64) []byte {
	b := make([]byte, 8*len(g))
	for i, v := range g {
		binaryLittleEndianPutUint64(b[i*8:], math.Float64bits(v))
	}
	return b
}

func binaryLittleEndianPutUint64(b []byte, v uint64) {
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
	b[4] = byte(v >> 32)
	b[5] = byte(v >> 40)
	b[6] = byte(v >> 48)
	b[7] = byte(v >> 56)
}

func minFloat(v []float64) float64 {
	m := math.Inf(1)
	for _, x := range v {
		if x < m {
			m = x
		}
	}
	return m
}

func maxFloat(v []float64) float64 {
	m := math.Inf(-1)
	for _, x := range v {
		if x > m {
			m = x
		}
	}
	return m
}
