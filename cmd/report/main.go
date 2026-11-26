// conflict 分析：两条判据并排、合取计数、空间分带、可辨识性上界、代理标注者与标注图。
//
// 判据定义（与 internal/render 一致）：
//
//	A  坡向朝光：N·L ≥ 0，代数等价于 u ≤ tan(elev)，u 为沿光照方位的方向导数
//	B  深度差在阈值内：0 < gap ≤ τ
//	C = A ∧ B
//
// 位移单位说明：高度图已归一化到 [0,1]，而 x/y 以格点计，故 u 的量纲是"高度/格"。
// 判据比较的是 u 与 tan(elev)，两者必须处在同一量纲；若把归一化高度当作真实高度、
// 格距当作 1，则 tan(elev) 与 u 可比，但地形最大坡度仅约 3.3°，山体阴影几乎无对比。
// 这里不引入垂直比例因子，保持与仓库现有渲染一致（仓库中并无渲染器，本包即定义）。
package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"sort"

	"goproc/api"
	"goproc/internal/render"
)

const tau = 0.02

type CellRow struct {
	X, Y   int
	U      float64
	TanE   float64
	NL     float64
	Gap    float64
	Shade  float64
	Shadow float64
	Comp   float64
}

type Rater struct {
	Name   string  `json:"name"`
	Method string  `json:"method"`
	Pixels int     `json:"pixels"`
	IoU    float64 `json:"iou_vs_criterion"`
	Prec   float64 `json:"precision"`
	Recall float64 `json:"recall"`
}

type AngleReport struct {
	AzDeg         float64   `json:"az_deg"`
	ElevDeg       float64   `json:"elev_deg"`
	Tau           float64   `json:"tau"`
	CountA        int       `json:"A_lit_count"`
	CountB        int       `json:"B_marginal_count"`
	CountC        int       `json:"C_conflict_count"`
	CountOpp      int       `json:"opposite_lee_unoccluded_count"`
	TotalPixels   int       `json:"total_pixels"`
	MaxUDeg       float64   `json:"max_abs_u_deg"`
	ShadeMean     float64   `json:"shade_mean"`
	ShadeStd      float64   `json:"shade_std"`
	ShadeRange    float64   `json:"shade_range"`
	ShadowCover   float64   `json:"shadow_cover_frac"`
	ConflictCells []CellRow `json:"conflict_cells"`
	RowBands      []int     `json:"conflict_row_bands_16px"`
	ColBands      []int     `json:"conflict_col_bands_16px"`
	ConflictBBox  [4]int    `json:"conflict_bbox_minx_miny_maxx_maxy"`
	SalientUnion  int       `json:"salient_union_px"`
	SalientInter  int       `json:"salient_inter_px"`
	UpperBoundIoU float64   `json:"salient_upper_bound_iou"`
	Raters        []Rater   `json:"proxy_raters"`
}

func main() {
	hm, err := api.Generate(api.Options{Size: 128, Octaves: 4, Lacunarity: 2, Gain: 0.5, Seed: 7, Frequency: 0.02})
	if err != nil {
		panic(err)
	}
	h := &render.Heightmap{Size: hm.Size, Data: hm.Data}
	out := "artifacts"
	os.MkdirAll(out, 0o755)

	reports := map[string]AngleReport{}
	masks := map[string][]bool{}
	for _, deg := range []float64{45, 135} {
		p := render.DefaultParams()
		p.Azimuth = deg * math.Pi / 180
		p.Elev = 2 * math.Pi / 180
		rep, m := analyze(h, p, out)
		key := fmt.Sprintf("%.0f", deg)
		reports[key] = rep
		masks[key] = m
	}
	cross := jaccard(masks["45"], masks["135"])

	blob, _ := json.MarshalIndent(map[string]any{
		"criterion":                  "A: N·L>=0 (u<=tan e); B: 0<gap<=tau; C=A∧B",
		"elevation_deg":              2,
		"tau":                        tau,
		"jaccard_conflict_45_vs_135": cross,
		"report":                     reports,
	}, "", "  ")
	os.WriteFile(filepath.Join(out, "conflict_report.json"), blob, 0o644)
	fmt.Printf("cross-angle Jaccard (C45 vs C135) = %.4f\n", cross)
}

func analyze(h *render.Heightmap, p render.Params, out string) (AngleReport, []bool) {
	n := h.Size * h.Size
	tanE := math.Tan(p.Elev)
	gap := h.DepthGap(p)
	u := h.DirectionalSlope(p.Azimuth)
	shade := render.Shading(h, p)
	shadow := render.Shadow(h, p, 0.05)
	comp := make([]float64, n)
	for i := range comp {
		comp[i] = shade[i] * shadow[i]
	}
	tag := fmt.Sprintf("az%.0f", p.Azimuth*180/math.Pi)

	rep := AngleReport{AzDeg: p.Azimuth * 180 / math.Pi, ElevDeg: p.Elev * 180 / math.Pi, Tau: tau, TotalPixels: n}
	isA := make([]bool, n)
	isB := make([]bool, n)
	isC := make([]bool, n)
	isOpp := make([]bool, n)
	dot := make([]float64, n)
	for i := 0; i < n; i++ {
		x, y := i%h.Size, i/h.Size
		dx := (h.AtClamped(x+1, y) - h.AtClamped(x-1, y)) * 0.5
		dy := (h.AtClamped(x, y+1) - h.AtClamped(x, y-1)) * 0.5
		dot[i] = (math.Sin(p.Elev) - u[i]*math.Cos(p.Elev)) / math.Sqrt(1+dx*dx+dy*dy)
		isA[i] = u[i] <= tanE
		isB[i] = gap[i] > 0 && gap[i] <= tau
		isC[i] = isA[i] && isB[i]
		isOpp[i] = u[i] > tanE && gap[i] == 0
		if isA[i] {
			rep.CountA++
		}
		if isB[i] {
			rep.CountB++
		}
		if isOpp[i] {
			rep.CountOpp++
		}
	}
	rep.RowBands = make([]int, 8)
	rep.ColBands = make([]int, 8)
	first := true
	for i := 0; i < n; i++ {
		if !isC[i] {
			continue
		}
		x, y := i%h.Size, i/h.Size
		rep.CountC++
		rep.ConflictCells = append(rep.ConflictCells, CellRow{
			X: x, Y: y, U: u[i], TanE: tanE, NL: dot[i], Gap: gap[i],
			Shade: shade[i], Shadow: shadow[i], Comp: comp[i],
		})
		rep.RowBands[y/16]++
		rep.ColBands[x/16]++
		if first {
			rep.ConflictBBox = [4]int{x, y, x, y}
			first = false
		}
		bb := rep.ConflictBBox
		rep.ConflictBBox = [4]int{min(bb[0], x), min(bb[1], y), max(bb[2], x), max(bb[3], y)}
	}
	sort.Slice(rep.ConflictCells, func(i, j int) bool { return rep.ConflictCells[i].Gap > rep.ConflictCells[j].Gap })

	rep.MaxUDeg = math.Atan(maxAbs(u)) * 180 / math.Pi
	rep.ShadeMean, rep.ShadeStd = meanStd(shade)
	rep.ShadeRange = rangeOf(shade)
	cover := 0
	for _, s := range shadow {
		if s < 0.5 {
			cover++
		}
	}
	rep.ShadowCover = float64(cover) / float64(n)

	salient := thresholdSalient(comp, h.Size, 2)
	for i := 0; i < n; i++ {
		if isC[i] && salient[i] {
			rep.SalientInter++
		}
		if isC[i] || salient[i] {
			rep.SalientUnion++
		}
	}
	if rep.SalientUnion > 0 {
		rep.UpperBoundIoU = float64(rep.SalientInter) / float64(rep.SalientUnion)
	}

	rep.Raters = []Rater{
		proxyRater("R1-代理(非人类)", "局部对比>2灰度级", salient, isC),
		proxyRater("R2-代理(非人类)", "局部对比>4灰度级", thresholdSalient(comp, h.Size, 4), isC),
		proxyRater("R3-代理(非人类)", "阴影<0.9 且 着色>均值", darkInShadow(shadow, shade, rep.ShadeMean), isC),
	}

	writeCSV(filepath.Join(out, "conflict_cells_"+tag+".csv"), rep.ConflictCells)
	writeGray(filepath.Join(out, "shading_"+tag+".png"), h.Size, shade)
	writeGray(filepath.Join(out, "shadow_"+tag+".png"), h.Size, shadow)
	writeGray(filepath.Join(out, "gap_"+tag+".png"), h.Size, normalize(gap))
	writeAnnotated(filepath.Join(out, "conflict_"+tag+".png"), h, isC, shade, shadow, 1)
	writeAnnotated(filepath.Join(out, "conflict_zoom_"+tag+".png"), h, isC, shade, shadow, 4)

	fmt.Printf("[az%.0f] A=%d B=%d C=A∧B=%d (%.4f%%) 反向=%d | 着色 std=%.4f range=%.4f | 阴影覆盖=%.3f%% | 上界IoU=%.3f\n",
		rep.AzDeg, rep.CountA, rep.CountB, rep.CountC, 100*float64(rep.CountC)/float64(n), rep.CountOpp,
		rep.ShadeStd, rep.ShadeRange, 100*rep.ShadowCover, rep.UpperBoundIoU)
	for _, r := range rep.Raters {
		fmt.Printf("    %s  %d px  IoU=%.4f  P=%.3f R=%.3f\n", r.Name, r.Pixels, r.IoU, r.Prec, r.Recall)
	}
	return rep, isC
}

func proxyRater(name, method string, mask, crit []bool) Rater {
	inter, uni, np, cp := 0, 0, 0, 0
	for i := range mask {
		if mask[i] {
			np++
		}
		if crit[i] {
			cp++
		}
		if mask[i] && crit[i] {
			inter++
		}
		if mask[i] || crit[i] {
			uni++
		}
	}
	r := Rater{Name: name, Method: method, Pixels: np}
	if uni > 0 {
		r.IoU = float64(inter) / float64(uni)
	}
	if np > 0 {
		r.Prec = float64(inter) / float64(np)
	}
	if cp > 0 {
		r.Recall = float64(inter) / float64(cp)
	}
	return r
}

func thresholdSalient(v []float64, size int, k float64) []bool {
	out := make([]bool, len(v))
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			i := y*size + x
			s, c := 0.0, 0
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					xx, yy := x+dx, y+dy
					if xx < 0 || yy < 0 || xx >= size || yy >= size {
						continue
					}
					s += v[yy*size+xx]
					c++
				}
			}
			out[i] = math.Abs(v[i]-s/float64(c))*255 > k
		}
	}
	return out
}

func darkInShadow(shadow, shade []float64, mean float64) []bool {
	out := make([]bool, len(shadow))
	for i := range shadow {
		out[i] = shadow[i] < 0.9 && shade[i] > mean
	}
	return out
}

func writeAnnotated(name string, h *render.Heightmap, conf []bool, shade, shadow []float64, k int) {
	size := h.Size
	img := image.NewRGBA(image.Rect(0, 0, size*k, size*k))
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			i := y*size + x
			v := shade[i] * shadow[i]
			g := uint8(math.Max(0, math.Min(1, v)) * 255)
			c := color.RGBA{g, g, g, 255}
			if conf[i] {
				c = color.RGBA{255, 0, 0, 255}
			}
			for dy := 0; dy < k; dy++ {
				for dx := 0; dx < k; dx++ {
					img.SetRGBA(x*k+dx, y*k+dy, c)
				}
			}
		}
	}
	f, err := os.Create(name)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	png.Encode(f, img)
}

func writeCSV(name string, rows []CellRow) {
	f, err := os.Create(name)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	w := csv.NewWriter(f)
	defer w.Flush()
	w.Write([]string{"x", "y", "u", "tan_elev", "NdotL", "gap", "shade", "shadow", "composite"})
	for _, r := range rows {
		w.Write([]string{
			fmt.Sprint(r.X), fmt.Sprint(r.Y),
			fmt.Sprintf("%.7f", r.U), fmt.Sprintf("%.7f", r.TanE),
			fmt.Sprintf("%.7f", r.NL), fmt.Sprintf("%.7f", r.Gap),
			fmt.Sprintf("%.5f", r.Shade), fmt.Sprintf("%.5f", r.Shadow), fmt.Sprintf("%.5f", r.Comp),
		})
	}
}

func writeGray(name string, size int, v []float64) {
	img := image.NewGray(image.Rect(0, 0, size, size))
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			g := int(v[y*size+x]*255 + 0.5)
			if g < 0 {
				g = 0
			}
			if g > 255 {
				g = 255
			}
			img.SetGray(x, y, color.Gray{uint8(g)})
		}
	}
	f, _ := os.Create(name)
	defer f.Close()
	png.Encode(f, img)
}

func jaccard(a, b []bool) float64 {
	inter, uni := 0, 0
	for i := range a {
		if a[i] && b[i] {
			inter++
		}
		if a[i] || b[i] {
			uni++
		}
	}
	if uni == 0 {
		return 0
	}
	return float64(inter) / float64(uni)
}

func maxAbs(v []float64) float64 {
	m := 0.0
	for _, x := range v {
		m = math.Max(m, math.Abs(x))
	}
	return m
}

func meanStd(v []float64) (float64, float64) {
	s := 0.0
	for _, x := range v {
		s += x
	}
	m := s / float64(len(v))
	q := 0.0
	for _, x := range v {
		q += (x - m) * (x - m)
	}
	return m, math.Sqrt(q / float64(len(v)))
}

func rangeOf(v []float64) float64 {
	lo, hi := v[0], v[0]
	for _, x := range v {
		lo = math.Min(lo, x)
		hi = math.Max(hi, x)
	}
	return hi - lo
}

func normalize(v []float64) []float64 {
	lo, hi := v[0], v[0]
	for _, x := range v {
		lo = math.Min(lo, x)
		hi = math.Max(hi, x)
	}
	if hi == lo {
		return v
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = (x - lo) / (hi - lo)
	}
	return out
}
