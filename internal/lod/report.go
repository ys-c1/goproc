package lod

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"
)

// TerrainInfo 记录本次实验使用的确定性地形参数。
type TerrainInfo struct {
	Size       int     `json:"size"`
	Seed       uint64  `json:"seed"`
	Octaves    int     `json:"octaves"`
	Frequency  float64 `json:"frequency"`
	Lacunarity float64 `json:"lacunarity"`
	Gain       float64 `json:"gain"`
	ReliefM    float64 `json:"reliefM"` // 归一化高程 1.0 对应的实地高差（米），仅用于误差换算
}

// LevelRow 是单个 LOD 层级（二/三/四级）的一组缝宽与高程误差。
type LevelRow struct {
	Level        int `json:"level"`
	CoarseStride int `json:"coarseStridePx"` // d_k=2^k

	// 主场景：近处全分辨率（步长 1）与远处 LOD-k（步长 2^k）在 y=SeamCoord 相接。
	FineStride       int     `json:"fineStridePx"`
	DeltaStride      int     `json:"deltaStridePx"`
	SeamCoord        int     `json:"seamCoordPx"`
	CoordDivisible   bool    `json:"coordinateDivisible"`     // SeamCoord % CoarseStride == 0?
	RemainderPx      int     `json:"remainderPx"`             // SeamCoord mod CoarseStride
	CrackBeforePx    float64 `json:"crackWidthBeforePx"`      // Hausdorff 实测，修复前
	CrackAfterPx     float64 `json:"crackWidthAfterPx"`       // 共享边界顶点 + zipper，修复后
	CoefficientC     float64 `json:"coefficientObserved"`     // CrackBefore / DeltaStride
	HalfCellBeforePx float64 `json:"halfCellCrackBeforePx"`   // rho=1/2 登记约定：Delta/2
	HalfCellMeasPx   float64 `json:"halfCellCrackMeasuredPx"` // Hausdorff 实测核对

	// “相邻两级”链：(k-1, k)，步长 (2^{k-1}, 2^k)。
	AdjFineLevel       int     `json:"adjacentFineLevel"`
	AdjFineStride      int     `json:"adjacentFineStridePx"`
	AdjDeltaStride     int     `json:"adjacentDeltaStridePx"`
	AdjCrackBeforePx   float64 `json:"adjacentCrackBeforePx"`
	AdjCrackAfterPx    float64 `json:"adjacentCrackAfterPx"`
	AdjCoefficientC    float64 `json:"adjacentCoefficientObserved"`
	AdjHalfBeforePx    float64 `json:"adjacentHalfCellBeforePx"`
	AdjDivisibleByFine bool    `json:"coarseStrideDivisibleByFine"`

	// 最大高程误差（LOD-k 对齐粗网格双线性重建 vs 全分辨率真值）。
	MaxElevErr      float64 `json:"maxElevationError"`
	MaxElevErrGray  float64 `json:"maxElevationErrorGrayLevels"`
	MaxElevErrM     float64 `json:"maxElevationErrorMeters"`
	MaxErrAtX       int     `json:"maxErrAtX"`
	MaxErrAtY       int     `json:"maxErrAtY"`
	SeamBandHalfPx  int     `json:"seamBandHalfPx"`
	SeamBandElevErr float64 `json:"seamBandMaxElevationError"`
	SeamBandErrGray float64 `json:"seamBandMaxErrorGrayLevels"`

	StitchedImage string `json:"stitchedImage"`
	CompareImage  string `json:"compareImage"`
}

// LawPoint 是比例定律 w=rho|H-h| 的一个独立实测核对点。
type LawPoint struct {
	Fine      int     `json:"fineStridePx"`
	H         int     `json:"coarseStridePx"`
	Rho       float64 `json:"rho"`
	Predicted float64 `json:"predictedPx"`
	Measured  float64 `json:"measuredPx"`
	AbsError  float64 `json:"absErrorPx"`
}

// Report 是完整结果，序列化为 metrics.json。
type Report struct {
	Schema   string      `json:"schema"`
	Terrain  TerrainInfo `json:"terrain"`
	Theorem  Theorem     `json:"theorem"`
	Levels   []LevelRow  `json:"levels"`
	LawSweep []LawPoint  `json:"lawSweep"`
	AllZero  bool        `json:"allCracksZeroAfterFix"`
	MaxCrack float64     `json:"maxCrackAfterFixPx"`
}

// Theorem 记录系数结论，方便程序化读取。
type Theorem struct {
	Divisibility string             `json:"divisibilityCondition"`
	GeneralLaw   string             `json:"generalLaw"`
	Coefficient  map[string]float64 `json:"coefficientByConvention"`
	Note         string             `json:"note"`
}

// BuildReport 在给定确定性高度图上计算 LOD 二/三/四级的全部指标。
// height 为 n×n、值域 [0,1] 的参考高程。
func BuildReport(height []float64, n int, ti TerrainInfo) *Report {
	const seamY = 63 // =64-1，64 被 4/8/16 整除：Y mod d = d-1
	x0, x1 := 0, n-1
	r := &Report{
		Schema:  "lod-seam-metrics/v1",
		Terrain: ti,
		Theorem: Theorem{
			Divisibility: "粗级每个边界顶点 (x,y) 必须满足 x%h==0 且 y%h==0；公共原点下等价于 H%h==0，且接缝坐标 X%H==0（或显式补边界顶点）。",
			GeneralLaw:   "w = rho*|H-h|，rho 为边界内缩占格距比例，即比例系数 C=rho。",
			Coefficient: map[string]float64{
				"corner_aligned_rho0": 0,
				"half_cell_rho1/2":    0.5,
				"full_stride_rho1":    1,
			},
			Note: "整数截断/整格错位的实测主序列 rho=1，系数 1；像素中心(半格)登记 rho=1/2，系数 1/2；修复成角点对齐+共享边界顶点后 rho=0，缝宽严格为 0。",
		},
	}

	for _, k := range []int{2, 3, 4} {
		H := Spacing(k)
		h := 1
		delta := H - h
		rem := Remainder(seamY, H)

		fb, cb := NaiveBoundaries(x0, x1, seamY, h, H)
		before := Hausdorff(fb, cb)
		ff, cf := StitchedBoundaries(x0, x1, seamY, h, H)
		after := Hausdorff(ff, cf)

		hi, ci := InsetBoundaries(x0, x1, seamY, h, H, 0.5)
		halfMeas := Hausdorff(hi, ci)

		// 相邻链 (k-1, k)
		ah := Spacing(k - 1)
		afb, acb := NaiveBoundaries(x0, x1, seamY, ah, H)
		adjBefore := Hausdorff(afb, acb)
		aff, acf := StitchedBoundaries(x0, x1, seamY, ah, H)
		adjAfter := Hausdorff(aff, acf)
		ahf, ahc := InsetBoundaries(x0, x1, seamY, ah, H, 0.5)
		adjHalf := Hausdorff(ahf, ahc)
		adjDelta := H - ah

		maxE, ex, ey := MaxError(height, n, H)
		const bandHalf = 8
		bandE := MaxErrorBand(height, n, H, seamY, bandHalf)

		row := LevelRow{
			Level: k, CoarseStride: H,
			FineStride: h, DeltaStride: delta, SeamCoord: seamY,
			CoordDivisible: Divisible(seamY, H), RemainderPx: rem,
			CrackBeforePx: before, CrackAfterPx: after,
			CoefficientC:     before / float64(delta),
			HalfCellBeforePx: 0.5 * float64(delta), HalfCellMeasPx: halfMeas,

			AdjFineLevel: k - 1, AdjFineStride: ah, AdjDeltaStride: adjDelta,
			AdjCrackBeforePx: adjBefore, AdjCrackAfterPx: adjAfter,
			AdjCoefficientC:    adjBefore / float64(adjDelta),
			AdjHalfBeforePx:    adjHalf,
			AdjDivisibleByFine: CoarseSubsetOfFine(ah, H),

			MaxElevErr: maxE, MaxElevErrGray: GrayLevels(maxE),
			MaxElevErrM: maxE * ti.ReliefM, MaxErrAtX: ex, MaxErrAtY: ey,
			SeamBandHalfPx: bandHalf, SeamBandElevErr: bandE, SeamBandErrGray: GrayLevels(bandE),

			StitchedImage: fmt.Sprintf("artifacts/lod/lod%d_stitched.png", k),
			CompareImage:  fmt.Sprintf("artifacts/lod/lod%d_compare.png", k),
		}
		r.Levels = append(r.Levels, row)
		if after > r.MaxCrack {
			r.MaxCrack = after
		}
	}
	r.AllZero = r.MaxCrack == 0

	// 比例定律独立扫描：任意步长组合 × 三种 rho，用 Hausdorff 实测核对。
	hs := []int{1, 2, 3, 4}
	Hs := []int{2, 3, 4, 5, 6, 8, 16}
	rhos := []float64{0.5, 1.0}
	for _, h := range hs {
		for _, H := range Hs {
			if H <= h {
				continue
			}
			for _, rho := range rhos {
				a, b := InsetBoundaries(0, 128, 127, h, H, rho)
				meas := Hausdorff(a, b)
				pred := InsetGap(h, H, rho)
				r.LawSweep = append(r.LawSweep, LawPoint{
					Fine: h, H: H, Rho: rho,
					Predicted: pred, Measured: meas, AbsError: math.Abs(meas - pred),
				})
			}
		}
	}
	return r
}

// WriteJSON 把完整指标写到 metrics.json。
func (r *Report) WriteJSON(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0o644)
}

// TableMarkdown 返回“三个层级一组”的缝宽与误差汇总表。
func (r *Report) TableMarkdown() string {
	var b strings.Builder
	fmt.Fprintln(&b, "# LOD 二级/三级/四级：缝宽与最大高程误差（一组）")
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "地形：%d×%d，seed=%d，octaves=%d，freq=%.3f；缝宽单位为源高度图像素；修复=整除对齐 + 共享边界顶点(zipper)；米误差按全高差 %.0f m 换算。\n\n",
		r.Terrain.Size, r.Terrain.Size, r.Terrain.Seed, r.Terrain.Octaves, r.Terrain.Frequency, r.Terrain.ReliefM)
	b.WriteString("| LOD 级 k | 格距 d_k(px) | 缝坐标 Y | Y 整除 d_k | 修复前缝宽(px) | C=w/Δd | 半格约定(px) | 相邻级全步(px) | 相邻级半格(px) | 修复后缝宽(px) | 最大高程误差(归一化) | 灰度级 | 米误差 | 缝带最大误差 |\n")
	fmt.Fprintln(&b, "|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
	for _, L := range r.Levels {
		fmt.Fprintf(&b,
			"| %d | %d | %d | 否(余 %d) | %.3f | %.3f | %.3f | %.3f (Δ=%d) | %.3f | **%.3f** | %.6f | %.3f | %.3f | %.6f |\n",
			L.Level, L.CoarseStride, L.SeamCoord, L.RemainderPx,
			L.CrackBeforePx, L.CoefficientC, L.HalfCellMeasPx,
			L.AdjCrackBeforePx, L.AdjDeltaStride, L.AdjHalfBeforePx,
			L.CrackAfterPx, L.MaxElevErr, L.MaxElevErrGray, L.MaxElevErrM, L.SeamBandElevErr)
	}
	fmt.Fprintln(&b)
	fmt.Fprintf(&b, "修复后三级最大缝宽 = %.3e px，全部为零：**%v**。\n\n", r.MaxCrack, r.AllZero)
	fmt.Fprintln(&b, "## 比例系数核对（Hausdorff 实测 vs w=ρ|H−h|，节选）")
	fmt.Fprintln(&b)
	fmt.Fprintln(&b, "| h | H | ρ | 预测 px | 实测 px | 绝对误差 |")
	fmt.Fprintln(&b, "|---:|---:|---:|---:|---:|---:|")
	for _, p := range r.LawSweep {
		if p.Rho == 1.0 || (p.Fine == 1 && (p.H == 4 || p.H == 8 || p.H == 16)) {
			fmt.Fprintf(&b, "| %d | %d | %.2f | %.4f | %.4f | %.2e |\n", p.Fine, p.H, p.Rho, p.Predicted, p.Measured, p.AbsError)
		}
	}
	return b.String()
}

// WriteTable 写 table.md。
func (r *Report) WriteTable(path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(r.TableMarkdown()), 0o644)
}
