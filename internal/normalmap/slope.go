package normalmap

import "math"

// SlopeTarget 是带上下平台的等角扫描斜坡高度场。
//
// 高度轮廓沿 x 方向、沿 y 方向不变：左右是梯度为零的平台，
// 中间斜坡的法线倾角 θ 从 Theta0 线性扫到 Theta1，
// 高度增量 dz/dx = tan θ 逐列积分得到。这样法线在角度域均匀扫描，
// 经过 b 位量化后跨过的码级与亮暗层数都可解析预测。
type SlopeTarget struct {
	Width      int
	Height     int
	Margin     int
	HeightData []float64
	Scale      float64
	Mask       []bool // 斜坡内部像素为 true（平台与差分边界除外）
	X0, X1     int    // 统计用斜坡内部列区间 [X0,X1)
	Theta0Deg  float64
	Theta1Deg  float64
	Light      Vec3
}

// SlopeConfig 控制标准斜坡靶。
type SlopeConfig struct {
	Rows      int     // 图像高度（行），默认 48
	Margin    int     // 两侧平台宽度，默认 48
	Length    int     // 斜坡长度（列），默认 1024
	Theta0Deg float64 // 起始倾角，默认 4
	Theta1Deg float64 // 终止倾角，默认 60
	LightDeg  float64 // 平行光相对竖直方向的倾角，默认 75（在扫描角之外，保证亮度单调）
}

// DefaultSlopeConfig 返回默认斜坡参数。
func DefaultSlopeConfig() SlopeConfig {
	return SlopeConfig{Rows: 48, Margin: 48, Length: 1024, Theta0Deg: 4, Theta1Deg: 60, LightDeg: 75}
}

// BuildSlopeTarget 构造标准斜坡靶。
func BuildSlopeTarget(cfg SlopeConfig) *SlopeTarget {
	if cfg.Rows <= 0 {
		cfg.Rows = 48
	}
	if cfg.Margin < 4 {
		cfg.Margin = 4
	}
	if cfg.Length <= 8 {
		cfg.Length = 1024
	}
	if cfg.Theta0Deg <= 0 {
		cfg.Theta0Deg = 4
	}
	if cfg.Theta1Deg <= cfg.Theta0Deg || cfg.Theta1Deg >= 89 {
		cfg.Theta1Deg = 60
	}
	if cfg.LightDeg <= cfg.Theta1Deg {
		cfg.LightDeg = cfg.Theta1Deg + 15
	}
	w := cfg.Margin*2 + cfg.Length
	h := cfg.Rows

	// 逐列高度增量。
	dz := make([]float64, w)
	for x := 0; x < w; x++ {
		switch {
		case x < cfg.Margin || x >= cfg.Margin+cfg.Length:
			dz[x] = 0 // 平台
		default:
			t := float64(x-cfg.Margin) / float64(cfg.Length-1)
			th := (cfg.Theta0Deg + (cfg.Theta1Deg-cfg.Theta0Deg)*t) * Deg2Rad
			dz[x] = math.Tan(th)
		}
	}
	colH := make([]float64, w)
	for x := 1; x < w; x++ {
		colH[x] = colH[x-1] + dz[x-1]
	}

	data := make([]float64, w*h)
	mask := make([]bool, w*h)
	// 避开中心差分在斜坡两端的混合列。
	x0 := cfg.Margin + 2
	x1 := cfg.Margin + cfg.Length - 2
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := y*w + x
			data[i] = colH[x]
			mask[i] = x >= x0 && x < x1
		}
	}
	lr := cfg.LightDeg * Deg2Rad
	// 法线 n = (-sinθ, 0, cosθ) 朝 -x 倾斜，因此光也必须朝 -x 倾斜，
	// 取 L=(-sinθL,0,cosθL) 后 n·L=cos(θL-θ)，在 θ<θL 区间严格单调，
	// 且不会被 max(0,·) 截断成大片死黑。
	light := Vec3{-math.Sin(lr), 0, math.Cos(lr)}.Normalize()
	return &SlopeTarget{
		Width: w, Height: h, Margin: cfg.Margin, HeightData: data, Scale: 1,
		Mask: mask, X0: x0, X1: x1,
		Theta0Deg: cfg.Theta0Deg, Theta1Deg: cfg.Theta1Deg,
		Light: light,
	}
}

// NormalMap 用当前高度场构造法线。
func (t *SlopeTarget) NormalMap() *NormalMap {
	return FromHeight(t.Width, t.Height, t.HeightData, t.Scale)
}

// Shade 用固定平行光做 Lambert 着色，返回逐像素亮度 [0,1]。
// ambient 是环境光底，不改变层数，只避免纯黑。
func Shade(normals []Vec3, light Vec3, ambient float64) []float64 {
	out := make([]float64, len(normals))
	for i, n := range normals {
		d := Dot(n, light)
		if d < 0 {
			d = 0
		}
		v := ambient + (1-ambient)*d
		if v > 1 {
			v = 1
		}
		out[i] = v
	}
	return out
}

// LayerCount 统计一维亮度剖面上的“亮暗层数”：
// 连续完全相同的一段算一层（一个台阶），并另外返回不同亮度值的数量。
// 对单调扫描剖面，run 数与不同值数相等。
func LayerCount(profile []float64) (runs int, levels int) {
	if len(profile) == 0 {
		return 0, 0
	}
	runs = 1
	seen := make(map[float64]struct{}, len(profile))
	seen[profile[0]] = struct{}{}
	for i := 1; i < len(profile); i++ {
		if profile[i] != profile[i-1] {
			runs++
		}
		seen[profile[i]] = struct{}{}
	}
	return runs, len(seen)
}

// LayerCount8Bit 先把亮度四舍五入到 0..255 的帧缓冲灰阶，再数亮暗层。
// 这与归档的 shaded_bN.png 像素完全一致，可由图片直接复核。
func LayerCount8Bit(profile []float64) (runs, levels int) {
	q := make([]float64, len(profile))
	for i, v := range profile {
		g := math.Round(v * 255)
		q[i] = g
	}
	return LayerCount(q)
}

// LayerWidths 返回一维剖面中每个连续等亮层的宽度（像素样本数）。
func LayerWidths(profile []float64) []int {
	if len(profile) == 0 {
		return nil
	}
	var widths []int
	w := 1
	for i := 1; i < len(profile); i++ {
		if profile[i] == profile[i-1] {
			w++
		} else {
			widths = append(widths, w)
			w = 1
		}
	}
	widths = append(widths, w)
	return widths
}

// MaxStepJump 返回相邻不同层之间最大的亮度跳变（[0,1] 量程），
// 用来刻画“台阶感”：位深越高，层数越多，但每级落差越小。
func MaxStepJump(profile []float64) float64 {
	maxJump := 0.0
	for i := 1; i < len(profile); i++ {
		if d := math.Abs(profile[i] - profile[i-1]); d > maxJump {
			maxJump = d
		}
	}
	return maxJump
}

// SlopeProfile 取中间一行在斜坡统计区间内的亮度剖面。
func (t *SlopeTarget) SlopeProfile(shaded []float64) []float64 {
	y := t.Height / 2
	out := make([]float64, 0, t.X1-t.X0)
	for x := t.X0; x < t.X1; x++ {
		out = append(out, shaded[y*t.Width+x])
	}
	return out
}

// StepReport 是单个位深的完整结果（误差与台阶数同组）。
type StepReport struct {
	Bits                int     `json:"bits"`
	MaxCode             int     `json:"max_code"`
	Pixels              int     `json:"slope_pixels"`
	MaxAngleErrorDeg    float64 `json:"max_angle_error_deg"`
	MeanAngleErrorDeg   float64 `json:"mean_angle_error_deg"`
	RMSAngleErrorDeg    float64 `json:"rms_angle_error_deg"`
	AngleStepDegAtGMax  float64 `json:"angle_quant_step_deg_at_max_gradient"`
	LocalBoundAtGMaxDeg float64 `json:"first_order_error_estimate_deg_at_max_gradient"`
	BoundDeg            float64 `json:"gradient_error_bound_deg_at_max_gradient"`
	RigorousBoundDeg    float64 `json:"rigorous_bound_deg"`
	BoundViolations     int     `json:"bound_violations"`
	WithinBound         bool    `json:"within_bound"`
	Steps               int     `json:"steps_bright_dark_layers"`
	Levels              int     `json:"distinct_shade_levels"`
	Steps8Bit           int     `json:"steps_on_archived_8bit_shading"`
	Levels8Bit          int     `json:"levels_on_archived_8bit_shading"`
	MaxShadeJumpGray    float64 `json:"max_shade_jump_gray_levels"`
	MaxLayerWidthPx     int     `json:"max_layer_width_px"`
	MeanLayerWidthPx    float64 `json:"mean_layer_width_px"`
	ProfileSamples      int     `json:"profile_samples"`
}

// EvaluateBits 在标准斜坡靶上跑完“量化→误差→着色→数台阶”。
func EvaluateBits(t *SlopeTarget, bits int) StepReport {
	nm := t.NormalMap()
	q := Quantize(nm, bits)
	st := MeasureErrors(nm, q, t.Mask)
	shaded := Shade(q.Normals, t.Light, 0)
	prof := t.SlopeProfile(shaded)
	runs, levels := LayerCount(prof)
	runs8, levels8 := LayerCount8Bit(prof)
	widths := LayerWidths(prof)
	maxW := 0
	sumW := 0
	for _, w := range widths {
		if w > maxW {
			maxW = w
		}
		sumW += w
	}
	meanW := 0.0
	if len(widths) > 0 {
		meanW = float64(sumW) / float64(len(widths))
	}

	gmax := 0.0
	for i, ok := range t.Mask {
		if ok && nm.Gradient[i] > gmax {
			gmax = nm.Gradient[i]
		}
	}
	return StepReport{
		Bits:                bits,
		MaxCode:             q.MaxCode,
		Pixels:              st.Pixels,
		MaxAngleErrorDeg:    st.MaxDeg,
		MeanAngleErrorDeg:   st.MeanDeg,
		RMSAngleErrorDeg:    st.RMSDeg,
		AngleStepDegAtGMax:  AngularStepDeg(bits, gmax),
		LocalBoundAtGMaxDeg: LocalErrorBoundDeg(bits, gmax),
		BoundDeg:            BoundDeg(bits, gmax),
		RigorousBoundDeg:    RigorousBoundDeg(bits),
		BoundViolations:     st.Violations,
		WithinBound:         st.Violations == 0,
		Steps:               runs,
		Levels:              levels,
		Steps8Bit:           runs8,
		Levels8Bit:          levels8,
		MaxShadeJumpGray:    MaxStepJump(prof) * 255,
		MaxLayerWidthPx:     maxW,
		MeanLayerWidthPx:    meanW,
		ProfileSamples:      len(prof),
	}
}
