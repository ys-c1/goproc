// Package calibration 做雾化曲线标定：在五个缩放比上实测局部方差比，
// 建立可反解的在线预判曲线。
//
// 提供两条曲线：
//   - 主模型：log-log 域分段线性曲线，过全部实测点、单调，逐段可直接反解；
//     标定区间外按端段局部幂律斜率外推。
//   - 紧凑模型：全局幂律 R(s)=A*s^gamma，gamma 是采样核支撑的单参数刻画，
//     支撑越宽（triangle 相对 box）高频损失越多，gamma 越大。
package calibration

import (
	"math"

	"goproc/internal/fog"
	"goproc/internal/resample"
)

// Point 是一个缩放比档位的实测点。
type Point struct {
	Scale     float64 // 实际缩放比 m/n
	ThumbSize int     // 缩略图边长
	R         float64 // 局部方差比
	F         float64 // 雾化程度 1-R
}

// Curve 是单个采样核标定出的雾化曲线。
type Curve struct {
	Kernel string  `json:"kernel"`
	A      float64 `json:"A"`     // 紧凑幂律系数（锚定 A=1）
	Gamma  float64 `json:"gamma"` // 紧凑幂律指数，反映核支撑
	RSq    float64 `json:"rSquared"`
	Points []Point `json:"points"`
}

// CalScales 是五个标定档：s=2^{-k/2}，k=0..4，在 log 轴上等距取点，
// 覆盖全尺寸到四分之一尺寸（1.000、0.707、0.500、0.354、0.250）。
// 取点位置在原边长 n 上对应缩略图边长 round(n*s)。
func CalScales() []float64 {
	return []float64{1, 1 / math.Sqrt2, 0.5, 1 / (2 * math.Sqrt2), 0.25}
}

// ValScales 是五个在线验证档，与标定档错开：前三档落在标定区间内做插值，
// 0.30 贴近标定下沿，0.15 落在区间外做外推检验。
func ValScales() []float64 {
	return []float64{0.85, 0.60, 0.42, 0.30, 0.15}
}

// MeasureAt 在给定缩放比下生成缩略图、放大回原尺寸并实测 R/F。
func MeasureAt(orig []float64, n int, s float64, k resample.Kind, window int) Point {
	m := int(math.Round(float64(n) * s))
	actual := float64(m) / float64(n)
	thumb := resample.Downsample(orig, n, actual, k)
	up := resample.UpscaleNearest(thumb, m, n)
	r, f := fog.Measure(orig, up, n, window, actual)
	return Point{Scale: actual, ThumbSize: m, R: r, F: f}
}

// Fit 在 log-log 域对实测点做幂律拟合。anchored 为真时令 A=1（强制过 s=1,R=1）。
// 同时返回决定系数 R²（在 log R 空间）。
func Fit(points []Point, anchored bool) (a, gamma, rSq float64) {
	type xy struct{ x, y float64 }
	pts := make([]xy, 0, len(points))
	meanX, meanY := 0.0, 0.0
	for _, p := range pts2xy(points) {
		pts = append(pts, p)
		meanX += p.x
		meanY += p.y
	}
	meanX /= float64(len(pts))
	meanY /= float64(len(pts))
	var cov, varX float64
	for _, p := range pts {
		dx := p.x - meanX
		cov += dx * (p.y - meanY)
		varX += dx * dx
	}
	gamma = cov / varX
	if anchored {
		a = 1
	} else {
		a = math.Exp(meanY - gamma*meanX)
	}
	var ssRes, ssTot float64
	for _, p := range pts {
		yhat := math.Log(a) + gamma*p.x
		ssRes += (p.y - yhat) * (p.y - yhat)
		ssTot += (p.y - meanY) * (p.y - meanY)
	}
	rSq = 1 - ssRes/ssTot
	return a, gamma, rSq
}

func pts2xy(points []Point) []struct{ x, y float64 } {
	out := make([]struct{ x, y float64 }, 0, len(points))
	for _, p := range points {
		if p.R > 0 {
			out = append(out, struct{ x, y float64 }{math.Log(p.Scale), math.Log(p.R)})
		}
	}
	return out
}

// segment 是 log-log 域相邻实测点之间的一段，logR = rLo + beta*(logS-lLo)，
// beta 即该段局部幂律指数。实测点按 s 从大到小排列，Lo/Hi 仅指端点次序。
type segment struct {
	sLo, sHi float64 // sLo > sHi：Lo 端缩放比更大
	lLo, lHi float64
	rLo, rHi float64 // rLo > rHi：Lo 端 R 更大
	beta     float64
}

func buildSegments(points []Point) []segment {
	segs := make([]segment, 0, len(points)-1)
	for i := 1; i < len(points); i++ {
		p0, p1 := points[i-1], points[i]
		l0, l1 := math.Log(p0.Scale), math.Log(p1.Scale)
		r0, r1 := math.Log(p0.R), math.Log(p1.R)
		segs = append(segs, segment{
			sLo: p0.Scale, sHi: p1.Scale,
			lLo: l0, lHi: l1, rLo: r0, rHi: r1,
			beta: (r1 - r0) / (l1 - l0),
		})
	}
	return segs
}

// PredictR 在线预判：仅输入缩放比，返回预判局部方差比，不生成缩略图。
// 标定区间内用 log-log 分段线性插值（过全部实测点、单调、可逐段反解）；
// 区间外用端段局部幂律斜率外推；s>=1 封頂为 1。
func (c Curve) PredictR(s float64) float64 {
	segs := buildSegments(c.Points)
	sMax := c.Points[0].Scale          // 1
	sMin := c.Points[len(c.Points)-1].Scale // 0.25
	switch {
	case s >= sMax:
		return 1
	case s <= sMin:
		g := segs[len(segs)-1] // 最末段，贴着最小标定档
		return math.Exp(g.rHi + g.beta*(math.Log(s)-g.lHi))
	}
	ls := math.Log(s)
	for _, g := range segs {
		if ls <= g.lLo && ls >= g.lHi {
			t := (ls - g.lLo) / (g.lHi - g.lLo)
			return math.Exp(g.rLo + t*(g.rHi-g.rLo))
		}
	}
	return 0
}

// InCalibratedRange 报告 s 是否落在标定取点区间 [0.25,1] 内。
func (c Curve) InCalibratedRange(s float64) bool {
	sMin := c.Points[len(c.Points)-1].Scale
	return s >= sMin && s <= c.Points[0].Scale
}

// PredictF 在线预判雾化程度 F=1-R。
func (c Curve) PredictF(s float64) float64 { return 1 - c.PredictR(s) }

// InvertScale 反解：给定局部方差比 R，返回对应缩放比。
// 分段线性在 log 域逐段直接反解；R 超出实测 R 区间时按端段斜率外推。
func (c Curve) InvertScale(r float64) float64 {
	segs := buildSegments(c.Points)
	rAtSMax := c.Points[0].R                  // s=1 处 R=1
	rAtSMin := c.Points[len(c.Points)-1].R    // 最小标定 s 处的 R
	switch {
	case r >= rAtSMax:
		return 1
	case r <= rAtSMin:
		g := segs[len(segs)-1]
		return math.Exp(g.lHi + (math.Log(r)-g.rHi)/g.beta)
	}
	lr := math.Log(r)
	for _, g := range segs {
		if lr <= g.rLo && lr >= g.rHi {
			t := (lr - g.rLo) / (g.rHi - g.rLo)
			return math.Exp(g.lLo + t*(g.lHi-g.lLo))
		}
	}
	return 0
}

// InvertScaleF 反解：给定雾化程度 F，返回对应缩放比。
func (c Curve) InvertScaleF(f float64) float64 {
	return c.InvertScale(1 - f)
}

// PowerLawR 是紧凑幂律模型 A*s^gamma 的预判，gamma 刻画采样核支撑。
func (c Curve) PowerLawR(s float64) float64 { return c.A * math.Pow(s, c.Gamma) }

// Calibrate 对一个采样核跑五个标定档并拟合紧凑幂律。
func Calibrate(orig []float64, n int, k resample.Kind, window int) Curve {
	pts := make([]Point, 0, 5)
	for _, s := range CalScales() {
		pts = append(pts, MeasureAt(orig, n, s, k, window))
	}
	a, g, r2 := Fit(pts, true)
	return Curve{Kernel: k.String(), A: a, Gamma: g, RSq: r2, Points: pts}
}
