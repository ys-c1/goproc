// Package calibrate 实现反射系数阈值的离线标定、二次曲线拟合与在线高光预判。
//
// 离线阶段在三个已知光照角下渲染并测出阈值 T，把 T 与光照角 t（度）的关系
// 拟合成二次曲线 T(t)=a*t^2+b*t+c，三个标定点即可唯一求解。
//
// 在线阶段只输入光照角：由曲线反解出该角度下的临界夹角（法线与半向量的
// 夹角上界），再逐像素比较几何夹角，不做任何渲染计算。
package calibrate

import (
	"errors"
	"math"

	"goproc/internal/water"
)

// Curve 是阈值关于光照角（度）的二次曲线 T(t)=a*t^2+b*t+c。
type Curve struct {
	A, B, C float64
}

// Threshold 返回光照角 t（度）对应的反射系数阈值。
func (c Curve) Threshold(angleDeg float64) float64 {
	return c.A*angleDeg*angleDeg + c.B*angleDeg + c.C
}

// FitQuadratic 用三个标定点解二次曲线，三点的光照角必须互异。
func FitQuadratic(p [3]water.CalibrationPoint) (Curve, error) {
	for i := 0; i < 3; i++ {
		for j := i + 1; j < 3; j++ {
			if p[i].LightElevDeg == p[j].LightElevDeg {
				return Curve{}, errors.New("标定点光照角必须互异")
			}
		}
	}
	// 范德蒙德矩阵求逆：三行分别为 (t^2,t,1)，直接用克拉默法则。
	m := func(i int) (float64, float64, float64) {
		t := p[i].LightElevDeg
		return t * t, t, 1
	}
	a0, a1, a2 := m(0)
	b0, b1, b2 := m(1)
	d0, d1, d2 := m(2)
	det := a0*(b1*d2-b2*d1) - a1*(b0*d2-b2*d0) + a2*(b0*d1-b1*d0)
	if det == 0 {
		return Curve{}, errors.New("标定点退化，无法拟合二次曲线")
	}
	y0, y1, y2 := p[0].Threshold, p[1].Threshold, p[2].Threshold
	replace := func(col int) float64 {
		r0, r1, r2 := a0, a1, a2
		s0, s1, s2 := b0, b1, b2
		u0, u1, u2 := d0, d1, d2
		if col == 0 {
			r0, s0, u0 = y0, y1, y2
		} else if col == 1 {
			r1, s1, u1 = y0, y1, y2
		} else {
			r2, s2, u2 = y0, y1, y2
		}
		return (r0*(s1*u2-s2*u1) - r1*(s0*u2-s2*u0) + r2*(s0*u1-s1*u0)) / det
	}
	return Curve{A: replace(0), B: replace(1), C: replace(2)}, nil
}

// CriticalAngle 由阈值曲线反解给定光照角下的临界夹角（弧度）。
//
// 临界夹角是法线与半向量夹角 gamma 的上界：gamma 不超过它的像素才会成为高光。
// 反射系数模型 rho=K*(F0+(1-F0)*cos(gamma)^q)*cos(gamma)^p 关于 gamma
// 单调递减，因此在 rho=threshold 处二分求根。
func CriticalAngle(c Curve, sh water.Shader, angleDeg float64) (float64, error) {
	threshold := c.Threshold(angleDeg)
	rhoAt := func(gamma float64) float64 {
		return sh.ReflectanceAt(math.Cos(gamma))
	}
	if rhoAt(0) < threshold {
		return 0, errors.New("阈值高于正入射反射系数，该光照角下预判为空高光")
	}
	lo, hi := 0.0, math.Pi/2
	if rhoAt(hi) > threshold {
		return hi, nil
	}
	for i := 0; i < 60; i++ {
		mid := (lo + hi) / 2
		if rhoAt(mid) > threshold {
			lo = mid
		} else {
			hi = mid
		}
	}
	return (lo + hi) / 2, nil
}

// Prediction 是不跑渲染的在线预判结果。
type Prediction struct {
	LightElevDeg  float64
	CriticalAngle float64 // 弧度
	Threshold      float64
	Mask           []bool
	Count          int
}

// Predict 只输入光照角（度）与几何法线场，预判高光像素数与位置。
// 全程只做角度比较，不调用任何渲染/着色函数。
func Predict(c Curve, sh water.Shader, nf *water.NormalField, angleDeg float64) (Prediction, error) {
	gammaStar, err := CriticalAngle(c, sh, angleDeg)
	if err != nil {
		return Prediction{}, err
	}
	h := sh.HalfDir(angleDeg * math.Pi / 180)
	cosCut := math.Cos(gammaStar)
	mask := make([]bool, nf.Size*nf.Size)
	count := 0
	for i, nn := range nf.N {
		if nn.Dot(h) >= cosCut {
			mask[i] = true
			count++
		}
	}
	return Prediction{
		LightElevDeg:  angleDeg,
		CriticalAngle: gammaStar,
		Threshold:      c.Threshold(angleDeg),
		Mask:           mask,
		Count:          count,
	}, nil
}

// Deviation 是一个验证光照角下预判与渲染统计的对照结果。
type Deviation struct {
	LightElevDeg       float64
	PredictedCount     int
	RenderedWhiteCount int
	OnlyPredicted      int // 预判有、渲染无
	OnlyRendered       int // 渲染有、预判无
	MaxOffsetCells     float64
	OverTwoCells       int // 位置偏差超过两格的像素数
	OverMask           []bool
	// OffsetCells 是逐像素偏差格距：两掩码一致处为零，不一致处为到对方集合
	// 最近像素的格距；对方集合为空时为正无穷。
	OffsetCells []float64
}

// Compare 对照预判掩码与渲染白像素掩码。位置偏差按到对方集合的最近格距计，
// 严格大于两格的像素在 OverMask 中标出。
func Compare(pred []bool, rendered *water.Frame, size int) Deviation {
	d := Deviation{
		PredictedCount:     countBool(pred),
		RenderedWhiteCount: rendered.WhiteCount,
		OverMask:           make([]bool, size*size),
		OffsetCells:        make([]float64, size*size),
	}
	// 分别对两个目标集合做距离变换：预判多出的像素量到渲染集合，反之亦然。
	dtRendered := distanceTransform(rendered.White, size)
	dtPredicted := distanceTransform(pred, size)
	for i := range pred {
		var off float64
		switch {
		case pred[i] && !rendered.White[i]:
			d.OnlyPredicted++
			off = dtRendered[i]
		case !pred[i] && rendered.White[i]:
			d.OnlyRendered++
			off = dtPredicted[i]
		default:
			continue
		}
		d.OffsetCells[i] = off
		if off > d.MaxOffsetCells {
			d.MaxOffsetCells = off
		}
		if off > 2.0 {
			d.OverMask[i] = true
			d.OverTwoCells++
		}
	}
	return d
}

func countBool(m []bool) int {
	n := 0
	for _, v := range m {
		if v {
			n++
		}
	}
	return n
}

// distanceTransform 用两遍 3-4 倒角距离求每个像素到最近真像素的格距，
// 真像素自身为零，没有真像素时返回正无穷。
func distanceTransform(mask []bool, size int) []float64 {
	const inf = math.MaxFloat64
	const d1, d2 = 1.0, 1.4142135623730951
	dist := make([]float64, size*size)
	for i, m := range mask {
		if m {
			dist[i] = 0
		} else {
			dist[i] = inf
		}
	}
	get := func(x, y int) float64 {
		if x < 0 || y < 0 || x >= size || y >= size {
			return inf
		}
		return dist[y*size+x]
	}
	relax := func(i, x, y int, w float64) {
		v := get(x, y)
		if v != inf && v+w < dist[i] {
			dist[i] = v + w
		}
	}
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			i := y*size + x
			if dist[i] == 0 {
				continue
			}
			relax(i, x-1, y, d1)
			relax(i, x, y-1, d1)
			relax(i, x-1, y-1, d2)
			relax(i, x+1, y-1, d2)
		}
	}
	for y := size - 1; y >= 0; y-- {
		for x := size - 1; x >= 0; x-- {
			i := y*size + x
			if dist[i] == 0 {
				continue
			}
			relax(i, x+1, y, d1)
			relax(i, x, y+1, d1)
			relax(i, x+1, y+1, d2)
			relax(i, x-1, y+1, d2)
		}
	}
	return dist
}
