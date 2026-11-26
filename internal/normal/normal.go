// Package normal 由高度图计算表面法线与法线朝向角。
//
// 坐标约定（在所有产物中保持一致）：
//   - 地形平面采用右手坐标，x 向右、y 向上（图像第 0 行对应地形 y 最大处）；
//   - 高度 h(x,y)，切平面梯度 g=(p,q)=(∂h/∂x, ∂h/∂y)；
//   - 单位法线 N=(-p,-q,1)/sqrt(1+p²+q²)；
//   - 法线水平投影 (-p,-q) 的方位角 phi=atan2(-q,-p)∈[0,2π)，
//     与梯度方位角 atan2(q,p) 恰好相差 π，故二者的分箱占比集合完全相同，
//     仅整箱错开。
package normal

import "math"

// Field 保存内部像素的梯度与单位法线；边界一圈无中心差分，不参与统计。
type Field struct {
	Size       int
	P          []float64 // ∂h/∂x，与高度图同尺寸，边界元素为 0
	Q          []float64 // ∂h/∂y（地形坐标，向上为正）
	Nx, Ny, Nz []float64 // 单位法线分量
	Interior   int       // 参与统计的内部像素数
}

// FromHeightmap 用中心差分求梯度并归一化为单位法线。
func FromHeightmap(size int, h []float64) *Field {
	if len(h) != size*size {
		panic("normal: 高度图尺寸不匹配")
	}
	f := &Field{
		Size: size,
		P:    make([]float64, size*size),
		Q:    make([]float64, size*size),
		Nx:   make([]float64, size*size),
		Ny:   make([]float64, size*size),
		Nz:   make([]float64, size*size),
	}
	for y := 1; y < size-1; y++ {
		for x := 1; x < size-1; x++ {
			i := y*size + x
			p := (h[i+1] - h[i-1]) * 0.5
			// 图像行向下增长，地形 y 向上，故对行索引取负号。
			q := -(h[i+size] - h[i-size]) * 0.5
			f.P[i] = p
			f.Q[i] = q
			inv := 1 / math.Sqrt(1+p*p+q*q)
			f.Nx[i] = -p * inv
			f.Ny[i] = -q * inv
			f.Nz[i] = inv
			f.Interior++
		}
	}
	return f
}

// Azimuth 返回像素 i 处法线水平投影的方位角，范围 [0,2π)。
func (f *Field) Azimuth(i int) float64 {
	phi := math.Atan2(f.Ny[i], f.Nx[i]) // N=(-p,-q,·)，即 atan2(-q,-p)
	if phi < 0 {
		phi += 2 * math.Pi
	}
	return phi
}

// GradientAngle 返回梯度方位角 atan2(q,p)，范围 [0,2π)，供证明中的对照使用。
func (f *Field) GradientAngle(i int) float64 {
	phi := math.Atan2(f.Q[i], f.P[i])
	if phi < 0 {
		phi += 2 * math.Pi
	}
	return phi
}

// Histogram 按 nb 个等宽角度箱统计内部像素方位角占比。
// 箱 k 覆盖 [2πk/nb, 2π(k+1)/nb)，返回每箱像素数与占比。
func (f *Field) Histogram(nb int) (counts []int, frac []float64) {
	counts = make([]int, nb)
	width := 2 * math.Pi / float64(nb)
	for y := 1; y < f.Size-1; y++ {
		for x := 1; x < f.Size-1; x++ {
			i := y*f.Size + x
			phi := f.Azimuth(i)
			k := int(phi / width)
			if k >= nb { // 数值上恰好落在 2π
				k = nb - 1
			}
			counts[k]++
		}
	}
	frac = make([]float64, nb)
	for k, c := range counts {
		frac[k] = float64(c) / float64(f.Interior)
	}
	return counts, frac
}

// Covariance 返回内部像素梯度 (p,q) 的经验协方差矩阵，
// 元素顺序为 [E[p²], E[pq]; E[pq], E[q²]]（场已去均值，梯度均值天然约为 0）。
func (f *Field) Covariance() (mxx, myy, mxy float64) {
	for y := 1; y < f.Size-1; y++ {
		for x := 1; x < f.Size-1; x++ {
			i := y*f.Size + x
			p, q := f.P[i], f.Q[i]
			mxx += p * p
			mxy += p * q
			myy += q * q
		}
	}
	n := float64(f.Interior)
	return mxx / n, myy / n, mxy / n
}
