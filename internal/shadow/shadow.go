// Package shadow 实现地形阴影的两种算法并互相校验：
//
//   - March：逐格步进，从每格出发沿光线方向以一格为步长采样（双线性插值），
//     慢，作基准真值；
//   - Sweep：包络一次扫掠。把高度场重采样到光线对齐网格后，
//     命题 2 的递推化为逐行一维累积最大，沿光线反方向扫一遍即得全图阴影。
//
// 命题 1 保证阴影判定只依赖包络：p 阴影 ⟺ h(p) ≤ E(p)。
//
// 注意：包络链不能对位置取整（round 链会相对真实射线线性漂移，实测误差率
// 4~5%），也不能在原地网格上边递推边插值（插值稀释沿链累积，实测可达 10%）；
// 必须先把高度场重采样到光线对齐网格，使递推成为精确的一维累积最大，
// 这样两法共享同一射线定义与同一插值高度场，比对才有意义。
package shadow

import (
	"math"
)

// Light 方向光。Azimuth 为指向光源的方位角（度，0 朝 +x，90 朝 +y，图像坐标 y 向下），
// Elevation 为高度角（度，0 贴地，90 正顶）。
type Light struct {
	Azimuth   float64
	Elevation float64
}

// dir 返回指向光源的水平单位向量。
func (l Light) dir() (dx, dy float64) {
	r := l.Azimuth * math.Pi / 180
	return math.Cos(r), math.Sin(r)
}

func (l Light) tan() float64 { return math.Tan(l.Elevation * math.Pi / 180) }

// bilinear 在浮点位置 (fx,fy) 处双线性插值采样。任一角出界则 ok=false。
func bilinear(v []float64, size int, fx, fy float64) (val float64, ok bool) {
	if fx < 0 || fy < 0 || fx > float64(size-1) || fy > float64(size-1) {
		return 0, false
	}
	x0, y0 := int(fx), int(fy)
	x1, y1 := x0+1, y0+1
	if x1 >= size {
		x1 = size - 1
	}
	if y1 >= size {
		y1 = size - 1
	}
	tx, ty := fx-float64(x0), fy-float64(y0)
	v00 := v[y0*size+x0]
	v10 := v[y0*size+x1]
	v01 := v[y1*size+x0]
	v11 := v[y1*size+x1]
	top := v00 + (v10-v00)*tx
	bot := v01 + (v11-v01)*tx
	return top + (bot-top)*ty, true
}

// March 逐格步进法：步长一格，在精确点 p+k·u 处双线性采样，
// 任一采样高度不低于 h(p)+k·tanα 则 p 判阴影。返回 size*size 的布尔图。
func March(h []float64, size int, l Light) []bool {
	dx, dy := l.dir()
	tan := l.tan()
	out := make([]bool, size*size)
	maxK := 2 * size // 对角线长度上界，出界即停
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			hp := h[y*size+x]
			for k := 1; k <= maxK; k++ {
				hv, ok := bilinear(h, size, float64(x)+float64(k)*dx, float64(y)+float64(k)*dy)
				if !ok {
					break
				}
				if hv >= hp+float64(k)*tan {
					out[y*size+x] = true
					break
				}
			}
		}
	}
	return out
}

// negBig 代替负无穷参与插值，避免 ±Inf 污染加权平均；远小于任何可达高度即可。
const negBig = -1e18

// Envelope 一次扫掠求每格包络 E(p)（命题 2 的递推式，步长 Δ=1 格）。
//
// 把高度场重采样到光线对齐网格 (s', t')：t' 沿光线方向、s' 垂直光线，
// 间距均为一格。在该网格上命题 2 的递推是逐行精确的一维累积最大：
//
//	E'(s', t') = max( H'(s', t'+1), E'(s', t'+1) ) − tanα
//
// 沿 t' 降序扫一遍即得，无需排序、无取整漂移、误差不沿链累积；
// 最后在 (s,t) 空间双线性插值回原格点。重采样与回插各引入一次
// 有界插值误差，只影响阴影边界带。
func Envelope(h []float64, size int, l Light) []float64 {
	dx, dy := l.dir()
	tan := l.tan()
	px, py := -dy, dx // 垂直于光线的单位向量 u⊥

	// 原格点在 (s,t) 坐标系的覆盖范围，外扩一格供插值。
	smin, smax := math.Inf(1), math.Inf(-1)
	tmin, tmax := smin, smax
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			s := float64(x)*px + float64(y)*py
			t := float64(x)*dx + float64(y)*dy
			smin, smax = math.Min(smin, s), math.Max(smax, s)
			tmin, tmax = math.Min(tmin, t), math.Max(tmax, t)
		}
	}
	s0 := int(math.Floor(smin)) - 1
	t0 := int(math.Floor(tmin)) - 1
	ns := int(math.Ceil(smax)) + 1 - s0 + 1
	nt := int(math.Ceil(tmax)) + 1 - t0 + 1

	// 重采样高度场到光线对齐网格；网格外无地形，高度记 negBig（不遮挡）。
	hp := make([]float64, ns*nt)
	for j := 0; j < ns; j++ {
		for i := 0; i < nt; i++ {
			s, t := float64(s0+j), float64(t0+i)
			v, ok := bilinear(h, size, t*dx+s*px, t*dy+s*py)
			if !ok {
				v = negBig
			}
			hp[j*nt+i] = v
		}
	}

	// 沿光线反方向（t' 降序）逐行累积最大：E' 即沿方向的累积最大高度。
	ep := make([]float64, ns*nt)
	for j := 0; j < ns; j++ {
		ep[j*nt+nt-1] = negBig
		for i := nt - 2; i >= 0; i-- {
			e := hp[j*nt+i+1]
			if ep[j*nt+i+1] > e {
				e = ep[j*nt+i+1]
			}
			ep[j*nt+i] = e - tan
		}
	}

	// 插值回原格点。negBig 不是高度而是"光线已出界、无遮挡信息"，
	// 不能参与加权平均（否则地图边缘一格宽的遮挡条会被抹掉）：
	// 改为只在有限角上按权重归一化插值；四角皆无约束则为 negBig。
	env := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			s := float64(x)*px + float64(y)*py - float64(s0)
			t := float64(x)*dx + float64(y)*dy - float64(t0)
			i0, j0 := int(t), int(s)
			ft, fs := t-float64(i0), s-float64(j0)
			c := [4]float64{
				ep[j0*nt+i0], ep[j0*nt+i0+1],
				ep[(j0+1)*nt+i0], ep[(j0+1)*nt+i0+1],
			}
			w := [4]float64{(1 - ft) * (1 - fs), ft * (1 - fs), (1 - ft) * fs, ft * fs}
			sum, wsum := 0.0, 0.0
			for k := 0; k < 4; k++ {
				if c[k] <= negBig/2 {
					continue
				}
				sum += w[k] * c[k]
				wsum += w[k]
			}
			if wsum > 0 {
				env[y*size+x] = sum / wsum
			} else {
				env[y*size+x] = negBig
			}
		}
	}
	return env
}

// Sweep 包络预判法：由命题 1，h(p) ≤ E(p) 即阴影。全程一次扫掠，无逐格步进。
func Sweep(h []float64, size int, l Light) []bool {
	env := Envelope(h, size, l)
	out := make([]bool, size*size)
	for i, e := range env {
		out[i] = h[i] <= e
	}
	return out
}

// Mismatch 两法判定不一致的格子，含坐标与两侧高度（本格高度与包络高度）。
type Mismatch struct {
	X, Y     int
	Height   float64 // 本格高度 h(p)
	Envelope float64 // 包络高度 E(p)
	March    bool    // 步进法判定（true = 阴影）
	Sweep    bool    // 包络法判定
}

// Diff 逐格比对两种判定结果，返回不一致格明细。env 由 Envelope 求得，避免重算。
func Diff(h, env []float64, march, sweep []bool, size int) []Mismatch {
	var out []Mismatch
	for i := range march {
		if march[i] == sweep[i] {
			continue
		}
		out = append(out, Mismatch{
			X: i % size, Y: i / size,
			Height: h[i], Envelope: env[i],
			March: march[i], Sweep: sweep[i],
		})
	}
	return out
}

// Count 统计阴影格数。
func Count(b []bool) int {
	n := 0
	for _, v := range b {
		if v {
			n++
		}
	}
	return n
}

// CountRows 逐行统计阴影格数，返回 size 个计数。
func CountRows(b []bool, size int) []int {
	out := make([]int, size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			if b[y*size+x] {
				out[y]++
			}
		}
	}
	return out
}

// ToMask 把布尔阴影图转成灰度：受光 1.0，阴影 0.0，供 hmap.EncodePGM 使用。
func ToMask(b []bool) []float64 {
	out := make([]float64, len(b))
	for i, v := range b {
		if !v {
			out[i] = 1
		}
	}
	return out
}
