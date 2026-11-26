// Package render 把高度图渲染成坡向着色图层与投射阴影图层，并给出冲突判据。
//
// 两个图层各自独立生成：
//
//	坡向着色  shading = ambient + (1-ambient) * max(0, N·L)   只关心坡面朝向
//	投射阴影  shadow  = 1 - occlusion                        只关心沿光线方向是否有地形挡光
//
// 冲突判据见 Criteria / Conflict。
package render

import (
	"math"
)

// Heightmap 只依赖网格尺寸与取值，避免与 hmap 互相绑定。
type Heightmap struct {
	Size int
	Data []float64
}

// At 取某点高度。
func (h *Heightmap) At(x, y int) float64 { return h.Data[y*h.Size+x] }

// AtClamped 取某点高度，越界时钳到边界。
func (h *Heightmap) AtClamped(x, y int) float64 {
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	if x >= h.Size {
		x = h.Size - 1
	}
	if y >= h.Size {
		y = h.Size - 1
	}
	return h.At(x, y)
}

// Params 是一次渲染的全部参数。
type Params struct {
	Azimuth float64 // 光照方位角，弧度，两个图层共用
	Elev    float64 // 光照仰角，弧度
	Ambient float64 // 环境光下限，0..1
}

// DefaultParams 返回默认参数：低位太阳，让阴影可辨。
func DefaultParams() Params {
	return Params{Azimuth: math.Pi / 4, Elev: 2 * math.Pi / 180, Ambient: 0.25}
}

// normalAt 返回中央差分法向量（单位向量）。
func (h *Heightmap) normalAt(x, y int) (float64, float64, float64) {
	dx := (h.AtClamped(x+1, y) - h.AtClamped(x-1, y)) * 0.5
	dy := (h.AtClamped(x, y+1) - h.AtClamped(x, y-1)) * 0.5
	nx, ny, nz := -dx, -dy, 1.0
	l := math.Sqrt(nx*nx + ny*ny + nz*nz)
	return nx / l, ny / l, nz / l
}

// DirectionalSlope 返回沿光照方位的方向导数 u = ∇h·(cos az, sin az)。
// u > 0 表示地面沿光线方向升高（坡面背向光源），u < 0 表示迎光。
func (h *Heightmap) DirectionalSlope(az float64) []float64 {
	ca, sa := math.Cos(az), math.Sin(az)
	out := make([]float64, len(h.Data))
	for y := 0; y < h.Size; y++ {
		for x := 0; x < h.Size; x++ {
			dx := (h.AtClamped(x+1, y) - h.AtClamped(x-1, y)) * 0.5
			dy := (h.AtClamped(x, y+1) - h.AtClamped(x, y-1)) * 0.5
			out[y*h.Size+x] = dx*ca + dy*sa
		}
	}
	return out
}

// Incidence 返回 N·L，负值截零。
func (h *Heightmap) Incidence(p Params) []float64 {
	ca, sa := math.Cos(p.Azimuth), math.Sin(p.Azimuth)
	lx, ly, lz := math.Cos(p.Elev)*ca, math.Cos(p.Elev)*sa, math.Sin(p.Elev)
	out := make([]float64, len(h.Data))
	for y := 0; y < h.Size; y++ {
		for x := 0; x < h.Size; x++ {
			nx, ny, nz := h.normalAt(x, y)
			d := nx*lx + ny*ly + nz*lz
			if d < 0 {
				d = 0
			}
			out[y*h.Size+x] = d
		}
	}
	return out
}

// Shading 计算坡向着色图层，值域 0..1。
func Shading(h *Heightmap, p Params) []float64 {
	inc := h.Incidence(p)
	out := make([]float64, len(inc))
	for i, d := range inc {
		out[i] = p.Ambient + (1-p.Ambient)*d
	}
	return out
}

// DepthGap 返回每个格点沿光线方向最近遮挡体的深度差：
//
//	gap(x,y) = max_{s>0} ( h(x + s·cos az, y + s·sin az) - h(x,y) - s·tan elev )_+
//
// 即遮挡体高出光线的量。gap == 0 表示该点没有被遮挡（光线一路走空）。
func (h *Heightmap) DepthGap(p Params) []float64 {
	ca, sa := math.Cos(p.Azimuth), math.Sin(p.Azimuth)
	tan := math.Tan(p.Elev)
	if tan <= 0 {
		return make([]float64, len(h.Data))
	}
	reach := math.Min(1/tan, float64(h.Size))
	out := make([]float64, len(h.Data))
	for y := 0; y < h.Size; y++ {
		for x := 0; x < h.Size; x++ {
			best := 0.0
			for s := 1.0; s <= reach; s++ {
				sx := float64(x) + ca*s
				sy := float64(y) + sa*s
				ix, iy := int(math.Round(sx)), int(math.Round(sy))
				if ix < 0 || iy < 0 || ix >= h.Size || iy >= h.Size {
					break
				}
				if d := h.At(ix, iy) - h.At(x, y) - s*tan; d > best {
					best = d
				}
			}
			out[y*h.Size+x] = best
		}
	}
	return out
}

// Shadow 计算投射阴影图层，1 为全亮，0 为全遮。只依赖仰角与方位角，与坡向着色无共享中间量。
func Shadow(h *Heightmap, p Params, scale float64) []float64 {
	gap := h.DepthGap(p)
	if scale <= 0 {
		scale = 0.05
	}
	out := make([]float64, len(gap))
	for i, g := range gap {
		out[i] = math.Exp(-g / scale)
	}
	return out
}

// Criteria 把两条判据分别摊开。
type Criteria struct {
	// Lit 为判据 A：坡向朝光，N·L ≥ 0，等价于 u ≤ tan(elev)。
	Lit []bool
	// Marginal 为判据 B：深度差在 (0, tau] 内，遮挡存在但只差一点。
	Marginal []bool
	// Conflict 为 A∧B。
	Conflict []bool
}

// Conflict 按两条判据求合取。tau 是深度差阈值。
func (h *Heightmap) Conflict(p Params, tau float64) Criteria {
	tanElev := math.Tan(p.Elev)
	u := h.DirectionalSlope(p.Azimuth)
	gap := h.DepthGap(p)
	n := len(h.Data)
	c := Criteria{
		Lit:      make([]bool, n),
		Marginal: make([]bool, n),
		Conflict: make([]bool, n),
	}
	for i := range h.Data {
		c.Lit[i] = u[i] <= tanElev
		c.Marginal[i] = gap[i] > 0 && gap[i] <= tau
		c.Conflict[i] = c.Lit[i] && c.Marginal[i]
	}
	return c
}

// Count 返回判定为真的格点数。
func Count(mask []bool) int {
	n := 0
	for _, v := range mask {
		if v {
			n++
		}
	}
	return n
}
