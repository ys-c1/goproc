package fog

import (
	"image"
	"math"
)

// RenderOptions 控制带雾渲染。
type RenderOptions struct {
	Chi                  float64 // beta * R，雾浓度与起伏的无量纲乘积
	Epsilon              float64 // 对比度阈值，可见判据 C>Epsilon
	FogGray              float64 // 雾色灰度 0..1，通常接近 1（白雾）
	VerticalExaggeration float64 // 晕渲垂直夸张系数，控制山体立体感
	Ambient              float64 // 环境光比例 0..1
}

// DefaultRenderOptions 返回一组经目视调整的渲染参数。
func DefaultRenderOptions(chi float64) RenderOptions {
	return RenderOptions{
		Chi:                  chi,
		Epsilon:              DefaultEpsilon,
		FogGray:              0.92,
		VerticalExaggeration: 22,
		Ambient:              0.26,
	}
}

// Result 是一次带雾渲染的全部逐像素结果。
type Result struct {
	Size       int
	Hill       []float64 // 山体晕渲光照 0..1
	Base       []float64 // 无雾时的地表灰度 0..1
	Contrast   []float64 // 雾后残余对比度 C=exp(-chi(1-h))
	Composite  []float64 // 带雾合成灰度 0..1
	Visible    []bool    // C>Epsilon
	FloatCount Count     // 直接对浮点渲染缓冲逐像素统计
	BaseGray   *image.Gray
	Gray       *image.Gray
	Contrast8  *image.Gray
	Mask       *image.Gray
}

func atClamped(h []float64, n, x, y int) float64 {
	if x < 0 {
		x = 0
	} else if x >= n {
		x = n - 1
	}
	if y < 0 {
		y = 0
	} else if y >= n {
		y = n - 1
	}
	return h[y*n+x]
}

// Hillshade 用中心差分做法线与西北方向光源点积，返回 0..1 光照强度。
func Hillshade(h []float64, n int, exaggeration, ambient float64) []float64 {
	out := make([]float64, n*n)
	// 指向光源（左上、俯角约 45 度）的单位向量。
	lx, ly, lz := -0.55, -0.55, 0.628
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			hl := atClamped(h, n, x-1, y)
			hr := atClamped(h, n, x+1, y)
			hu := atClamped(h, n, x, y-1)
			hd := atClamped(h, n, x, y+1)
			dzdx := exaggeration * (hr - hl) / 2
			dzdy := exaggeration * (hd - hu) / 2
			nx, ny, nz := -dzdx, -dzdy, 1.0
			norm := math.Sqrt(nx*nx + ny*ny + nz*nz)
			dot := (nx*lx + ny*ly + nz*lz) / norm
			if dot < 0 {
				dot = 0
			}
			out[y*n+x] = clamp01(ambient + (1-ambient)*dot)
		}
	}
	return out
}

// Render 执行晕渲、雾合成与可见性判定，并生成 8-bit 图像缓冲。
func Render(h []float64, n int, opt RenderOptions) *Result {
	if opt.Epsilon <= 0 {
		opt.Epsilon = DefaultEpsilon
	}
	if opt.FogGray <= 0 {
		opt.FogGray = 0.92
	}
	hill := Hillshade(h, n, opt.VerticalExaggeration, opt.Ambient)

	r := &Result{
		Size:      n,
		Hill:      hill,
		Base:      make([]float64, n*n),
		Contrast:  make([]float64, n*n),
		Composite: make([]float64, n*n),
		Visible:   make([]bool, n*n),
		BaseGray:  image.NewGray(image.Rect(0, 0, n, n)),
		Gray:      image.NewGray(image.Rect(0, 0, n, n)),
		Contrast8: image.NewGray(image.Rect(0, 0, n, n)),
		Mask:      image.NewGray(image.Rect(0, 0, n, n)),
	}

	vis := 0
	for i, hv := range h {
		// 海拔灰度 + 晕渲调制，得到无雾地表（整体压深以衬托白雾）。
		elev := 0.14 + 0.44*clamp01(hv)
		base := clamp01(elev * (0.42 + 0.92*hill[i]))
		r.Base[i] = base
		r.BaseGray.Pix[i] = gray8(base)

		c := ContrastAt(hv, opt.Chi)
		r.Contrast[i] = c

		// 白雾合成：I = C*base + (1-C)*FogGray。
		comp := c*base + (1-c)*opt.FogGray
		r.Composite[i] = clamp01(comp)

		ok := c > opt.Epsilon // 严格“高于”阈值
		r.Visible[i] = ok
		if ok {
			vis++
			r.Mask.Pix[i] = 255
		}

		r.Gray.Pix[i] = gray8(r.Composite[i])
		r.Contrast8.Pix[i] = gray8(c)
	}

	r.FloatCount = Count{
		Visible:      vis,
		Total:        n * n,
		Fraction:     float64(vis) / float64(n*n),
		Chi:          opt.Chi,
		Epsilon:      opt.Epsilon,
		TauStar:      TauThreshold(opt.Epsilon),
		CutoffHeight: HeightCutoff(opt.Chi, opt.Epsilon),
	}
	return r
}

func gray8(v float64) uint8 {
	g := int(math.Round(clamp01(v) * 255))
	if g < 0 {
		return 0
	}
	if g > 255 {
		return 255
	}
	return uint8(g)
}

// CountContrastGray8 从保存/回读的 8-bit 对比度图逐像素统计：
// C8 = Y/255，可见 iff C8 > epsilon。这是真正“看成图文件”的口径。
func CountContrastGray8(g *image.Gray, chi, eps float64) Count {
	b := g.Bounds()
	total := b.Dx() * b.Dy()
	vis := 0
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			c := float64(g.GrayAt(x, y).Y) / 255.0
			if c > eps {
				vis++
			}
		}
	}
	return Count{
		Visible:      vis,
		Total:        total,
		Fraction:     float64(vis) / float64(total),
		Chi:          chi,
		Epsilon:      eps,
		TauStar:      TauThreshold(eps),
		CutoffHeight: HeightCutoff(chi, eps),
	}
}
