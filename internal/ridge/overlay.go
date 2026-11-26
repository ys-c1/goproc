package ridge

import (
	"math"
)

// Overlay 把高度图（灰度）与脊线叠加成三通道图像，尺寸为 outN×outN。
//
// 若 outN == N，按格点直接映射；否则先做双线性放大。
// 高度图为灰色底，Mask 用暗红半透明，裁剪后的脊线用亮青色画在最上层。
// 返回值是 RGB24 的字节序列，可直接写 PPM。
func (f *Field) Overlay(mask []bool, r Ridge, outN int) []byte {
	if outN < 2 {
		outN = f.N
	}
	scale := float64(f.N-1) / float64(outN-1)
	img := make([]byte, outN*outN*3)
	// 先把高度图铺到底层，再做叠加。
	for oy := 0; oy < outN; oy++ {
		for ox := 0; ox < outN; ox++ {
			h := f.Bilinear(f.F, float64(ox)*scale, float64(oy)*scale)
			if h < 0 {
				h = 0
			}
			if h > 1 {
				h = 1
			}
			g := uint8(h*255 + 0.5)
			i := (oy*outN + ox) * 3
			img[i], img[i+1], img[i+2] = g, g, g
		}
	}
	// Mask 用暗红着色，强度按面积比例压暗。
	if mask != nil {
		for oy := 0; oy < outN; oy++ {
			for ox := 0; ox < outN; ox++ {
				x := int(math.Floor(float64(ox) * scale))
				y := int(math.Floor(float64(oy) * scale))
				if x >= f.N {
					x = f.N - 1
				}
				if y >= f.N {
					y = f.N - 1
				}
				if !mask[y*f.N+x] {
					continue
				}
				i := (oy*outN + ox) * 3
				img[i] = mix(img[i], 255, 0.45)
				img[i+1] = mix(img[i+1], 40, 0.45)
				img[i+2] = mix(img[i+2], 40, 0.45)
			}
		}
	}
	// 脊线用亮青色，按折线画点并做一次 3×3 加粗。
	for _, s := range r.Segments {
		p := r.Points[s.A]
		q := r.Points[s.B]
		steps := int(math.Ceil(math.Hypot(q[0]-p[0], q[1]-p[1]) * float64(outN) / float64(f.N)))
		if steps < 1 {
			steps = 1
		}
		for k := 0; k <= steps; k++ {
			t := float64(k) / float64(steps)
			px := (p[0] + t*(q[0]-p[0])) / scale
			py := (p[1] + t*(q[1]-p[1])) / scale
			plot(img, outN, int(math.Round(px)), int(math.Round(py)), 0, 255, 255)
		}
	}
	return img
}

func mix(dst, src uint8, a float64) uint8 {
	v := float64(dst)*(1-a) + float64(src)*a
	if v < 0 {
		v = 0
	}
	if v > 255 {
		v = 255
	}
	return uint8(v + 0.5)
}

// OverlayWithArcs 在三通道叠加图上再画出上升合并树的弧（橙色），
// 以及 Atlas 要求的“任务书判据、脊判据”两套裁剪结果。
//
// protected 是 Mask 裁剪后的 G=0 曲线（品红），crest 是 L1<0 裁剪结果（青）。
// 合并树的每条弧沿格点路径绘制，让“段数 = 极大点 + 鞍点 - 1”可以直接目视核对。
func (f *Field) OverlayWithArcs(mask []bool, crest, protected Ridge, outN int, jt JoinTree) []byte {
	img := f.Overlay(mask, crest, outN)
	scale := float64(outN-1) / float64(f.N-1)
	toPix := func(x, y float64) (int, int) {
		return int(math.Round(x * scale)), int(math.Round(y * scale))
	}
	// Mask 裁剪的 G=0 曲线用品红细线。
	for _, s := range protected.Segments {
		p, q := protected.Points[s.A], protected.Points[s.B]
		x0, y0 := toPix(p[0], p[1])
		x1, y1 := toPix(q[0], q[1])
		line(img, outN, x0, y0, x1, y1, 255, 0, 200)
	}
	// 合并树弧用橙色，只描一次。
	for _, a := range jt.Arcs {
		for k := 1; k < len(a.Path); k++ {
			x0, y0 := toPix(float64(a.Path[k-1][0]), float64(a.Path[k-1][1]))
			x1, y1 := toPix(float64(a.Path[k][0]), float64(a.Path[k][1]))
			line(img, outN, x0, y0, x1, y1, 255, 150, 0)
		}
	}
	return img
}

// line 用 Bresenham 画 1 像素细线。
func line(img []byte, n, x0, y0, x1, y1 int, r, g, b uint8) {
	dx := abs(x1 - x0)
	dy := -abs(y1 - y0)
	sx := 1
	if x0 > x1 {
		sx = -1
	}
	sy := 1
	if y0 > y1 {
		sy = -1
	}
	err := dx + dy
	for {
		if x0 >= 0 && y0 >= 0 && x0 < n && y0 < n {
			i := (y0*n + x0) * 3
			img[i], img[i+1], img[i+2] = r, g, b
		}
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// plot 画 3×3 加粗点。
func plot(img []byte, n, x, y int, r, g, b uint8) {
	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			xx, yy := x+dx, y+dy
			if xx < 0 || yy < 0 || xx >= n || yy >= n {
				continue
			}
			i := (yy*n + xx) * 3
			img[i], img[i+1], img[i+2] = r, g, b
		}
	}
}

// EncodePPM 输出二进制 P6 格式。
//
// 注意：internal/hmap 的 EncodePGM 使用了 "P5\\n" 字面量，写出的文件头
// 是反斜杠加 n 而不是换行，因此这里的 PPM 头单独实现，避免沿用该缺陷。
func EncodePPM(n int, rgb []byte) []byte {
	head := make([]byte, 0, 32)
	head = append(head, "P6\n"...)
	head = appendInt(head, n)
	head = append(head, ' ')
	head = appendInt(head, n)
	head = append(head, "\n255\n"...)
	out := make([]byte, 0, len(head)+len(rgb))
	out = append(out, head...)
	out = append(out, rgb...)
	return out
}

func appendInt(b []byte, v int) []byte {
	if v == 0 {
		return append(b, '0')
	}
	var tmp [20]byte
	i := len(tmp)
	for v > 0 {
		i--
		tmp[i] = byte('0' + v%10)
		v /= 10
	}
	return append(b, tmp[i:]...)
}
