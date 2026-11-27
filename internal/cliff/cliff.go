// Package cliff 建立陡崖倒角的双约束模型（几何约束与灰度量化约束），
// 并提供量化后的过渡带实测函数。高度一律用归一化值，零为崖底、H 为崖顶。
package cliff

import "math"

// Levels 返回指定位深的灰度级数，八位为 256，十六位为 65536。
func Levels(bits int) int { return 1 << bits }

// Quantize 按四舍五入把归一化高度量化为灰度码，
// 口径与 hmap.EncodePGM 的八位编码一致，十六位 PGM 同源。
func Quantize(h float64, levels int) int {
	q := int(h*float64(levels-1) + 0.5)
	if q < 0 {
		return 0
	}
	if q > levels-1 {
		return levels - 1
	}
	return q
}

// GeomLimit 是几何约束 w_g：设计陡度 s（每像素落差）下，
// 容纳整个落差 H 所需的最小水平像素数 H/s。
// s 等于倒角面相对水平平台的坡度角 β 的正切值 tanβ，
// 而倒角法线与平台法线（竖直方向）的夹角同样等于 β。
func GeomLimit(H, s float64) float64 { return H / s }

// QuantLimit 是量化约束 w_q：灰度量化能分辨的最大过渡像素数。
// 整个落差只对应 H(L-1) 个量化台阶，每个台阶至少要占一个像素，
// 超过这个宽度就只能重复灰度码，不会再增加可分辨层次。
func QuantLimit(H float64, levels int) float64 { return H * float64(levels-1) }

// UsedWidth 是实际可用倒角宽度，取两条限制中的较小者。
func UsedWidth(H, s float64, levels int) float64 {
	return math.Min(GeomLimit(H, s), QuantLimit(H, levels))
}

// CrossoverSteepness 返回两约束相等处的临界陡度 s* = 1/(L-1)。
// s 大于它时几何约束更紧（几何主导），小于它时量化约束更紧（量化主导）。
func CrossoverSteepness(levels int) float64 { return 1 / float64(levels-1) }

// CrossoverWidth 返回交点处的倒角宽度 w* = H(L-1)。
func CrossoverWidth(H float64, levels int) float64 { return H * float64(levels-1) }

// UnclampedRamp 生成 n 个像素的线性倒角剖面，按像素左边缘采样，
// 即第 i 个像素高度为 H - H·i/n，像素 i 覆盖水平位置 i 到 i+1。
// 不施加任何宽度限制。
func UnclampedRamp(H float64, n int) []float64 {
	ramp := make([]float64, n)
	for i := range ramp {
		ramp[i] = H - H*float64(i)/float64(n)
	}
	return ramp
}

// ClampedRamp 按 min(w_g, w_q) 四舍五入取整后构造线性倒角。
// 量化主导时宽度被压到量化上限，多余落差只能停在最高一级台阶上。
func ClampedRamp(H, s float64, levels int) []float64 {
	n := int(float64(math.Round(UsedWidth(H, s, levels))))
	return UnclampedRamp(H, n)
}

// Profile1D 拼接顶部平台、倒角、底部平台，flank 为每侧平台像素数。
func Profile1D(H float64, ramp []float64, flank int) []float64 {
	p := make([]float64, 0, flank+len(ramp)+flank)
	for i := 0; i < flank; i++ {
		p = append(p, H)
	}
	p = append(p, ramp...)
	for i := 0; i < flank; i++ {
		p = append(p, 0)
	}
	return p
}

// QuantizeLine 量化整条一维剖面。
func QuantizeLine(line []float64, levels int) []int {
	codes := make([]int, len(line))
	for i, h := range line {
		codes[i] = Quantize(h, levels)
	}
	return codes
}

// BandStats 统计灰度过渡带。top、bot 为两端平台的灰度码；
// B 是严格介于两平台码之间的像素数，即过渡带占据的空间像素数；
// D 是这些像素中出现的不同灰度码数，即量化真正分辨出的台阶数。
func BandStats(codes []int) (top, bot, B, D int) {
	top, bot = codes[0], codes[len(codes)-1]
	seen := make(map[int]struct{})
	for _, c := range codes {
		if c > bot && c < top {
			B++
			seen[c] = struct{}{}
		}
	}
	D = len(seen)
	return
}

// Extrude 把一维剖面逐行复制成 size×size 二维高度图，陡崖沿水平方向延伸。
func Extrude(line []float64) (int, []float64) {
	size := len(line)
	out := make([]float64, size*size)
	for y := 0; y < size; y++ {
		copy(out[y*size:], line)
	}
	return size, out
}
