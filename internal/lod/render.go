package lod

import (
	"fmt"
	"strings"
)

// levelColors 是 LOD 层级 0..4 的序数色阶（单蓝色相，由浅到深），
// 已经 dataviz 序数色板校验（明度单调、相邻 ΔL≥0.06、浅端对比度达标）。
var levelColors = [][3]int{
	{134, 182, 239}, // #86b6ef L0 最粗
	{85, 152, 231},  // #5598e7 L1
	{42, 120, 214},  // #2a78d6 L2
	{28, 92, 171},   // #1c5cab L3
	{16, 66, 129},   // #104281 L4 最细
}

// LevelColor 返回某层级的 RGB；超界钳到合法层级。
func LevelColor(level, lmax int) [3]int {
	if level < 0 {
		level = 0
	}
	if level > lmax || level >= len(levelColors) {
		return levelColors[len(levelColors)-1]
	}
	return levelColors[level]
}

// EncodeLevelPPM 把层级矩阵渲染为二进制 PPM 色块图。
// 每个块放大为 cellPx 见方的纯色单元格，单元格间用表面色细线分隔。
// 行优先的 levels 与 Grid.LevelGrid 输出一致。
func EncodeLevelPPM(levels [][]int, lmax, cellPx int) []byte {
	n := len(levels)
	img := n * cellPx
	var b strings.Builder
	fmt.Fprintf(&b, "P6\n%d %d\n255\n", img, img)
	out := []byte(b.String())
	surface := [3]int{252, 252, 251} // #fcfcfb，与色板校验表面一致
	gap := 1
	for ry := 0; ry < img; ry++ {
		r := ry / cellPx
		for cx := 0; cx < img; cx++ {
			c := cx / cellPx
			inGap := (cx%cellPx < gap) || (ry%cellPx < gap)
			col := LevelColor(levels[r][c], lmax)
			if inGap {
				col = surface
			}
			out = append(out, byte(col[0]), byte(col[1]), byte(col[2]))
		}
	}
	return out
}

// LegendSVG 返回层级色阶的 SVG 图例（含层级号与“粗→细”方向）。
func LegendSVG(lmax int) string {
	const w, h, sw = 560, 64, 96
	var sb strings.Builder
	fmt.Fprintf(&sb, `<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" font-family="system-ui,sans-serif">`, w, h, w, h)
	fmt.Fprintf(&sb, `<text x="8" y="20" font-size="13" fill="#52514e">LOD 层级（浅=粗 L0 → 深=细 L%d）</text>`, lmax)
	for l := 0; l <= lmax; l++ {
		col := LevelColor(l, lmax)
		x := 8 + l*sw
		fmt.Fprintf(&sb, `<rect x="%d" y="30" width="%d" height="26" fill="#%02x%02x%02x"/>`, x, sw-6, col[0], col[1], col[2])
		ink := "#ffffff"
		if l <= 1 {
			ink = "#0b0b0b"
		}
		fmt.Fprintf(&sb, `<text x="%d" y="48" font-size="14" text-anchor="middle" fill="%s">L%d</text>`, x+(sw-6)/2, ink, l)
	}
	sb.WriteString(`</svg>`)
	return sb.String()
}
