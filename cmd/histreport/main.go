// histreport 对比拉平前后的值分布：对三档层数输出十六箱计数方差、
// 单调映射曲线，并为每档保存一幅前后对比直方图。
package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"

	"goproc/internal/fbm"
	"goproc/internal/hmap"
	"goproc/internal/noise"
)

const bins = 16

func rawField(size, octaves int, seed uint64) []float64 {
	src := noise.New(seed)
	out := make([]float64, size*size)
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			out[y*size+x] = fbm.Eval(src, float64(x)*0.02, float64(y)*0.02, octaves, 2.0, 0.5)
		}
	}
	return out
}

// drawHist 画上下两排直方图，上排是拉平前，下排是拉平后。
func drawHist(before, after []int, maxCount int) *image.Gray {
	barW, gap, panelH := 24, 4, 120
	w := bins * barW
	img := image.NewGray(image.Rect(0, 0, w, 2*panelH+3*gap))
	white := color.Gray{Y: 255}
	draw := func(counts []int, top int) {
		for b, c := range counts {
			h := 0
			if maxCount > 0 {
				h = c * panelH / maxCount
			}
			for x := b*barW + gap/2; x < (b+1)*barW-gap/2; x++ {
				for y := top + panelH - h; y < top+panelH; y++ {
					img.SetGray(x, y, white)
				}
			}
		}
	}
	draw(before, gap)
	draw(after, 2*gap+panelH)
	return img
}

func maxOf(a, b []int) int {
	m := 0
	for _, s := range [][]int{a, b} {
		for _, c := range s {
			if c > m {
				m = c
			}
		}
	}
	return m
}

func main() {
	outDir := flag.String("outdir", "artifacts", "直方图输出目录")
	size := flag.Int("size", 512, "边长")
	seed := flag.Uint64("seed", 1, "种子")
	flag.Parse()
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir failed:", err)
		os.Exit(1)
	}
	for _, oct := range []int{1, 4, 8} {
		raw := rawField(*size, oct, *seed)
		hmap.Normalize(raw) // 旧管线：只做线性拉伸
		before := hmap.HistCounts(raw, bins)
		m := hmap.Equalize(raw) // 新管线：经验累积分布单调映射
		after := hmap.HistCounts(raw, bins)
		fmt.Printf("octaves=%d 拉平前方差=%.1f 拉平后方差=%.1f\n",
			oct, hmap.CountVariance(before), hmap.CountVariance(after))
		for i := 0; i < bins; i++ {
			x := (float64(i) + 0.5) / bins
			fmt.Printf("  bin=%2d 箱中心=%.4f 映射值=%.4f 前计数=%6d 后计数=%6d\n",
				i, x, m.Apply(x), before[i], after[i])
		}
		img := drawHist(before, after, maxOf(before, after))
		path := filepath.Join(*outDir, fmt.Sprintf("hist_oct%d.png", oct))
		f, err := os.Create(path)
		if err != nil {
			fmt.Fprintln(os.Stderr, "create failed:", err)
			os.Exit(1)
		}
		if err := png.Encode(f, img); err != nil {
			f.Close()
			fmt.Fprintln(os.Stderr, "encode failed:", err)
			os.Exit(1)
		}
		f.Close()
		fmt.Printf("  直方图已写入 %s（上排拉平前，下排拉平后）\n", path)
	}
}
