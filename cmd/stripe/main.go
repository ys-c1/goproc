// Command stripe 分析等高距与条纹强度的关系。
//
// 模型：设 g 为期望坡度（灰度档/像素），等高距 Δ（灰度档）对应的等高线
// 间距为 s = Δ/g（像素）。条纹强度取两项之和：
//
//	I(Δ) = Δ/g + g/Δ
//
// 第一项随间距增大而上升（太稀看不出起伏），第二项随间距减小而上升
// （太密看着乱）。I 在 Δ* = g 处取最小值 2，是先降后升的函数。
//
// 实测：把高度图量化到 0..255 灰度，按等高距 Δ 提取等高线像素
// （量化档位跨越边界的像素），把图划分为 16x16 邻域窗口，统计每个窗口
// 的等高线像素数，取其方差作为条纹强度。
package main

import (
	"fmt"
	"math"

	"goproc/api"
)

const (
	size   = 256
	seed   = 7
	window = 16
	maxD   = 64 // 细扫等高距上限（灰度档）
)

// gradMean 返回期望坡度 |∇h|，单位灰度档/像素，中心差分。
func gradMean(h *api.Heightmap) float64 {
	n := h.Size
	sum := 0.0
	cnt := 0
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			xm, xp := x-1, x+1
			if xm < 0 {
				xm = 0
			}
			if xp >= n {
				xp = n - 1
			}
			ym, yp := y-1, y+1
			if ym < 0 {
				ym = 0
			}
			if yp >= n {
				yp = n - 1
			}
			gx := (h.At(xp, y) - h.At(xm, y)) / float64(xp-xm)
			gy := (h.At(x, yp) - h.At(x, ym)) / float64(yp-ym)
			sum += math.Hypot(gx, gy)
			cnt++
		}
	}
	return sum / float64(cnt) * 255 // 归一化高度 -> 灰度档
}

// quantize 把归一化高度量化为 0..255 灰度档（与 PGM 导出一致）。
func quantize(h *api.Heightmap) []int {
	q := make([]int, h.Size*h.Size)
	for i, v := range h.Data {
		g := int(v*255 + 0.5)
		if g < 0 {
			g = 0
		}
		if g > 255 {
			g = 255
		}
		q[i] = g
	}
	return q
}

// contourMask 按等高距 delta（灰度档）标记等高线像素：
// 像素与右邻或下邻的量化档位 floor(g/delta) 不同即为等高线。
func contourMask(q []int, n, delta int) []bool {
	m := make([]bool, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			g := q[y*n+x] / delta
			if x+1 < n && q[y*n+x+1]/delta != g {
				m[y*n+x] = true
			}
			if y+1 < n && q[(y+1)*n+x]/delta != g {
				m[y*n+x] = true
			}
		}
	}
	return m
}

// stripeVar 把图划分为 window x window 邻域，统计每窗等高线像素数，
// 返回邻域方差（总体方差）与均值。
func stripeVar(mask []bool, n, win int) (variance, mean float64) {
	nw := n / win
	counts := make([]float64, 0, nw*nw)
	for wy := 0; wy < nw; wy++ {
		for wx := 0; wx < nw; wx++ {
			c := 0.0
			for y := wy * win; y < (wy+1)*win; y++ {
				for x := wx * win; x < (wx+1)*win; x++ {
					if mask[y*n+x] {
						c++
					}
				}
			}
			counts = append(counts, c)
		}
	}
	for _, c := range counts {
		mean += c
	}
	mean /= float64(len(counts))
	for _, c := range counts {
		d := c - mean
		variance += d * d
	}
	variance /= float64(len(counts))
	return variance, mean
}

// modelI 模型条纹强度 I(Δ) = Δ/g + g/Δ。
func modelI(delta, g float64) float64 { return delta/g + g/delta }

func main() {
	cases := []struct {
		name    string
		octaves int
	}{
		{"低复杂度 octaves=2", 2},
		{"高复杂度 octaves=8", 8},
	}
	for _, c := range cases {
		h, err := api.Generate(api.Options{Size: size, Octaves: c.octaves, Lacunarity: 2, Gain: 0.5, Seed: seed, Frequency: 0.02})
		if err != nil {
			panic(err)
		}
		q := quantize(h)
		g := gradMean(h)

		dStar := g // 模型解：Δ* = g
		dHat := math.Round(dStar)
		if dHat < 1 {
			dHat = 1
		}
		lossPct := (modelI(dHat, g) - 2) / 2 * 100 // 相对最小值 2 的取整损失

		// 三档：Δ̂/2、Δ̂、2Δ̂（整数灰度档，至少 1）
		l1 := int(math.Round(dHat / 2))
		if l1 < 1 {
			l1 = 1
		}
		l2 := int(dHat)
		l3 := int(dHat) * 2
		levels := []int{l1, l2, l3}

		// 细扫 1..maxD 找实测最优等高距与方差峰值位置
		vars := make(map[int]float64)
		bestD, bestV := 0, math.Inf(1)
		peakD, peakV := 0, math.Inf(-1)
		for d := 1; d <= maxD; d++ {
			v, _ := stripeVar(contourMask(q, size, d), size, window)
			vars[d] = v
			if v < bestV {
				bestV, bestD = v, d
			}
			if v > peakV {
				peakV, peakD = v, d
			}
		}

		fmt.Printf("== %s (size=%d seed=%d) ==\n", c.name, size, seed)
		fmt.Printf("期望坡度 g = %.4f 灰度档/像素\n", g)
		fmt.Printf("模型解 Δ* = g = %.4f 灰度档, 模型最小强度 I(Δ*) = 2\n", dStar)
		fmt.Printf("取整档位 Δ̂ = %d, 取整损失 = %.4f%% (相对模型最小值)\n", int(dHat), lossPct)
		fmt.Printf("细扫实测最优等高距 = %d 灰度档 (邻域方差 %.2f), 方差峰值在 Δ=%d (%.2f)\n", bestD, bestV, peakD, peakV)
		fmt.Printf("细扫曲线 V(Δ), Δ=1..%d:\n  ", maxD)
		for d := 1; d <= maxD; d++ {
			fmt.Printf("%d:%.0f ", d, vars[d])
			if d%8 == 0 {
				fmt.Printf("\n  ")
			}
		}
		fmt.Println()
		fmt.Printf("实测最优与取整解的档位偏差 = %.2f 档 (log2 比)\n", math.Log2(float64(bestD)/dHat))
		fmt.Printf("三档实测（窗口 %dx%d，强度=等高线像素数邻域方差）:\n", window, window)
		vHat := vars[l2]
		fmt.Printf("  %-8s %-14s %-14s %-16s %-16s\n", "等高距Δ", "实测方差V(Δ)", "模型I(Δ)/I_min", "实测V/V(Δ̂)", "与模型比值偏差")
		for _, d := range levels {
			mr := modelI(float64(d), g) / 2 // 模型归一化（最小=1）
			dr := vars[d] / vHat            // 实测归一化（预测最小档=1）
			dev := (dr - mr) / mr * 100
			flag := ""
			if math.Abs(dev) > 10 {
				flag = "  <-- 偏差超过一成"
			}
			fmt.Printf("  %-8d %-14.2f %-14.4f %-16.4f %+.2f%%%s\n", d, vars[d], mr, dr, dev, flag)
		}
		fmt.Println()
	}
}
