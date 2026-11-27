// Plateau 分析：阈值截断与平台面积。本文件为分析脚本，不属于生成管线。
package main

import (
	"fmt"

	"goproc/api"
)

func quantize(h float64) int {
	g := int(h*255 + 0.5)
	if g < 0 {
		g = 0
	}
	if g > 255 {
		g = 255
	}
	return g
}

func main() {
	hm, err := api.Generate(api.Options{Size: 128, Octaves: 4, Lacunarity: 2, Gain: 0.5, Seed: 7, Frequency: 0.02})
	if err != nil {
		panic(err)
	}
	n := hm.Size * hm.Size
	hist := make([]int, 256)
	minQ, maxQ := 255, 0
	for _, v := range hm.Data {
		q := quantize(v)
		hist[q]++
		if q < minQ {
			minQ = q
		}
		if q > maxQ {
			maxQ = q
		}
	}
	fmt.Printf("grid=%dx%d N=%d quant range=[%d,%d]\n", hm.Size, hm.Size, n, minQ, maxQ)
	// exceedance counts at candidate thresholds (level L means tau=L/255)
	tail := 0
	for L := maxQ; L >= 0; L-- {
		tail += hist[L]
		if L == 255 || L == 240 || L == 230 || L == 220 || L == 217 || L == 210 || L == 204 || L == 200 || L == 191 {
			fmt.Printf("tau=%.4f L=%3d  m(>tau)=%4d  m/N=%.4f  hist[L]=%d\n",
				float64(L)/255, L, tail, float64(tail)/float64(n), hist[L])
		}
	}
	fmt.Print("tail histogram L>=190: ")
	for L := 190; L <= maxQ; L++ {
		if hist[L] > 0 {
			fmt.Printf("%d:%d ", L, hist[L])
		}
	}
	fmt.Println()
}
