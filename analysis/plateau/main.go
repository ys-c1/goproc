// Plateau 分析：阈值截断、平台面积与超出质量下界。
//
// 数据管线与 cmd/goproc 完全一致（api.Generate），分析对象是 PGM 量化后的
// 0..255 整数高度场，即实际导出 map.pgm 的内容。
package main

import (
	"fmt"
	"os"
	"strconv"

	"goproc/api"
)

const (
	gridSize = 128
	seed     = 7
	octaves  = 4
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

// hardClip 硬截断：L 以上全部压到 L。
func hardClip(g, L int) int {
	if g > L {
		return L
	}
	return g
}

// softClip 软截断：L 以上的超出量按斜率 s∈(0,1) 压缩，保持严格单调，
// 只在量化合并时才出现同高邻格（形成的是坡度平台而非封顶平台）。
func softClip(g, L int, s float64) int {
	if g > L {
		return L + int(float64(g-L)*s+0.5)
	}
	return g
}

// components 对网格做四邻域连通分量（同高度才连通）。
// 返回分量大小列表，以及“格 -> 分量编号”的映射。
func components(size int, h []int) ([]int, []int) {
	N := size * size
	comp := make([]int, N)
	for i := range comp {
		comp[i] = -1
	}
	var sizes []int
	for start := 0; start < N; start++ {
		if comp[start] != -1 {
			continue
		}
		id := len(sizes)
		comp[start] = id
		stack := []int{start}
		cnt := 0
		for len(stack) > 0 {
			p := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			cnt++
			x, y := p%size, p/size
			for _, nb := range [4][2]int{{x - 1, y}, {x + 1, y}, {x, y - 1}, {x, y + 1}} {
				nx, ny := nb[0], nb[1]
				if nx < 0 || nx >= size || ny < 0 || ny >= size {
					continue
				}
				q := ny*size + nx
				if comp[q] == -1 && h[q] == h[p] {
					comp[q] = id
					stack = append(stack, q)
				}
			}
		}
		sizes = append(sizes, cnt)
	}
	return sizes, comp
}

type result struct {
	cappedCells   int // 截断后位于封顶高度 L 的格数（含 L 以下本来就是 L 的格）
	cappedComps   int // 封顶高度 L 上的四邻域连通分量数
	singletons    int // 封顶高度上大小为 1 的分量数
	plateauCells  int // 封顶高度上大小 >=2 的分量所占格数（真正的平台）
	slopeCells    int // 原超出格中被压成坡度（截断后高度 != L）的格数
	tiedExceed    int // 软截断：原超出格量化后与某邻格同高、且高度 != L 的格数
	slopePlatCell int // 软截断：坡度带上大小 >=2 的同高连通分量格数
}

// plateauTotals 按“同高度 + 四邻域”统计全图：返回平台格总数（分量>=2）、
// 孤立单格总数、分量大小直方图（仅 >=2 的分量）。
func plateauTotals(size int, h []int) (int, int, map[int]int) {
	sizes, _ := components(size, h)
	plat, single := 0, 0
	dist := map[int]int{}
	for _, c := range sizes {
		if c == 1 {
			single++
		} else {
			plat += c
			dist[c]++
		}
	}
	return plat, single, dist
}

// bandCompSizes 返回指定高度区间 [lo,hi] 内各同高连通分量的大小列表。
func bandCompSizes(size int, h []int, lo, hi int) []int {
	sizes, comp := components(size, h)
	keep := map[int]bool{}
	for i, v := range h {
		if v >= lo && v <= hi {
			keep[comp[i]] = true
		}
	}
	var out []int
	for id := range keep {
		out = append(out, sizes[id])
	}
	return out
}

func analyze(size int, orig, clipped []int, L int, soft bool) result {
	N := size * size
	_, comp := components(size, clipped)
	compSize := map[int]int{}
	for _, id := range comp {
		compSize[id]++
	}
	var r result
	for i := 0; i < N; i++ {
		if orig[i] <= L {
			continue
		}
		// 只统计原超出集合里的格，下界 m 与它们一一对应。
		if clipped[i] == L {
			r.cappedCells++
			id := comp[i]
			if compSize[id] == 1 {
				r.singletons++
			} else {
				// 分量内所有格同高，该格高度为 L 即封顶平台
				r.plateauCells++
			}
		} else {
			r.slopeCells++
			x, y := i%size, i/size
			tied := false
			for _, nb := range [4][2]int{{x - 1, y}, {x + 1, y}, {x, y - 1}, {x, y + 1}} {
				nx, ny := nb[0], nb[1]
				if nx < 0 || nx >= size || ny < 0 || ny >= size {
					continue
				}
				if clipped[ny*size+nx] == clipped[i] {
					tied = true
					break
				}
			}
			if tied {
				r.tiedExceed++
				if compSize[comp[i]] >= 2 {
					r.slopePlatCell++
				}
			}
		}
	}
	// 封顶高度上的连通分量（分量可能包含非超出格，按分量整体统计）
	seen := map[int]bool{}
	for i := 0; i < N; i++ {
		if clipped[i] == L && !seen[comp[i]] {
			seen[comp[i]] = true
			r.cappedComps++
		}
	}
	_ = soft
	return r
}

func main() {
	levels := []int{217, 240}
	softSlope := 0.5
	if len(os.Args) > 1 {
		if v, err := strconv.ParseFloat(os.Args[1], 64); err == nil {
			softSlope = v
		}
	}

	hm, err := api.Generate(api.Options{Size: gridSize, Octaves: octaves, Lacunarity: 2, Gain: 0.5, Seed: seed, Frequency: 0.02})
	if err != nil {
		panic(err)
	}
	size := hm.Size
	N := size * size
	orig := make([]int, N)
	for i, v := range hm.Data {
		orig[i] = quantize(v)
	}

	fmt.Printf("grid=%dx%d N=%d seed=%d octaves=%d softSlope=%.2f\n\n", size, size, N, seed, octaves, softSlope)
	op, os, _ := plateauTotals(size, orig)
	fmt.Printf("未截断原图: 全图平台格=%d (%.2f%%) 孤立单格=%d\n\n", op, 100*float64(op)/float64(N), os)
	for _, L := range levels {
		m := 0
		for _, g := range orig {
			if g > L {
				m++
			}
		}
		hard := make([]int, N)
		soft := make([]int, N)
		for i, g := range orig {
			hard[i] = hardClip(g, L)
			soft[i] = softClip(g, L, softSlope)
		}
		rh := analyze(size, orig, hard, L, false)
		rs := analyze(size, orig, soft, L, true)

		// 全图平台统计（同高度四邻域，分量 >=2 记为平台）
		hp, hs, _ := plateauTotals(size, hard)
		sp, ss, sdist := plateauTotals(size, soft)
		band := bandCompSizes(size, soft, L+1, L+int(float64(255-L)*softSlope+0.5))
		bandComps, bandPlat := len(band), 0
		for _, c := range band {
			if c >= 2 {
				bandPlat += c
			}
		}

		fmt.Printf("===== 阈值 L=%d (tau=%.4f)  超出质量 m=%d (m/N=%.4f) =====\n", L, float64(L)/255, m, float64(m)/float64(N))
		fmt.Printf("硬截断: 封顶格=%d 封顶分量=%d 孤立单格=%d 封顶平台格(分量>=2)=%d 坡度格=%d\n",
			rh.cappedCells, rh.cappedComps, rh.singletons, rh.plateauCells, rh.slopeCells)
		fmt.Printf("  恒等式 m = 平台格 + 孤立单格 + 坡度格: %d = %d+%d+%d = %d\n",
			m, rh.plateauCells, rh.singletons, rh.slopeCells, rh.plateauCells+rh.singletons+rh.slopeCells)
		fmt.Printf("  全图平台格=%d 全图孤立单格=%d\n", hp, hs)
		fmt.Printf("软截断: 封顶格=%d 封顶分量=%d 孤立单格=%d 封顶平台格=%d 坡度格=%d\n",
			rs.cappedCells, rs.cappedComps, rs.singletons, rs.plateauCells, rs.slopeCells)
		fmt.Printf("  恒等式 m = 平台格 + 孤立单格 + 坡度格: %d = %d+%d+%d = %d\n",
			m, rs.plateauCells, rs.singletons, rs.slopeCells, rs.plateauCells+rs.singletons+rs.slopeCells)
		fmt.Printf("  坡度带[%d,%d]: 分量数=%d 其中大小>=2的平台格=%d; 原超出格坡度同高=%d\n",
			L+1, L+int(float64(255-L)*softSlope+0.5), bandComps, bandPlat, rs.tiedExceed)
		fmt.Printf("  全图平台格=%d 全图孤立单格=%d 软截断分量大小分布(>=2): %v\n", sp, ss, sdist)
		fmt.Println()
	}
}
