// terrainpairs 导出弱基线与散列扰动在三档种子间隔下的代表性 PGM 高度图,
// 用于目视确认"相邻种子像同一个地方"以及扩散后地形去相关。
package main

import (
	"fmt"
	"os"
	"path/filepath"
)

func writePGM(path string, size int, data []float64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if _, err := fmt.Fprintf(f, "P5\n%d %d\n255\n", size, size); err != nil {
		return err
	}
	buf := make([]byte, size*size)
	for i, v := range data {
		g := int(v*255 + 0.5)
		if g < 0 {
			g = 0
		}
		if g > 255 {
			g = 255
		}
		buf[i] = byte(g)
	}
	_, err = f.Write(buf)
	return err
}

func main() {
	dir := "artifacts/figures/terrain"
	if err := os.MkdirAll(dir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir failed:", err)
		os.Exit(1)
	}
	base := uint64(42)
	specs := []struct {
		name string
		makeP func(seed uint64) [256]uint8
	}{
		{"weak_gap1", func(s uint64) [256]uint8 { return weakPerm(s, weakK[1]) }},
		{"weak_gap16", func(s uint64) [256]uint8 { return weakPerm(s, weakK[16]) }},
		{"weak_gap256", func(s uint64) [256]uint8 { return weakPerm(s, weakK[256]) }},
		{"hash_q1_gap1", func(s uint64) [256]uint8 { return hashPerm(s, 1.0) }},
		{"hash_q1_gap256", func(s uint64) [256]uint8 { return hashPerm(s, 1.0) }},
	}
	for _, sp := range specs {
		var gap int
		switch sp.name {
		case "weak_gap1", "hash_q1_gap1":
			gap = 1
		case "weak_gap16":
			gap = 16
		default:
			gap = 256
		}
		p1 := sp.makeP(base)
		p2 := sp.makeP(base + uint64(gap))
		f1 := filepath.Join(dir, sp.name+"_a.pgm")
		f2 := filepath.Join(dir, sp.name+"_b.pgm")
		if err := writePGM(f1, mapSize, terrain(p1, mapSize, frequency)); err != nil {
			fmt.Fprintln(os.Stderr, "write failed:", err)
			os.Exit(1)
		}
		if err := writePGM(f2, mapSize, terrain(p2, mapSize, frequency)); err != nil {
			fmt.Fprintln(os.Stderr, "write failed:", err)
			os.Exit(1)
		}
		fmt.Println("wrote", f1, f2)
	}
}
