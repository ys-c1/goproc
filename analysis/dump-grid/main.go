// Command dump-grid 导出谱斜率/层数扫描所需的 PGM 高度图。
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"goproc/api"
)

func main() {
	out := flag.String("out", "artifacts/grid", "输出目录")
	size := flag.Int("size", 512, "边长")
	seed := flag.Uint64("seed", 7, "种子")
	freq := flag.Float64("freq", 0.02, "基础频率")
	flag.Parse()

	if err := os.MkdirAll(*out, 0o755); err != nil {
		panic(err)
	}
	gains := []float64{0.3535533905932738, 0.5, 0.7071067811865476} // beta = 3, 2, 1 under beta=-2 log2(gain)
	for _, gain := range gains {
		for oct := 1; oct <= 12; oct++ {
			h, err := api.Generate(api.Options{
				Size: *size, Octaves: oct, Lacunarity: 2, Gain: gain,
				Seed: *seed, Frequency: *freq,
			})
			if err != nil {
				panic(err)
			}
			name := fmt.Sprintf("s%02d_g%.6f_o%d.pgm", *seed, gain, oct)
			if err := os.WriteFile(filepath.Join(*out, name), h.ToPGM(), 0o644); err != nil {
				panic(err)
			}
		}
	}
	fmt.Println(*out)
}
