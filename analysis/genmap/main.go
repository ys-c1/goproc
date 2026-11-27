// 分析用生成器：按缩放比 1:k 批量生成高度图。
// 缩放比 1:k 映射为 Frequency = k * baseFreq（拉远 k 倍，空间频率升高 k 倍）。
// 每张图同时导出 PGM（目视）和原始 float64（分析，避免 8bit 量化损失）。
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"goproc/api"
	"goproc/internal/fbm"
	"goproc/internal/hmap"
	"goproc/internal/noise"
)

func main() {
	size := flag.Int("size", 256, "边长像素")
	seed := flag.Uint64("seed", 7, "种子")
	octaves := flag.Int("octaves", 4, "FBM 层数")
	baseFreq := flag.Float64("basefreq", 0.02, "1:1 时的基础频率")
	outDir := flag.String("outdir", "artifacts/maps", "输出目录")
	ss := flag.Int("ss", 1, "超采样倍数, >1 时盒滤波降采样回 size (抗混叠基准)")
	kList := flag.String("ks", "2,8", "缩放比分母列表，如 2,2.5,3.125,4,8")
	flag.Parse()

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir:", err)
		os.Exit(1)
	}

	for _, tok := range strings.Split(*kList, ",") {
		k, err := strconv.ParseFloat(strings.TrimSpace(tok), 64)
		if err != nil {
			fmt.Fprintln(os.Stderr, "bad k:", tok)
			os.Exit(2)
		}
		opt := api.DefaultOptions()
		opt.Size = *size
		opt.Seed = *seed
		opt.Octaves = *octaves
		opt.Frequency = k * *baseFreq

		var data []float64
		if *ss > 1 {
			data = supersampled(opt, *ss)
		} else {
			h, err := api.Generate(opt)
			if err != nil {
				fmt.Fprintln(os.Stderr, "generate k="+tok+":", err)
				os.Exit(1)
			}
			data = h.Data
		}
		stem := filepath.Join(*outDir, fmt.Sprintf("map_k%06.3f", k))

		pgm := hmap.EncodePGM(*size, data)
		if err := os.WriteFile(stem+".pgm", pgm, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write pgm:", err)
			os.Exit(1)
		}
		// 原始浮点：小端 float64，先行后列，附带同名单列参数侧车。
		f, err := os.Create(stem + ".f64")
		if err != nil {
			fmt.Fprintln(os.Stderr, "write f64:", err)
			os.Exit(1)
		}
		if err := binary.Write(f, binary.LittleEndian, data); err != nil {
			fmt.Fprintln(os.Stderr, "binary write:", err)
			os.Exit(1)
		}
		f.Close()

		topMult := 1 << (*octaves - 1) // 最高 FBM 层相对基础频率的倍数
		fm := opt.Frequency * float64(topMult)
		fmt.Printf("k=%.4f freq=%.5f fm=%.5f r=%.4f -> %s.pgm\n",
			k, opt.Frequency, fm, fm/1.0, stem)
	}
}

// supersampled 在 ss×ss 网格上求值, 每个输出像素取对应 ss×ss 块均值(盒滤波),
// 再做与 api.Generate 相同的线性归一化。盒滤波把基带外能量压下去, 作为无混叠基准。
func supersampled(opt api.Options, ss int) []float64 {
	src := noise.New(opt.Seed)
	n := opt.Size
	hi := n * ss
	grid := make([]float64, hi*hi)
	for y := 0; y < hi; y++ {
		for x := 0; x < hi; x++ {
			grid[y*hi+x] = fbm.Eval(src, float64(x)*opt.Frequency/float64(ss),
				float64(y)*opt.Frequency/float64(ss),
				opt.Octaves, opt.Lacunarity, opt.Gain)
		}
	}
	out := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			sum := 0.0
			for dy := 0; dy < ss; dy++ {
				base := (y*ss + dy) * hi
				for dx := 0; dx < ss; dx++ {
					sum += grid[base+x*ss+dx]
				}
			}
			out[y*n+x] = sum / float64(ss*ss)
		}
	}
	hmap.Normalize(out)
	return out
}
