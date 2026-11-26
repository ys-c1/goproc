// 命令 dumphm 调用项目自身的 api.Generate，把归一化后的全精度高度场
// 导出为二进制，供等高线分析读取，避免 8-bit PGM 量化台阶。
// 格式（小端）：magic "GOPROC-HM"，size uint32，随后 size*size 个 float64。
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"os"

	"goproc/api"
)

func main() {
	fs := flag.NewFlagSet("dumphm", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长像素数")
	seed := fs.Uint64("seed", 7, "种子")
	octaves := fs.Int("octaves", 4, "分形层数")
	freq := fs.Float64("freq", 0.02, "基础频率")
	lac := fs.Float64("lacunarity", 2.0, "频率倍增")
	gain := fs.Float64("gain", 0.5, "振幅衰减")
	out := fs.String("out", "height.bin", "输出路径")
	if err := fs.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	h, err := api.Generate(api.Options{
		Size: *size, Octaves: *octaves, Lacunarity: *lac,
		Gain: *gain, Seed: *seed, Frequency: *freq,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		os.Exit(1)
	}
	f, err := os.Create(*out)
	if err != nil {
		fmt.Fprintln(os.Stderr, "create failed:", err)
		os.Exit(1)
	}
	defer f.Close()
	if _, err := f.Write([]byte("GOPROC-HM")); err != nil {
		os.Exit(1)
	}
	if err := binary.Write(f, binary.LittleEndian, uint32(h.Size)); err != nil {
		os.Exit(1)
	}
	if err := binary.Write(f, binary.LittleEndian, h.Data); err != nil {
		os.Exit(1)
	}
	fmt.Printf("wrote %s size=%d seed=%d octaves=%d\n", *out, h.Size, *seed, *octaves)
}
