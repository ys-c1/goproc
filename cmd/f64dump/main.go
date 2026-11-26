// Command f64dump 调用仓库自己的确定性生成器，导出无损高度场：
// 小端 uint64 边长 N，随后 N*N 个小端 float64（行优先，已归一化到 0..1）。
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"os"

	"goproc/api"
)

func main() {
	size := flag.Int("size", 1024, "边长")
	seed := flag.Uint64("seed", 7, "种子")
	octaves := flag.Int("octaves", 5, "分形层数")
	freq := flag.Float64("freq", 0.02, "基础频率")
	gain := flag.Float64("gain", 0.5, "振幅增益")
	lac := flag.Float64("lacunarity", 2.0, "频率倍增")
	out := flag.String("out", "out.f64", "输出 .f64 路径")
	flag.Parse()

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

	var nb [8]byte
	binary.LittleEndian.PutUint64(nb[:], uint64(*size))
	if _, err := f.Write(nb[:]); err != nil {
		panic(err)
	}
	if err := binary.Write(f, binary.LittleEndian, h.Data); err != nil {
		panic(err)
	}
	fmt.Printf("wrote %s n=%d octaves=%d seed=%d\n", *out, *size, *octaves, *seed)
}
