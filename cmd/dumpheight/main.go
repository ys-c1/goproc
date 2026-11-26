// dumpheight 把精确的归一化浮点高度场导出为二进制，供外部分析。
// 格式：小端 int64 尺寸 N，随后 N*N 个小端 float64，行优先。
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"os"

	"goproc/api"
)

func main() {
	fs := flag.NewFlagSet("dumpheight", flag.ContinueOnError)
	size := fs.Int("size", 128, "边长")
	seed := fs.Uint64("seed", 1, "种子")
	octaves := fs.Int("octaves", 4, "层数")
	out := fs.String("out", "height.bin", "输出文件")
	if err := fs.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	h, err := api.Generate(api.Options{Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5, Seed: *seed, Frequency: 0.02})
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
	if err := binary.Write(f, binary.LittleEndian, int64(h.Size)); err != nil {
		fmt.Fprintln(os.Stderr, "write size failed:", err)
		os.Exit(1)
	}
	if err := binary.Write(f, binary.LittleEndian, h.Data); err != nil {
		fmt.Fprintln(os.Stderr, "write data failed:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s size=%d cells=%d\n", *out, h.Size, h.Size*h.Size)
}
