package main

import (
	"flag"
	"fmt"
	"os"

	"goproc/api"
	"goproc/internal/lod"
)

// runLodCheck 在确定性地形上评估 LOD 二/三/四级接缝，
// 写出 metrics.json、table.md 与每级分别存档的拼接图。
func runLodCheck(args []string) int {
	fs := flag.NewFlagSet("lodcheck", flag.ContinueOnError)
	size := fs.Int("size", 128, "高度图边长（建议为 64 的倍数）")
	seed := fs.Uint64("seed", 7, "种子")
	octaves := fs.Int("octaves", 5, "分形层数")
	freq := fs.Float64("freq", 0.05, "基础频率")
	relief := fs.Float64("relief-m", 100, "归一化全高差对应的实地米数")
	outDir := fs.String("out", "artifacts/lod", "产物目录")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	h, err := api.Generate(api.Options{
		Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5,
		Seed: *seed, Frequency: *freq,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate failed:", err)
		return 1
	}

	ti := lod.TerrainInfo{
		Size: *size, Seed: *seed, Octaves: *octaves, Frequency: *freq,
		Lacunarity: 2, Gain: 0.5, ReliefM: *relief,
	}
	rep := lod.BuildReport(h.Data, h.Size, ti)

	if err := rep.WriteJSON(*outDir + "/metrics.json"); err != nil {
		fmt.Fprintln(os.Stderr, "write metrics failed:", err)
		return 1
	}
	if err := rep.WriteTable(*outDir + "/table.md"); err != nil {
		fmt.Fprintln(os.Stderr, "write table failed:", err)
		return 1
	}
	if err := lod.WriteImages(*outDir, h.Data, h.Size, rep); err != nil {
		fmt.Fprintln(os.Stderr, "write images failed:", err)
		return 1
	}

	fmt.Print(rep.TableMarkdown())
	for _, L := range rep.Levels {
		fmt.Printf("LOD%d: crack before=%.3f after=%.3f px | maxElevErr=%.6f (%.3f gray, %.3f m)\n",
			L.Level, L.CrackBeforePx, L.CrackAfterPx,
			L.MaxElevErr, L.MaxElevErrGray, L.MaxElevErrM)
	}
	fmt.Printf("artifacts written to %s/\n", *outDir)
	if !rep.AllZero {
		fmt.Fprintf(os.Stderr, "FAIL: 修复后仍有缝宽，最大 %.6g px\n", rep.MaxCrack)
		return 1
	}
	fmt.Println("PASS: 三个 LOD 层级修复后缝宽均为 0")
	return 0
}
