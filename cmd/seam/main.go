// seam 命令构造两块地形，分别在左右、上下两个方向拼接，
// 度量对齐前后的接缝高度差，并输出带接缝标注的拼接图。
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"goproc/internal/seam"
)

type runResult struct {
	dir           seam.Direction
	slots         int // 边界实际访问的去重置换表槽位数
	diffEntries   int // 对齐前两表在这些槽位上取值不同的数量
	copied        int // 对齐时从 A 表复制进 B 表的条目数
	before, after seam.Metrics
}

func main() {
	fs := flag.NewFlagSet("seam", flag.ContinueOnError)
	size := fs.Int("size", 128, "单块地形边长像素数")
	seedA := fs.Uint64("seed-a", 11, "A 块种子")
	seedB := fs.Uint64("seed-b", 47, "B 块种子（对齐前）")
	outDir := fs.String("outdir", "artifacts/seam", "输出目录")
	if err := fs.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	if *seedA == *seedB {
		fmt.Fprintln(os.Stderr, "seed-a 与 seed-b 必须不同，否则边界条目批次相同，无法展示对齐前的非零接缝差")
		os.Exit(2)
	}

	cfg := seam.DefaultConfig()
	cfg.N = *size
	cfg.SeedA = *seedA
	cfg.SeedB = *seedB
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir failed:", err)
		os.Exit(1)
	}

	// 先把两个方向、对齐前后的五块地形全部生成出来，统一确定灰度尺度。
	tileA := seam.Generate(cfg, cfg.SeedA, 0, 0)
	type pair struct{ pre, post *seam.Tile }
	tilesByDir := map[seam.Direction]pair{}
	var allTiles = []*seam.Tile{tileA}
	var meta = map[seam.Direction]runResult{}

	for _, dir := range []seam.Direction{seam.Horizontal, seam.Vertical} {
		bx, by := seam.EdgeOffset(cfg, dir)
		bPre := seam.Generate(cfg, cfg.SeedB, bx, by)
		bPost := seam.Generate(cfg, cfg.SeedB, bx, by) // 与 bPre 同源，对齐只改边界槽位条目
		slots, diff := seam.BoundaryEntryDiff(cfg, tileA, bPre, dir)
		_, copied := seam.Align(cfg, tileA, bPost, dir)
		if copied != slots {
			fmt.Fprintf(os.Stderr, "内部错误：复制条目数 %d 与边界槽位数 %d 不一致\n", copied, slots)
			os.Exit(1)
		}
		meta[dir] = runResult{dir: dir, slots: slots, diffEntries: diff, copied: copied}
		tilesByDir[dir] = pair{bPre, bPost}
		allTiles = append(allTiles, bPre, bPost)
	}
	scale := seam.NewScale(allTiles...)

	var results []runResult
	for _, dir := range []seam.Direction{seam.Horizontal, seam.Vertical} {
		p := tilesByDir[dir]
		before := seam.Stitch(tileA, p.pre, dir, scale)
		after := seam.Stitch(tileA, p.post, dir, scale)
		r := meta[dir]
		r.before = before.Measure()
		r.after = after.Measure()
		results = append(results, r)
		if err := writePair(*outDir, dir, before, after); err != nil {
			fmt.Fprintln(os.Stderr, "write failed:", err)
			os.Exit(1)
		}
	}

	report := buildReport(cfg, results)
	fmt.Print(report)
	if err := os.WriteFile(filepath.Join(*outDir, "metrics.txt"), []byte(report), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write metrics failed:", err)
		os.Exit(1)
	}
}

func writePair(outDir string, dir seam.Direction, before, after *seam.Composite) error {
	name := dir.String()
	for _, item := range []struct {
		stage string
		comp  *seam.Composite
	}{
		{"before", before},
		{"after", after},
	} {
		base := name + "_" + item.stage
		pf, err := os.Create(filepath.Join(outDir, base+"_marked.png"))
		if err != nil {
			return err
		}
		if err := item.comp.WriteMarkedPNG(pf); err != nil {
			pf.Close()
			return err
		}
		pf.Close()
		gf, err := os.Create(filepath.Join(outDir, base+".pgm"))
		if err != nil {
			return err
		}
		if err := item.comp.WritePGM(gf); err != nil {
			gf.Close()
			return err
		}
		gf.Close()
	}
	return nil
}

func buildReport(cfg seam.Config, results []runResult) string {
	var b strings.Builder
	fmt.Fprintf(&b, "接缝高度差实验  size=%d octaves=%d lacunarity=%.1f gain=%.2f frequency=%.4f seedA=%d seedB=%d\n",
		cfg.N, cfg.Octaves, cfg.Lacunarity, cfg.Gain, cfg.Frequency, cfg.SeedA, cfg.SeedB)
	fmt.Fprintf(&b, "接缝差定义：接缝两侧相邻像素灰度差的最大值（竖拼按同列；横拼按同行，旋转90度后等价于同列）\n")
	fmt.Fprintf(&b, "边界槽位：共享边界全部像素与全部八度实际访问的置换表去重槽位；条目不一致数为零即两表取同一批条目\n\n")
	header := fmt.Sprintf("%-6s %-8s %-10s %-12s %-12s %-12s %-12s %-14s %-14s %-12s %-12s\n",
		"方向", "阶段", "像素对数", "最大灰度差", "平均灰度差", "非零对数", "旋转后最大差", "原始最大差",
		"边界槽位数", "条目不一致", "对齐条目数")
	b.WriteString(header)
	for _, r := range results {
		writeRow := func(stage string, m seam.Metrics, slots, diff, copied int) {
			fmt.Fprintf(&b, "%-6s %-8s %-10d %-12d %-12.4f %-12d %-12d %-14.3e %-14d %-12d %-12d\n",
				r.dir.String(), stage, m.PairCount, m.MaxGrayDiff, m.MeanGrayDiff,
				m.NonZeroPairs, m.RotatedMaxGray, m.MaxRawDiff, slots, diff, copied)
		}
		writeRow("对齐前", r.before, r.slots, r.diffEntries, 0)
		writeRow("对齐后", r.after, r.slots, 0, r.copied)
	}
	return b.String()
}
