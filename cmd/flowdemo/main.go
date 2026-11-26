// flowdemo 生成两个画幅的高度图，按 D8 规则一次性预判全图流向，
// 再用不依赖预判结果的逐格追踪做逐格比对，报告不一致坐标，并把
// 流向图、出口清单与证明文档存档到输出目录。
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"time"

	"goproc/api"
	"goproc/internal/flow"
)

type frameSpec struct {
	name    string
	options api.Options
}

func main() {
	fs := flag.NewFlagSet("flowdemo", flag.ContinueOnError)
	outDir := fs.String("out", "artifacts/flow", "输出目录")
	sizeA := fs.Int("size-a", 128, "画幅 A 边长")
	seedA := fs.Uint64("seed-a", 7, "画幅 A 种子")
	sizeB := fs.Int("size-b", 96, "画幅 B 边长")
	seedB := fs.Uint64("seed-b", 23, "画幅 B 种子")
	if err := fs.Parse(os.Args[1:]); err != nil {
		os.Exit(2)
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir failed:", err)
		os.Exit(1)
	}

	frames := []frameSpec{
		{name: "A", options: api.Options{Size: *sizeA, Octaves: 4, Lacunarity: 2, Gain: 0.5, Seed: *seedA, Frequency: 0.02}},
		{name: "B", options: api.Options{Size: *sizeB, Octaves: 5, Lacunarity: 2, Gain: 0.5, Seed: *seedB, Frequency: 0.03}},
	}

	var report []byte
	add := func(format string, args ...any) {
		report = append(report, fmt.Sprintf(format, args...)...)
	}

	add("goproc 流向预判 vs 逐格追踪 比对报告\n")
	add("生成时间: %s\n", time.Now().Format(time.RFC3339))
	add("离散规则: D8 八邻域最陡下坡；坡度=落差/距离(斜距 sqrt2)；并列取方向编号最小者(E,SE,S,SW,W,NW,N,NE)；无严格下坡记 Sink\n")
	add("终点定义: 追踪首达边界格=出口(TermOutlet)，到达 Sink=内陆汇(TermSink)\n")
	add("注意: 判定与比对均在原始 float64 高度上进行；高度 PGM 经 8 位量化会改变近并列坡度(实测约 9%%-11%% 格翻转)，复算请用随档 .f64 文件\n\n")

	allMatch := true
	for _, fr := range frames {
		h, err := api.Generate(fr.options)
		if err != nil {
			fmt.Fprintln(os.Stderr, "generate failed:", err)
			os.Exit(1)
		}
		size := h.Size
		add("================ 画幅 %s: size=%d seed=%d octaves=%d freq=%v ================\n",
			fr.name, size, fr.options.Seed, fr.options.Octaves, fr.options.Frequency)

		// 1) 一次性预判全图流向
		t0 := time.Now()
		dirMap := flow.Build(h.Data, size)
		buildDur := time.Since(t0)

		// 2) 基于预判流向一次性结算每格终点
		t0 = time.Now()
		predicted := dirMap.ResolveAll()
		resolveDur := time.Since(t0)

		// 3) 参照：逐格重算流向并追踪
		t0 = time.Now()
		naive := make([]flow.Terminus, size*size)
		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				naive[y*size+x] = flow.NaiveTrace(h.Data, size, x, y)
			}
		}
		naiveDur := time.Since(t0)

		// 4) 逐格比对
		type mismatch struct{ x, y int }
		var mismatches []mismatch
		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				id := y*size + x
				if predicted[id] != naive[id] {
					mismatches = append(mismatches, mismatch{x, y})
				}
			}
		}
		if len(mismatches) != 0 {
			allMatch = false
		}

		flats := flow.FlatCells(h.Data, size)
		sinks := dirMap.Sinks()
		outlets := flow.Outlets(predicted, size)

		add("格子总数: %d\n", size*size)
		add("耗时: 预判流向=%v  预判结算终点=%v  逐格追踪=%v\n", buildDur, resolveDur, naiveDur)
		add("零坡度格(无严格下坡，判 Sink，单独处理): %d\n", len(flats))
		add("  其中内陆汇(非边界 Sink): %d\n", len(sinks))
		add("  其中边界零坡度格: %d\n", len(flats)-len(sinks))
		add("汇流出口数: %d\n", len(outlets))
		add("逐格比对: %d 格，不一致 %d 格\n", size*size, len(mismatches))
		if len(mismatches) > 0 {
			add("不一致坐标 (x,y)，最多列 100 个:\n")
			limit := len(mismatches)
			if limit > 100 {
				limit = 100
			}
			for i := 0; i < limit; i++ {
				mc := mismatches[i]
				id := mc.y*size + mc.x
				add("  (%d,%d) 预判={x:%d y:%d kind:%d} 追踪={x:%d y:%d kind:%d}\n",
					mc.x, mc.y,
					predicted[id].X, predicted[id].Y, predicted[id].Kind,
					naive[id].X, naive[id].Y, naive[id].Kind)
			}
		} else {
			add("全部格子终点完全一致。\n")
		}

		// 出口与内陆汇逐行列出（按 y 再按 x）
		add("出口位置逐行列出 (kind=1 出口):\n")
		listByRow(add, outlets, size, flow.TermOutlet)
		add("内陆汇位置逐行列出 (kind=0 汇):\n")
		listByRow(add, toCells(sinks), size, flow.TermSink)
		add("\n")

		// 存档：流向图 PGM + 高度图 PGM
		flowPath := filepath.Join(*outDir, fmt.Sprintf("flow_%s_s%03d_seed%d.pgm", fr.name, size, fr.options.Seed))
		if err := os.WriteFile(flowPath, dirMap.EncodePGM(), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write flow pgm failed:", err)
			os.Exit(1)
		}
		hPath := filepath.Join(*outDir, fmt.Sprintf("height_%s_s%03d_seed%d.pgm", fr.name, size, fr.options.Seed))
		if err := os.WriteFile(hPath, h.ToPGM(), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write height pgm failed:", err)
			os.Exit(1)
		}
		// 原始 float64 高度（小端、行优先），供外部工具不含量化地独立复算流向。
		f64Path := filepath.Join(*outDir, fmt.Sprintf("height_%s_s%03d_seed%d.f64", fr.name, size, fr.options.Seed))
		if err := writeFloat64(f64Path, h.Data); err != nil {
			fmt.Fprintln(os.Stderr, "write float dump failed:", err)
			os.Exit(1)
		}
		add("存档: %s\n", flowPath)
		add("存档: %s\n", hPath)
		add("存档: %s (float64 LE, %d 个值, 供无损复算)\n\n", f64Path, len(h.Data))
	}

	if allMatch {
		add("结论: 两个画幅均逐格一致，一次性预判与逐格追踪等价得到验证。\n")
	} else {
		add("结论: 存在不一致格子，见上方坐标。\n")
	}

	reportPath := filepath.Join(*outDir, "report.txt")
	if err := os.WriteFile(reportPath, report, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write report failed:", err)
		os.Exit(1)
	}
	fmt.Print(string(report))
	fmt.Println("报告已存档:", reportPath)
}

func toCells(cs []flow.Cell) []flow.Cell { return cs }

// listByRow 按行 (y) 分组、行内按 x 排序后逐行输出坐标。
func listByRow(add func(string, ...any), cells []flow.Cell, size int, kind flow.TerminusKind) {
	sorted := make([]flow.Cell, len(cells))
	copy(sorted, cells)
	sort.Slice(sorted, func(i, j int) bool {
		if sorted[i].Y != sorted[j].Y {
			return sorted[i].Y < sorted[j].Y
		}
		return sorted[i].X < sorted[j].X
	})
	byRow := make(map[int][]flow.Cell)
	var rows []int
	for _, c := range sorted {
		if _, ok := byRow[c.Y]; !ok {
			rows = append(rows, c.Y)
		}
		byRow[c.Y] = append(byRow[c.Y], c)
	}
	sort.Ints(rows)
	add("  kind=%d, 共 %d 个, 分布在 %d 行:\n", kind, len(cells), len(rows))
	for _, y := range rows {
		line := fmt.Sprintf("  y=%3d:", y)
		for _, c := range byRow[y] {
			line += fmt.Sprintf(" (%d,%d)", c.X, c.Y)
		}
		add("%s\n", line)
	}
	if len(rows) == 0 {
		add("  (无)\n")
	}
	_ = size
}

// writeFloat64 以小端二进制行优先落盘原始高度，避免 PGM 八位量化
// 影响外部对 D8 规则的独立复算。
func writeFloat64(path string, v []float64) error {
	buf := make([]byte, 8*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(x))
	}
	return os.WriteFile(path, buf, 0o644)
}
