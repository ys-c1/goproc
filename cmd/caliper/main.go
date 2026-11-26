// Command caliper 对两个画幅分别用「凸包 + 旋转卡壳」预判最远点对，
// 并与对全部候选点暴力遍历的实测结果逐画幅比对，输出报告、汇总表与连线图。
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"goproc/api"
	"goproc/internal/caliper"
)

// 连线图的图层灰度：地形压到 0..114（保持明暗可读），其余图层用互不
// 相同的更高灰度，脚本可按灰度还原图层。
const (
	hullLayer  = 150 // 凸包边
	lineLayer  = 235 // 最远点连线
	pointLayer = 255 // 最远点端点（5×5）
	terrainMax = 114 // 地形背景上限
)

type frame struct {
	Name      string
	Size      int
	Seed      uint64
	Octaves   int
	Threshold float64
}

type side struct {
	A     caliper.Point `json:"a"`
	B     caliper.Point `json:"b"`
	Dist2 int64         `json:"dist2"`
	Dist  int           `json:"dist_px"`
	Ties  int64         `json:"ties"`
}

type stats struct {
	Frame       string          `json:"frame"`
	Size        int             `json:"size"`
	Seed        uint64          `json:"seed"`
	Octaves     int             `json:"octaves"`
	Threshold   float64         `json:"threshold_height"`
	Candidates  int             `json:"candidates"`
	Hull        []caliper.Point `json:"hull"`
	HullVerts   int             `json:"hull_vertices"`
	Pairs       []caliper.Pair  `json:"antipodal_pairs"`
	PairDist2   []int64         `json:"antipodal_pair_dist2"`
	Antipodal   int             `json:"antipodal_count"`
	Rounds      int             `json:"caliper_rounds"`
	Steps       int             `json:"caliper_cursor_steps"`
	Predicted   side            `json:"predicted"`
	Measured    side            `json:"measured"`
	MatchDist2  bool            `json:"match_dist2"`
	MatchRound  bool            `json:"match_rounded"`
	MismatchAt  string          `json:"mismatch_coords"`
	GenMillis   int64           `json:"generate_ms"`
	PreMillis   int64           `json:"predict_ms"`
	BruteMillis int64           `json:"brute_ms"`
	BrutePairs  int64           `json:"brute_pair_count"`
}

func toSide(r caliper.Result) side {
	return side{A: r.A, B: r.B, Dist2: r.Dist2, Dist: r.DistPx(), Ties: r.Ties}
}

func main() {
	out := flag.String("out", "artifacts/caliper", "输出目录")
	thr := flag.Float64("threshold", 0.5, "候选点阈值（归一化高度，等价灰度 128）")
	oct := flag.Int("octaves", 4, "分形层数")
	aSize := flag.Int("a-size", 128, "画幅 A 边长")
	aSeed := flag.Uint64("a-seed", 7, "画幅 A 种子")
	bSize := flag.Int("b-size", 192, "画幅 B 边长")
	bSeed := flag.Uint64("b-seed", 9, "画幅 B 种子")
	flag.Parse()

	frames := []frame{
		{Name: "A", Size: *aSize, Seed: *aSeed, Octaves: *oct, Threshold: *thr},
		{Name: "B", Size: *bSize, Seed: *bSeed, Octaves: *oct, Threshold: *thr},
	}
	if err := os.MkdirAll(*out, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir failed:", err)
		os.Exit(1)
	}

	all := make([]stats, 0, len(frames))
	for _, f := range frames {
		st, err := run(f, *out)
		if err != nil {
			fmt.Fprintf(os.Stderr, "画幅 %s 失败: %v\n", f.Name, err)
			os.Exit(1)
		}
		all = append(all, st)
	}

	if err := writeJSON(filepath.Join(*out, "summary.json"), all); err != nil {
		fmt.Fprintln(os.Stderr, "写 summary.json 失败:", err)
		os.Exit(1)
	}
	if err := writeCSV(filepath.Join(*out, "summary.csv"), all); err != nil {
		fmt.Fprintln(os.Stderr, "写 summary.csv 失败:", err)
		os.Exit(1)
	}
	txt, err := renderReport(all)
	if err != nil {
		fmt.Fprintln(os.Stderr, "生成报告失败:", err)
		os.Exit(1)
	}
	if err := os.WriteFile(filepath.Join(*out, "report.txt"), []byte(txt), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "写 report.txt 失败:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s/{report.txt,summary.csv,summary.json,link_*.pgm,terrain_*.pgm}\n", *out)
}

func run(f frame, out string) (stats, error) {
	opt := api.DefaultOptions()
	opt.Size = f.Size
	opt.Seed = f.Seed
	opt.Octaves = f.Octaves

	t0 := time.Now()
	hm, err := api.Generate(opt)
	if err != nil {
		return stats{}, err
	}
	tGen := time.Since(t0)

	pts := hm.Above(f.Threshold)

	t0 = time.Now()
	pred, hull, pairs, rounds, steps := caliper.Farthest(pts)
	tPred := time.Since(t0)

	t0 = time.Now()
	meas := caliper.Brute(pts)
	tBrute := time.Since(t0)

	pairDist := make([]int64, len(pairs))
	for i, p := range pairs {
		pairDist[i] = caliper.Dist2(hull[p.I], hull[p.J])
	}
	st := stats{
		Frame:       f.Name,
		Size:        f.Size,
		Seed:        f.Seed,
		Octaves:     f.Octaves,
		Threshold:   f.Threshold,
		Candidates:  len(pts),
		Hull:        hull,
		HullVerts:   len(hull),
		Pairs:       pairs,
		PairDist2:   pairDist,
		Antipodal:   len(pairs),
		Rounds:      rounds,
		Steps:       steps,
		Predicted:   toSide(pred),
		Measured:    toSide(meas),
		MatchDist2:  pred.Dist2 == meas.Dist2,
		MatchRound:  pred.DistPx() == meas.DistPx(),
		GenMillis:   tGen.Milliseconds(),
		PreMillis:   tPred.Milliseconds(),
		BruteMillis: tBrute.Milliseconds(),
		BrutePairs:  int64(len(pts)) * int64(len(pts)-1) / 2,
	}
	st.MismatchAt = "—"
	if !st.MatchRound {
		st.MismatchAt = fmt.Sprintf("预判(%d,%d)-(%d,%d) / 实测(%d,%d)-(%d,%d)",
			pred.A.X, pred.A.Y, pred.B.X, pred.B.Y, meas.A.X, meas.A.Y, meas.B.X, meas.B.Y)
	}

	base := filepath.Join(out, "terrain_"+f.Name)
	if err := os.WriteFile(base+".pgm", hm.ToPGM(), 0o644); err != nil {
		return st, err
	}
	if err := os.WriteFile(filepath.Join(out, "link_"+f.Name+".pgm"), overlay(hm, hull, pred), 0o644); err != nil {
		return st, err
	}
	return st, nil
}

// overlay 生成连线图：地形压暗作背景，凸包边与最远点连线依次覆盖。
//
// 灰度分层（脚本按此还原）：0..114 地形背景；150 凸包边；
// 235 最远点连线；255 最远点端点（5×5）。图层灰度不与地形区间重叠。
func overlay(hm *api.Heightmap, hull []caliper.Point, far caliper.Result) []byte {
	n := hm.Size
	img := make([]byte, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			img[y*n+x] = byte(int(hm.Gray(x, y)) * terrainMax / 255)
		}
	}
	put := func(x, y int, v byte) {
		if x < 0 || y < 0 || x >= n || y >= n {
			return
		}
		img[y*n+x] = v
	}
	for i := range hull {
		a := hull[i]
		b := hull[(i+1)%len(hull)]
		line(a, b, func(x, y int) { put(x, y, hullLayer) })
	}
	if far.OK {
		line(far.A, far.B, func(x, y int) { put(x, y, lineLayer) })
		for _, p := range []caliper.Point{far.A, far.B} {
			for dy := -2; dy <= 2; dy++ {
				for dx := -2; dx <= 2; dx++ {
					put(p.X+dx, p.Y+dy, pointLayer)
				}
			}
		}
	}
	head := fmt.Sprintf("P5\n%d %d\n255\n", n, n)
	out := make([]byte, 0, len(head)+len(img))
	out = append(out, head...)
	out = append(out, img...)
	return out
}

// line 用 Bresenham 画整数格线段。
func line(a, b caliper.Point, put func(x, y int)) {
	x0, y0 := a.X, a.Y
	x1, y1 := b.X, b.Y
	dx := abs(x1 - x0)
	dy := -abs(y1 - y0)
	sx, sy := 1, 1
	if x0 > x1 {
		sx = -1
	}
	if y0 > y1 {
		sy = -1
	}
	err := dx + dy
	for {
		put(x0, y0)
		if x0 == x1 && y0 == y1 {
			return
		}
		e2 := 2 * err
		if e2 >= dy {
			err += dy
			x0 += sx
		}
		if e2 <= dx {
			err += dx
			y0 += sy
		}
	}
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

func writeJSON(path string, all []stats) error {
	b, err := json.MarshalIndent(all, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	return os.WriteFile(path, b, 0o644)
}

func writeCSV(path string, all []stats) error {
	fh, err := os.Create(path)
	if err != nil {
		return err
	}
	defer fh.Close()
	w := csv.NewWriter(fh)
	defer w.Flush()
	head := []string{
		"frame", "size", "seed", "octaves", "threshold",
		"candidates", "hull_vertices", "caliper_rounds", "cursor_steps", "antipodal_pairs",
		"pred_ax", "pred_ay", "pred_bx", "pred_by", "pred_dist2", "pred_px", "pred_ties",
		"meas_ax", "meas_ay", "meas_bx", "meas_by", "meas_dist2", "meas_px", "meas_ties",
		"match_dist2", "match_px", "mismatch_coords",
	}
	if err := w.Write(head); err != nil {
		return err
	}
	for _, s := range all {
		rec := []string{
			s.Frame, itoa(s.Size), fmt.Sprint(s.Seed), itoa(s.Octaves), ftoa(s.Threshold),
			itoa(s.Candidates), itoa(s.HullVerts), itoa(s.Rounds), itoa(s.Steps), itoa(s.Antipodal),
			itoa(s.Predicted.A.X), itoa(s.Predicted.A.Y), itoa(s.Predicted.B.X), itoa(s.Predicted.B.Y),
			fmt.Sprint(s.Predicted.Dist2), itoa(s.Predicted.Dist), fmt.Sprint(s.Predicted.Ties),
			itoa(s.Measured.A.X), itoa(s.Measured.A.Y), itoa(s.Measured.B.X), itoa(s.Measured.B.Y),
			fmt.Sprint(s.Measured.Dist2), itoa(s.Measured.Dist), fmt.Sprint(s.Measured.Ties),
			bool2(s.MatchDist2), bool2(s.MatchRound), s.MismatchAt,
		}
		if err := w.Write(rec); err != nil {
			return err
		}
	}
	return nil
}

func itoa(v int) string     { return fmt.Sprint(v) }
func ftoa(v float64) string { return fmt.Sprintf("%g", v) }
func bool2(v bool) string {
	if v {
		return "true"
	}
	return "false"
}

func renderReport(all []stats) (string, error) {
	if len(all) != 2 {
		return "", fmt.Errorf("需要恰好两个画幅，实际 %d", len(all))
	}
	var b []byte
	add := func(format string, a ...any) {
		b = append(b, []byte(fmt.Sprintf(format, a...))...)
	}
	add("地形最远点对：凸包 + 旋转卡壳预判 vs 暴力遍历实测\n")
	add("生成时间: %s\n", time.Now().Format(time.RFC3339))
	add("候选点定义: 归一化高度 >= 阈值 %g（阈值 0.5 恰等价 8 位灰度 >= 128）的整数格点\n",
		all[0].Threshold)
	add("距离单位: 像素；坐标: 整数格 (x,y)；距离取整到像素\n")
	add("卡壳轮数 = 每条凸包边处理一轮；游标推进步数 = 卡壳内层游标总前进次数（上界 2h）\n")
	add("\n")

	add("================ 两画幅并排 ================\n")
	add("%-28s %14s %14s\n", "项目", "画幅 A", "画幅 B")
	row := func(label, a, bb string) {
		add("%-28s %14s %14s\n", label, a, bb)
	}
	rowStr := func(label string, get func(stats) string) {
		row(label, get(all[0]), get(all[1]))
	}
	rowStr("画幅尺寸", func(s stats) string { return fmt.Sprintf("%d×%d", s.Size, s.Size) })
	rowStr("种子", func(s stats) string { return fmt.Sprint(s.Seed) })
	rowStr("分形层数", func(s stats) string { return itoa(s.Octaves) })
	rowStr("候选点数", func(s stats) string { return itoa(s.Candidates) })
	rowStr("凸包顶点数", func(s stats) string { return itoa(s.HullVerts) })
	rowStr("卡壳轮数", func(s stats) string { return itoa(s.Rounds) })
	rowStr("游标推进步数", func(s stats) string { return itoa(s.Steps) })
	rowStr("对踵点对数", func(s stats) string { return itoa(s.Antipodal) })
	rowStr("预判距离(像素)", func(s stats) string { return itoa(s.Predicted.Dist) })
	rowStr("预判点 A 坐标", func(s stats) string { return fmt.Sprintf("(%d,%d)", s.Predicted.A.X, s.Predicted.A.Y) })
	rowStr("预判点 B 坐标", func(s stats) string { return fmt.Sprintf("(%d,%d)", s.Predicted.B.X, s.Predicted.B.Y) })
	rowStr("实测距离(像素)", func(s stats) string { return itoa(s.Measured.Dist) })
	rowStr("实测点 A 坐标", func(s stats) string { return fmt.Sprintf("(%d,%d)", s.Measured.A.X, s.Measured.A.Y) })
	rowStr("实测点 B 坐标", func(s stats) string { return fmt.Sprintf("(%d,%d)", s.Measured.B.X, s.Measured.B.Y) })
	rowStr("取整距离一致", func(s stats) string { return yesno(s.MatchRound) })
	rowStr("精确平方距离一致", func(s stats) string { return yesno(s.MatchDist2) })
	rowStr("不一致时两点坐标", func(s stats) string { return s.MismatchAt })
	rowStr("最远距离并列数", func(s stats) string { return fmt.Sprint(s.Predicted.Ties) })
	add("\n")

	add("================ 对踵点对逐对（距离用整数平方距离） ================\n")
	for _, s := range all {
		add("画幅 %s：共 %d 对\n", s.Frame, s.Antipodal)
		for k, p := range s.Pairs {
			a, bb := s.Hull[p.I], s.Hull[p.J]
			d2 := s.PairDist2[k]
			add("  V%d-V%d  (%d,%d)-(%d,%d)  d2=%d  d=%d px\n",
				p.I, p.J, a.X, a.Y, bb.X, bb.Y, d2, caliper.RoundPx(d2))
		}
		add("\n")
	}

	for _, s := range all {
		add("================ 画幅 %s 明细 ================\n", s.Frame)
		add("参数: size=%d seed=%d octaves=%d threshold=%g\n", s.Size, s.Seed, s.Octaves, s.Threshold)
		add("候选 %d 点，凸包 %d 顶点，对踵点对 %d 对，卡壳 %d 轮，游标推进 %d 步\n",
			s.Candidates, s.HullVerts, s.Antipodal, s.Rounds, s.Steps)
		add("预判: (%d,%d)-(%d,%d) d2=%d 距离=%d px（并列 %d 对）\n",
			s.Predicted.A.X, s.Predicted.A.Y, s.Predicted.B.X, s.Predicted.B.Y,
			s.Predicted.Dist2, s.Predicted.Dist, s.Predicted.Ties)
		add("实测: (%d,%d)-(%d,%d) d2=%d 距离=%d px（并列 %d 对）\n",
			s.Measured.A.X, s.Measured.A.Y, s.Measured.B.X, s.Measured.B.Y,
			s.Measured.Dist2, s.Measured.Dist, s.Measured.Ties)
		add("耗时: 预判 %d ms，暴力 %d ms，生成 %d ms（候选 %d 点，暴力遍历 %d 对）\n",
			s.PreMillis, s.BruteMillis, s.GenMillis, s.Candidates, s.BrutePairs)
		add("一致性: 平方距离 %s；取整距离 %s", yesno(s.MatchDist2), yesno(s.MatchRound))
		if !s.MatchRound {
			add("；不一致坐标 %s", s.MismatchAt)
		}
		add("\n")
		add("凸包顶点（逆时针，图像坐标）:\n")
		for i, p := range s.Hull {
			add("  V%d = (%d,%d)\n", i, p.X, p.Y)
		}
		add("\n")
	}
	return string(b), nil
}

func yesno(v bool) string {
	if v {
		return "是"
	}
	return "否"
}
