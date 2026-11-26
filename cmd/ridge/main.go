// Command ridge 分析 goproc 高度图，提取脊线并与局部极值计数逐项比对。
//
// 用法：
//
//	go run ./cmd/ridge --seeds 7,42 --size 128 --octaves 4 --out artifacts
//
// 输出：
//   - artifacts/<tag>.txt   逐画幅报告（段数、端点坐标、多出/缺少段的位置）
//   - artifacts/<tag>.ppm   脊线叠加图（P6）
//   - artifacts/<tag>.pgm   仅 Mask 分量（P5）
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"goproc/api"
	"goproc/internal/ridge"
)

func main() {
	seeds := flag.String("seeds", "7,42", "逗号分隔的种子")
	size := flag.Int("size", 128, "边长")
	octaves := flag.Int("octaves", 4, "分形层数")
	freq := flag.Float64("freq", 0.02, "基础频率")
	outDir := flag.String("out", "artifacts", "输出目录")
	overlayN := flag.Int("overlay", 512, "叠加图边长")
	flag.Parse()

	var list []uint64
	for _, s := range strings.Split(*seeds, ",") {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		var v uint64
		if _, err := fmt.Sscan(s, &v); err != nil {
			fmt.Fprintf(os.Stderr, "bad seed %q: %v\n", s, err)
			os.Exit(2)
		}
		list = append(list, v)
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir:", err)
		os.Exit(1)
	}

	for _, seed := range list {
		h, err := api.Generate(api.Options{
			Size: *size, Octaves: *octaves, Lacunarity: 2, Gain: 0.5,
			Seed: seed, Frequency: *freq,
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, "generate:", err)
			os.Exit(1)
		}
		f, err := ridge.NewField(ridge.Frame{N: h.Size, F: h.Data})
		if err != nil {
			fmt.Fprintln(os.Stderr, "field:", err)
			os.Exit(1)
		}
		tag := fmt.Sprintf("seed%03d_n%d", seed, *size)
		text, err := runFrame(f, tag)
		if err != nil {
			fmt.Fprintln(os.Stderr, tag, err)
			os.Exit(1)
		}
		if err := os.WriteFile(filepath.Join(*outDir, tag+".txt"), []byte(text), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write txt:", err)
			os.Exit(1)
		}

		mask := f.Mask()
		crest := f.Extract(ridge.ModeCrest)
		masked := f.Extract(ridge.ModeMasked)
		if err := os.WriteFile(filepath.Join(*outDir, tag+".ppm"),
			ridge.EncodePPM(*overlayN, f.OverlayWithArcs(mask, crest, masked, *overlayN, f.BuildJoinTree())), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write ppm:", err)
			os.Exit(1)
		}
		if err := os.WriteFile(filepath.Join(*outDir, tag+"_mask.pgm"), pgmMask(f, mask), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write pgm:", err)
			os.Exit(1)
		}

		// 逐类叠加图：判据区域、高度图脊弧、两套 G=0 曲线各自单独存档，
		// 便于逐幅比对哪一层产生了多出/缺少的段。
		if err := os.WriteFile(filepath.Join(*outDir, tag+"_mask_layer.ppm"),
			ridge.EncodePPM(*overlayN, f.OverlayWithArcs(mask, ridge.Ridge{}, ridge.Ridge{}, *overlayN, ridge.JoinTree{Arities: map[int]int{}})), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write mask layer:", err)
			os.Exit(1)
		}
		if err := os.WriteFile(filepath.Join(*outDir, tag+"_crest_layer.ppm"),
			ridge.EncodePPM(*overlayN, f.OverlayWithArcs(nil, crest, ridge.Ridge{}, *overlayN, ridge.JoinTree{Arities: map[int]int{}})), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write crest layer:", err)
			os.Exit(1)
		}
		if err := os.WriteFile(filepath.Join(*outDir, tag+"_masked_layer.ppm"),
			ridge.EncodePPM(*overlayN, f.OverlayWithArcs(nil, ridge.Ridge{}, masked, *overlayN, ridge.JoinTree{Arities: map[int]int{}})), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "write masked layer:", err)
			os.Exit(1)
		}
		fmt.Printf("wrote %s.*\n", filepath.Join(*outDir, tag))
	}
}

func runFrame(f *ridge.Field, tag string) (string, error) {
	maxima := f.LocalMaxima()
	mask := f.Mask()
	rep := f.Analyze(tag, mask, maxima)
	crest := f.Extract(ridge.ModeCrest)
	masked := f.Extract(ridge.ModeMasked)
	jt := f.BuildJoinTree()

	var b strings.Builder
	fmt.Fprintf(&b, "画幅 %s  N=%d\n", tag, f.N)
	fmt.Fprintf(&b, "%s\n", rep.Line())
	fmt.Fprintf(&b, "任务书判据 Mask 面积 %d/%d（%.1f%%），与 Δf<0 逐格不一致数 = %d\n",
		countTrue(mask), f.N*f.N, 100*float64(countTrue(mask))/float64(f.N*f.N), maskLapDiff(f, mask))
	fmt.Fprintf(&b, "上升合并树：P=%d 鞍点=%d 弧=%d 二歧=%v 等高邻接=%d；P-1=%d，弧=P+S-1=%d\n",
		jt.P, len(jt.Saddles), len(jt.Arcs), jt.Arities, jt.TieCount, jt.P-1, jt.P+len(jt.Saddles)-1)
	fmt.Fprintf(&b, "预测段数（内部）= Δf 内部局部极小点数 %d；触边分量另计 %d\n",
		rep.Prediction.InteriorMinima, rep.Prediction.BoundaryIslands)
	fmt.Fprintf(&b, "精确段数分解（离散 Morse）：%s；Δf<0 局部极小点 %d，与出生数一致=%v\n",
		rep.MorseLine(), f.NegativeLaplacianMinima(), f.NegativeLaplacianMinima() == rep.Morse.Births)
	fmt.Fprintf(&b, "定理（命题 4）：每个内部段至少含一个 Δf 局部极小点；本轮未触边段 %d 个，\n",
		rep.Actual.Interior)
	fmt.Fprintf(&b, "Δf 内部局部极小点 %d 个，故上界 %d 成立=%v（该上界很松，不是对段数的精确预测）。\n",
		rep.Prediction.InteriorMinima, rep.Prediction.InteriorMinima,
		rep.Actual.Interior <= rep.Prediction.InteriorMinima)

	fmt.Fprintf(&b, "\n-- Mask 连通段清单（按大小降序）--\n")
	fmt.Fprintf(&b, "%4s %6s %-22s %-16s %-8s %s\n", "id", "cells", "bbox", "centroid", "触边", "内部极大点")
	isls := append([]ridge.IslandReport(nil), rep.Islands...)
	// 按大小降序显示，ID 仍是原始分量号。
	sortReportsBySize(isls)
	for _, is := range isls {
		pts := ""
		for _, m := range is.Maxima {
			pts += fmt.Sprintf("(%d,%d)%s", m.X, m.Y, " ")
		}
		if pts == "" {
			pts = "-"
		}
		fmt.Fprintf(&b, "%4d %6d [%3d,%3d..%3d,%3d]  (%6.1f,%6.1f)  %-6v %s\n",
			is.ID, is.Size, is.BBox[0], is.BBox[1], is.BBox[2], is.BBox[3],
			is.CentroidX, is.CentroidY, is.TouchesBorder, pts)
	}

	fmt.Fprintf(&b, "\n-- 预判段数与实际段数 --\n")
	fmt.Fprintf(&b, "按局部极大点预判：内部极大点 %d 个；实际 Mask 段总数 %d（内部 %d + 触边 %d）\n",
		rep.Actual.InteriorMaxima, rep.Actual.Total, rep.Actual.Interior, rep.Actual.Boundary)
	fmt.Fprintf(&b, "多出 %d 段 = 总段数 %d − 落在 Mask 内的内部极大点 %d。\n",
		rep.Slack, rep.Actual.Total, rep.Actual.MaximaInMask)
	fmt.Fprintf(&b, "其中不含内部极大点的段 %d 个，含极大点的段 %d 个；%d 个极大点被 %d 个段分摊，冗余 %d。\n",
		rep.NoMaxSegments, rep.WithMaxSegments, rep.Actual.MaximaInMask, rep.WithMaxSegments, rep.SharedSurplus)
	fmt.Fprintf(&b, "缺少 %d 段：内部极大点不在 Mask 内 %d 个。\n",
		rep.Actual.InteriorMaxima-rep.Actual.MaximaInMask,
		rep.Actual.InteriorMaxima-rep.Actual.MaximaInMask)

	fmt.Fprintf(&b, "\n-- 不含内部极大点的段（共 %d 个；它们解释了净多 %d 段中的一部分）--\n",
		len(f.ExtraMeta(rep, maxima)), rep.Slack)
	extra := f.ExtraMeta(rep, maxima)
	if len(extra) == 0 {
		fmt.Fprintln(&b, "无")
	}
	for _, e := range extra {
		fmt.Fprintf(&b, "id=%d cells=%d bbox=[%d,%d..%d,%d] centroid=(%.1f,%.1f) 触边=%v 最近极大点=%s 距离=%.1f\n",
			e.ID, e.Size, e.BBox[0], e.BBox[1], e.BBox[2], e.BBox[3],
			e.CentroidX, e.CentroidY, e.TouchesBorder, e.NearestMaxLabel, e.NearestMaxDist)
	}

	fmt.Fprintf(&b, "\n-- 缺少的段（内部极大点落在 Mask 外）--\n")
	if rep.Actual.InteriorMaxima == rep.Actual.MaximaInMask {
		fmt.Fprintln(&b, "无")
	}
	for _, m := range maxima {
		if m.X <= 0 || m.Y <= 0 || m.X >= f.N-1 || m.Y >= f.N-1 {
			continue
		}
		if mask[m.Y*f.N+m.X] {
			continue
		}
		id, dist := f.NearestIsland(rep, m.X, m.Y)
		fmt.Fprintf(&b, "极大点 (%d,%d) 值=%.6f 最近 Mask 段 id=%d 距离=%.1f 到脊线顶点距离=%.1f\n",
			m.X, m.Y, m.Value, id, dist, ridge.DistToRidge(crest, float64(m.X), float64(m.Y)))
	}

	fmt.Fprintf(&b, "\n-- 局部极大点坐标（%d 个，y,x）--\n", len(maxima))
	for _, m := range maxima {
		where := "Mask内"
		if !mask[m.Y*f.N+m.X] {
			where = "Mask外"
		}
		interior := m.X > 0 && m.Y > 0 && m.X < f.N-1 && m.Y < f.N-1
		kind := "内部"
		if !interior {
			kind = "边界"
		}
		fmt.Fprintf(&b, "(%3d,%3d) f=%.6f %s %s\n", m.X, m.Y, m.Value, kind, where)
	}

	fmt.Fprintf(&b, "\n-- 脊线（G=0 且 L1<0）--\n")
	full := f.ExtractFull()
	eps := crest.CrestEndpoints()
	fmt.Fprintf(&b, "marching squares 全部 G=0：pts=%d segs=%d curves=%d\n",
		len(full.Points), len(full.Segments), len(full.Curves))
	fmt.Fprintf(&b, "按 L1<0 裁剪后：curves=%d（开 %d，闭 %d），裁掉谷线段 %d 条\n",
		len(crest.Curves), countOpen(crest), len(crest.Curves)-countOpen(crest),
		len(full.Segments)-len(crest.Segments))
	fmt.Fprintf(&b, "按 Mask 裁剪后：curves=%d，segs=%d\n", len(masked.Curves), len(masked.Segments))
	fmt.Fprintf(&b, "开曲线端点（%d 个，按 y 升序）：\n", len(eps))
	for i, e := range eps {
		kind := "内部"
		if e.Y <= 0 || e.Y >= float64(f.N-1) {
			kind = "触上/下边"
		} else if e.X <= 0 || e.X >= float64(f.N-1) {
			kind = "触左/右边"
		}
		fmt.Fprintf(&b, "endpoint#%03d curve=%d (%.2f,%.2f) %s\n", i, e.CurveIndex, e.X, e.Y, kind)
	}

	fmt.Fprintf(&b, "\n-- 脊线统计 --\n")
	fmt.Fprintf(&b, "Mask 裁剪：curves=%d，segs=%d\n", len(masked.Curves), len(masked.Segments))
	fmt.Fprintf(&b, "L1<0 裁剪（脊）：curves=%d（开 %d，闭 %d）\n",
		len(crest.Curves), countOpen(crest), len(crest.Curves)-countOpen(crest))
	return b.String(), nil
}

func pgmMask(f *ridge.Field, mask []bool) []byte {
	out := make([]byte, len(mask))
	for i, b := range mask {
		if b {
			out[i] = 1
		}
	}
	// 这里不用 hmap.EncodePGM，避免它的 "\\n" 头缺陷。
	head := fmt.Sprintf("P5\n%d %d\n255\n", f.N, f.N)
	buf := make([]byte, 0, len(head)+len(out))
	buf = append(buf, head...)
	for _, v := range out {
		buf = append(buf, byte(v*255))
	}
	return buf
}

func countTrue(v []bool) int {
	n := 0
	for _, b := range v {
		if b {
			n++
		}
	}
	return n
}

func countOpen(r ridge.Ridge) int {
	n := 0
	for _, c := range r.Curves {
		if !c.Closed {
			n++
		}
	}
	return n
}

func maskLapDiff(f *ridge.Field, mask []bool) int {
	lap := f.Laplacian()
	n := 0
	for i := range mask {
		if mask[i] != (lap[i] < 0) {
			n++
		}
	}
	return n
}

func sortReportsBySize(v []ridge.IslandReport) {
	for i := 1; i < len(v); i++ {
		for j := i; j > 0 && v[j].Size > v[j-1].Size; j-- {
			v[j], v[j-1] = v[j-1], v[j]
		}
	}
}
