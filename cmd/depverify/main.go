// Command depverify 读取 goproc 导出的 PGM，用独立 Go 实现复核：
// 四邻域极小、minimax 溢流深度、限深填洼后逐格极小与预判的一一对应。
//
//	go run ./cmd/depverify --pgm artifacts/terrain_128_s7.pgm \
//	    --t1 20 --t2 40 --out artifacts/depression_go_verify.json
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sort"

	"goproc/internal/depress"
)

type posJSON struct {
	K          int     `json:"k"`
	X          int     `json:"x"`
	Y          int     `json:"y"`
	Gray       int     `json:"gray"`
	SpillDepth int     `json:"spillDepth"`
	MeasX      *int    `json:"measX"`
	MeasY      *int    `json:"measY"`
	DX         int     `json:"dx"`
	DY         int     `json:"dy"`
	Cheb       int     `json:"cheb"`
	Euclid     float64 `json:"euclid"`
	Over1      bool    `json:"over1"`
}

type rowJSON struct {
	Tau              int       `json:"tau"`
	NPredicted       int       `json:"nPredicted"`
	NMeasured        int       `json:"nMeasured"`
	NFlaggedOver1    int       `json:"nFlaggedOver1Cell"`
	MaxChebDeviation int       `json:"maxChebDeviation"`
	IdentityHolds    bool      `json:"identityHolds"`
	Positions        []posJSON `json:"positions"`
}

type reportJSON struct {
	PGM                     string    `json:"pgm"`
	Size                    int       `json:"size"`
	NMinima4                int       `json:"nMinima4"`
	CompleteFillResidual    int       `json:"completeFillResidualMinima"`
	AllThresholdsIdentityOK bool      `json:"allThresholdsIdentityOK"`
	Rows                    []rowJSON `json:"thresholdRows"`
}

func main() {
	pgm := flag.String("pgm", "artifacts/terrain_128_s7.pgm", "8-bit P5 PGM 路径")
	t1 := flag.Int("t1", 20, "第一档深度阈值（灰度级）")
	t2 := flag.Int("t2", 40, "第二档深度阈值（灰度级）")
	out := flag.String("out", "artifacts/depression_go_verify.json", "JSON 复核结果输出")
	flag.Parse()

	g, n, err := readP5Gray(*pgm)
	if err != nil {
		fmt.Fprintln(os.Stderr, "读取 PGM 失败:", err)
		os.Exit(2)
	}
	mins := depress.Minima4(g, n)
	spill := depress.Spill(g, n)
	completeResidual := depress.Minima4(spill, n)

	rep := reportJSON{PGM: *pgm, Size: n, NMinima4: len(mins),
		CompleteFillResidual: len(completeResidual), AllThresholdsIdentityOK: true}

	// 额外对 0..maxDepth 全阈值验证恒等式
	maxD := 0
	depthOf := map[depress.Point]int{}
	for _, m := range mins {
		d := depress.Depth(g, spill, m, n)
		depthOf[m] = d
		if d > maxD {
			maxD = d
		}
	}
	for tau := 0; tau <= maxD; tau++ {
		f := depress.FillLimited(g, spill, n, tau)
		got := depress.Minima4(f, n)
		var want []depress.Point
		for _, m := range mins {
			if depthOf[m] > tau {
				want = append(want, m)
			}
		}
		if !depress.SetEq(got, want) {
			rep.AllThresholdsIdentityOK = false
		}
	}

	for _, tau := range []int{*t1, *t2} {
		f := depress.FillLimited(g, spill, n, tau)
		meas := depress.Minima4(f, n)
		var pred []depress.Point
		for _, m := range mins {
			if depthOf[m] > tau {
				pred = append(pred, m)
			}
		}
		sort.Slice(pred, func(i, j int) bool {
			if depthOf[pred[i]] != depthOf[pred[j]] {
				return depthOf[pred[i]] > depthOf[pred[j]]
			}
			if pred[i].Y != pred[j].Y {
				return pred[i].Y < pred[j].Y
			}
			return pred[i].X < pred[j].X
		})
		pairs := depress.Compare(pred, meas)
		ident := depress.SetEq(pred, meas)
		nflag, maxCheb := 0, 0
		var ps []posJSON
		for i, pr := range pairs {
			cheb := pr.ChebCells
			if cheb < 0 {
				cheb = 99
			}
			if pr.OverOne {
				nflag++
			}
			if cheb > maxCheb {
				maxCheb = cheb
			}
			pj := posJSON{K: i + 1, X: pr.Pred.X, Y: pr.Pred.Y,
				Gray: g[pr.Pred.Y*n+pr.Pred.X], SpillDepth: depthOf[pr.Pred],
				DX: pr.DX, DY: pr.DY, Cheb: pr.ChebCells,
				Euclid: pr.Euclid, Over1: pr.OverOne}
			if pr.Meas != nil {
				mx, my := pr.Meas.X, pr.Meas.Y
				pj.MeasX, pj.MeasY = &mx, &my
			}
			ps = append(ps, pj)
		}
		row := rowJSON{Tau: tau, NPredicted: len(pred), NMeasured: len(meas),
			NFlaggedOver1: nflag, MaxChebDeviation: maxCheb,
			IdentityHolds: ident && nflag == 0, Positions: ps}
		if !row.IdentityHolds {
			rep.AllThresholdsIdentityOK = false
		}
		rep.Rows = append(rep.Rows, row)
		fmt.Printf("tau=%3d pred=%d meas=%d flagged>1=%d maxCheb=%d identity=%v\n",
			tau, len(pred), len(meas), nflag, maxCheb, row.IdentityHolds)
		for _, p := range ps {
			mx, my := "-", "-"
			if p.MeasX != nil && p.MeasY != nil {
				mx, my = fmt.Sprintf("%d", *p.MeasX), fmt.Sprintf("%d", *p.MeasY)
			}
			fmt.Printf("   #%d (%d,%d) G=%d d=%d meas=(%s,%s) cheb=%d\n",
				p.K, p.X, p.Y, p.Gray, p.SpillDepth, mx, my, p.Cheb)
		}
	}

	b, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		panic(err)
	}
	if err := os.WriteFile(*out, b, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "写出失败:", err)
		os.Exit(2)
	}
	fmt.Println("wrote", *out)
	if !rep.AllThresholdsIdentityOK {
		fmt.Fprintln(os.Stderr, "恒等式复核失败")
		os.Exit(1)
	}
}

// readP5Gray 解析 8-bit P5 PGM（容忍 # 注释），返回一维灰度与边长。
func readP5Gray(path string) ([]int, int, error) {
	buf, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	pos := 0
	skipWS := func() {
		for pos < len(buf) {
			c := buf[pos]
			if c == '#' {
				for pos < len(buf) && buf[pos] != '\n' {
					pos++
				}
				continue
			}
			if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
				pos++
				continue
			}
			break
		}
	}
	readTok := func() (string, error) {
		skipWS()
		st := pos
		for pos < len(buf) {
			c := buf[pos]
			if c == '#' || c == ' ' || c == '\t' || c == '\n' || c == '\r' {
				break
			}
			pos++
		}
		if st == pos {
			return "", fmt.Errorf("PGM 头部不完整")
		}
		return string(buf[st:pos]), nil
	}
	magic, err := readTok()
	if err != nil {
		return nil, 0, err
	}
	if magic != "P5" {
		return nil, 0, fmt.Errorf("只支持 P5，得到 %q", magic)
	}
	wt, err := readTok()
	if err != nil {
		return nil, 0, err
	}
	ht, err := readTok()
	if err != nil {
		return nil, 0, err
	}
	mv, err := readTok()
	if err != nil {
		return nil, 0, err
	}
	var w, h, maxv int
	if _, err := fmt.Sscanf(wt, "%d", &w); err != nil {
		return nil, 0, err
	}
	if _, err := fmt.Sscanf(ht, "%d", &h); err != nil {
		return nil, 0, err
	}
	if _, err := fmt.Sscanf(mv, "%d", &maxv); err != nil {
		return nil, 0, err
	}
	if maxv != 255 || w != h {
		return nil, 0, fmt.Errorf("期望 8-bit 方形 PGM，得到 %dx%d max=%d", w, h, maxv)
	}
	// 头后恰好一个空白分隔符
	if pos < len(buf) && (buf[pos] == ' ' || buf[pos] == '\t' ||
		buf[pos] == '\n' || buf[pos] == '\r') {
		pos++
	}
	need := w * h
	if len(buf)-pos < need {
		return nil, 0, fmt.Errorf("像素数据不足: 需 %d 剩 %d", need, len(buf)-pos)
	}
	g := make([]int, need)
	for i := 0; i < need; i++ {
		g[i] = int(buf[pos+i])
	}
	return g, w, nil
}
