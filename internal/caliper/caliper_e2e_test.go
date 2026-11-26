// Package caliper 的端到端测试：地形生成 → 候选点 → 卡壳预判 → 暴力实测，
// 两条路径必须逐画幅一致，并检查最远点确实落在凸包上、出现在对踵点对里。
package caliper_test

import (
	"testing"

	"goproc/api"
	"goproc/internal/caliper"
)

// frames 是 CLI 默认的两个画幅与一组画幅档位。用较小画幅让暴力枚举保持秒级。
var frames = []struct {
	name      string
	size      int
	seed      uint64
	octaves   int
	threshold float64
}{
	{"A", 128, 7, 4, 0.5},
	{"B", 192, 9, 4, 0.5},
	{"small", 64, 3, 4, 0.5},
	{"dense", 96, 11, 6, 0.5},
}

func candidates(t *testing.T, size int, seed uint64, octaves int, threshold float64) []caliper.Point {
	t.Helper()
	opt := api.DefaultOptions()
	opt.Size = size
	opt.Seed = seed
	opt.Octaves = octaves
	hm, err := api.Generate(opt)
	if err != nil {
		t.Fatalf("生成失败: %v", err)
	}
	return hm.Above(threshold)
}

// TestEndToEndPredictionMatchesBrute 预判与暴力必须给出相同的最远点对和距离。
func TestEndToEndPredictionMatchesBrute(t *testing.T) {
	for _, f := range frames {
		f := f
		t.Run(f.name, func(t *testing.T) {
			pts := candidates(t, f.size, f.seed, f.octaves, f.threshold)
			if len(pts) < 2 {
				t.Fatalf("候选点不足: %d", len(pts))
			}
			pred, hull, pairs, rounds, steps := caliper.Farthest(pts)
			meas := caliper.Brute(pts)

			if !pred.OK || !meas.OK {
				t.Fatalf("结果缺失")
			}
			if pred.Dist2 != meas.Dist2 {
				t.Fatalf("平方距离不一致: 预判 %d (%v,%v) 实测 %d (%v,%v)",
					pred.Dist2, pred.A, pred.B, meas.Dist2, meas.A, meas.B)
			}
			if pred.DistPx() != meas.DistPx() {
				t.Fatalf("取整距离不一致: %d != %d", pred.DistPx(), meas.DistPx())
			}
			if pred.A != meas.A || pred.B != meas.B {
				t.Fatalf("并列裁决后坐标不一致: 预判 (%v,%v) 实测 (%v,%v)",
					pred.A, pred.B, meas.A, meas.B)
			}
			if len(hull) < 2 {
				t.Fatalf("凸包顶点不足: %d", len(hull))
			}
			if rounds != len(hull) {
				t.Fatalf("卡壳轮数 %d != 凸包顶点数 %d", rounds, len(hull))
			}
			if steps > 2*len(hull) {
				t.Fatalf("游标步数 %d 超过 2h=%d", steps, 2*len(hull))
			}
			// 最远点必须都是凸包顶点。
			hv := make(map[caliper.Point]bool, len(hull))
			for _, p := range hull {
				hv[p] = true
			}
			if !hv[pred.A] || !hv[pred.B] {
				t.Fatalf("最远点不在凸包顶点上: %v %v", pred.A, pred.B)
			}
			// 预判找到的最远点对必须出现在对踵点对集合里。
			found := false
			for _, p := range pairs {
				a, b := hull[p.I], hull[p.J]
				if (a == pred.A && b == pred.B) || (a == pred.B && b == pred.A) {
					found = true
					break
				}
			}
			if !found {
				t.Fatalf("最远点对不在卡壳给出的对踵点对里")
			}
			t.Logf("画幅 %s: 候选 %d 凸包 %d 轮 %d 步 %d 对踵 %d 距离 %d px 点 (%d,%d)-(%d,%d)",
				f.name, len(pts), len(hull), rounds, steps, len(pairs),
				pred.DistPx(), pred.A.X, pred.A.Y, pred.B.X, pred.B.Y)
		})
	}
}

// TestCandidatesAreExactlyAboveThreshold 候选点必须与阈值筛选逐格一致。
func TestCandidatesAreExactlyAboveThreshold(t *testing.T) {
	opt := api.DefaultOptions()
	opt.Size = 64
	hm, err := api.Generate(opt)
	if err != nil {
		t.Fatal(err)
	}
	for _, thr := range []float64{0.4, 0.5, 0.6} {
		pts := hm.Above(thr)
		got := make(map[caliper.Point]bool, len(pts))
		for _, p := range pts {
			got[p] = true
		}
		for y := 0; y < hm.Size; y++ {
			for x := 0; x < hm.Size; x++ {
				want := hm.At(x, y) >= thr
				if got[caliper.Point{X: x, Y: y}] != want {
					t.Fatalf("threshold=%g 格 (%d,%d) 高度 %v 归类错误", thr, x, y, hm.At(x, y))
				}
			}
		}
	}
}

// TestAboveSorted 候选点按 (x,y) 字典序升序，便于复算脚本逐项对齐。
func TestAboveSorted(t *testing.T) {
	opt := api.DefaultOptions()
	opt.Size = 48
	hm, err := api.Generate(opt)
	if err != nil {
		t.Fatal(err)
	}
	pts := hm.Above(0.5)
	for i := 1; i < len(pts); i++ {
		if caliper.Less(pts[i], pts[i-1]) {
			t.Fatalf("候选点未按字典序排列: %v 在 %v 之后", pts[i], pts[i-1])
		}
	}
}
