package gully

import (
	"testing"

	"goproc/api"
	"goproc/internal/hydro"
)

func TestConjunctionStrictBothThresholds(t *testing.T) {
	// 直接构造水文场，验证合取与严格不等号。
	n := 2
	f := &hydro.Field{N: n, Acc: []int{33, 100, 40, 33}, Slope: []float64{0.04, 0.02, 0.04, 0.03}}
	th := Thresholds{Acc: 32, Slope: 0.03}
	m := CoreMask(f, th)
	// i0: Acc33>32 且 Slope0.04>0.03 -> true
	// i1: 坡度不足 -> false
	// i2: 两条件都过 -> true
	// i3: 坡度恰好等于阈值，严格 > 为假 -> false
	want := []bool{true, false, true, false}
	for i := range want {
		if m[i] != want[i] {
			t.Fatalf("格 %d 合取判定应为 %v，实得 %v", i, want[i], m[i])
		}
	}
}

func TestComponents8DiagonalCountsAsOne(t *testing.T) {
	n := 3
	mask := []bool{
		true, false, false,
		false, true, false,
		false, false, false,
	}
	comp, segs := Components8(mask, n)
	if len(segs) != 1 {
		t.Fatalf("对角相接在八邻域下应为 1 段，实得 %d 段", len(segs))
	}
	if comp[0] != comp[4] || comp[0] == 0 {
		t.Fatal("两个对角格段号应相同且非 0")
	}

	// 同样两格改成同行被一格隔开：八邻域也不相连 -> 2 段。
	mask2 := []bool{
		true, false, true,
		false, false, false,
		false, false, false,
	}
	_, segs2 := Components8(mask2, n)
	if len(segs2) != 2 {
		t.Fatalf("相距两格应为 2 段，实得 %d", len(segs2))
	}
}

func TestComponents8DiagonalBridgesClusters(t *testing.T) {
	n := 3
	mask := []bool{
		true, true, false,
		true, false, true,
		false, false, true,
	}
	_, segs := Components8(mask, n)
	// (0,0)(1,0)(0,1) 与 (2,1)(2,2) 看似两簇，但 (1,0) 与 (2,1) 对角相邻，
	// 八邻域下对角即连通，故全部 5 格同属 1 段。
	if len(segs) != 1 {
		t.Fatalf("对角桥接后应为 1 段，实得 %d", len(segs))
	}
	if segs[0].Count != 5 {
		t.Fatalf("唯一段应含 5 格，实得 %d", segs[0].Count)
	}
}

func TestDilate8RadiusOne(t *testing.T) {
	n := 3
	mask := make([]bool, n*n)
	mask[4] = true
	z := Dilate8(mask, n, 1)
	cnt := 0
	for _, v := range z {
		if v {
			cnt++
		}
	}
	if cnt != 9 {
		t.Fatalf("中心格 r1 膨胀应覆盖全 3x3=9 格，实得 %d", cnt)
	}
}

// TestCanonicalEndToEnd 是题目的核心机器验证：
// 在真实确定性地形上，两组强度都必须做到
// 预判核心 == 侵蚀后逐格实测核心，且所有侵蚀格都在核心 r1 八邻域内。
func TestCanonicalEndToEnd(t *testing.T) {
	h, err := api.Generate(api.Options{Size: 128, Octaves: 4, Lacunarity: 2, Gain: 0.5, Seed: 7, Frequency: 0.02})
	if err != nil {
		t.Fatal(err)
	}
	f := hydro.Build(h.Data, h.Size)
	th := CanonicalThresholds()
	pred := CoreMask(f, th)
	predComp, predSegs := Components8(pred, h.Size)
	predCount := len(predSegs)
	if predCount == 0 {
		t.Fatal("基准参数下不应为 0 段")
	}

	for _, p := range IntensityGroups() {
		res := Erode(h.Data, f, th, pred, p)
		meas := MeasuredCore(res.Cut, DCore)
		d := Compare(pred, meas)
		if !d.Match() {
			t.Fatalf("%s: 逐格不一致 missing=%v extra=%v", p.Name, d.Missing, d.Extra)
		}
		measComp, measSegs := Components8(meas, h.Size)
		if len(measSegs) != predCount {
			t.Fatalf("%s: 段数预判 %d 实测 %d", p.Name, predCount, len(measSegs))
		}
		// 预判段号划分与实测段号划分必须逐格相同（掩膜相等且扫描规则一致）。
		for i := range predComp {
			if predComp[i] != measComp[i] {
				t.Fatalf("%s: 格 %d 段号不一致 pred=%d meas=%d", p.Name, i, predComp[i], measComp[i])
			}
		}
		_, viol := NearCheck(pred, res.Cut, h.Size, 1, EpsEroded)
		if len(viol) != 0 {
			t.Fatalf("%s: %d 个侵蚀格超出核心 r1 八邻域: %v", p.Name, len(viol), viol[:min(5, len(viol))])
		}
		if res.Clipped != 0 {
			t.Fatalf("%s: 出现 %d 个撞底截断格，参数区间假设被破坏", p.Name, res.Clipped)
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
