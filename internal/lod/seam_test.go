package lod

import (
	"math"
	"testing"
)

func TestSpacingLevels234(t *testing.T) {
	want := map[int]int{2: 4, 3: 8, 4: 16}
	for k, w := range want {
		if got := Spacing(k); got != w {
			t.Fatalf("Spacing(%d)=%d, want %d", k, got, w)
		}
	}
}

func TestDivisibility(t *testing.T) {
	cases := []struct {
		coord, d int
		want     bool
	}{
		{64, 4, true}, {64, 8, true}, {64, 16, true},
		{63, 4, false}, {63, 8, false}, {63, 16, false},
		{128, 16, true}, {48, 16, true}, {49, 16, false},
	}
	for _, c := range cases {
		if got := Divisible(c.coord, c.d); got != c.want {
			t.Fatalf("Divisible(%d,%d)=%v want %v", c.coord, c.d, got, c.want)
		}
	}
	// 公共原点下，粗级顶点全部落在细级网格线等价于 H%h==0。
	if !CoarseSubsetOfFine(4, 16) {
		t.Fatal("16 应是 4 的整数倍")
	}
	if CoarseSubsetOfFine(3, 4) {
		t.Fatal("4 不是 3 的整数倍，不应判定为包含")
	}
}

func TestRemainderMinusOneBoundary(t *testing.T) {
	// Y=63=64-1，而 4/8/16 都整除 64，故余数恰为 d-1。
	for _, d := range []int{4, 8, 16} {
		if r := Remainder(63, d); r != d-1 {
			t.Fatalf("Remainder(63,%d)=%d want %d", d, r, d-1)
		}
	}
}

func TestInsetGapCoefficientLaw(t *testing.T) {
	for _, h := range []int{1, 2, 4} {
		for _, H := range []int{4, 8, 16} {
			if H <= h {
				continue
			}
			for _, rho := range []float64{0.5, 1.0} {
				pred := InsetGap(h, H, rho)
				a, b := InsetBoundaries(0, 128, 127, h, H, rho)
				meas := Hausdorff(a, b)
				if math.Abs(meas-pred) > 1e-9 {
					t.Fatalf("w rho=%v h=%d H=%d: measured %.12f != predicted %.12f",
						rho, h, H, meas, pred)
				}
				// 比例系数必须正是 rho。
				if c := meas / float64(H-h); math.Abs(c-rho) > 1e-12 {
					t.Fatalf("coefficient=%v want rho=%v", c, rho)
				}
			}
		}
	}
	// 两个点名系数。
	if InsetGap(1, 16, 1) != 15 {
		t.Fatal("rho=1 时系数 1，w=|H-h|")
	}
	if InsetGap(1, 16, 0.5) != 7.5 {
		t.Fatal("rho=1/2 时系数 1/2，w=|H-h|/2")
	}
}

func TestNaiveCrackWidthsLevels234(t *testing.T) {
	want := map[int]float64{2: 3, 3: 7, 4: 15} // 全分辨率细级 vs LOD-k
	for k, w := range want {
		H := Spacing(k)
		f, c := NaiveBoundaries(0, 127, 63, 1, H)
		if got := Hausdorff(f, c); math.Abs(got-w) > 1e-12 {
			t.Fatalf("LOD%d before crack=%v want %v", k, got, w)
		}
	}
	// 相邻链 (k-1,k) 的宽度 = H-h：2,4,8。
	for k, w := range map[int]float64{2: 2, 3: 4, 4: 8} {
		h, H := Spacing(k-1), Spacing(k)
		f, c := NaiveBoundaries(0, 127, 63, h, H)
		if got := Hausdorff(f, c); math.Abs(got-w) > 1e-12 {
			t.Fatalf("adjacent LOD%d before crack=%v want %v", k, got, w)
		}
	}
}

func TestStitchedCrackExactlyZero(t *testing.T) {
	for k := 2; k <= 4; k++ {
		H := Spacing(k)
		for _, h := range []int{1, Spacing(k - 1)} {
			f, c := StitchedBoundaries(0, 127, 63, h, H)
			if got := Hausdorff(f, c); got != 0 {
				t.Fatalf("LOD%d h=%d after crack=%v want exactly 0", k, h, got)
			}
			if len(f) != len(c) {
				t.Fatalf("两侧公共顶点数不一致: %d vs %d", len(f), len(c))
			}
		}
	}
}

func TestBilinearStrideOneExact(t *testing.T) {
	n := 16
	h := make([]float64, n*n)
	for i := range h {
		h[i] = float64(i%n) / float64(n-1)
	}
	if e, _, _ := MaxError(h, n, 1); e != 0 {
		t.Fatalf("d=1 双线性重建应逐点精确，最大误差 %v", e)
	}
}

func TestBuildReportThreeLevelsZeroAndErrors(t *testing.T) {
	n := 128
	h := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			h[y*n+x] = 0.5 + 0.5*math.Sin(float64(x)*0.3)*math.Cos(float64(y)*0.27)
		}
	}
	rep := BuildReport(h, n, TerrainInfo{Size: n, ReliefM: 100})
	if len(rep.Levels) != 3 {
		t.Fatalf("要有三个层级， got %d", len(rep.Levels))
	}
	if !rep.AllZero || rep.MaxCrack != 0 {
		t.Fatalf("修复后缝宽必须全零: allZero=%v max=%v", rep.AllZero, rep.MaxCrack)
	}
	wantStride := []int{4, 8, 16}
	for i, L := range rep.Levels {
		if L.Level != i+2 || L.CoarseStride != wantStride[i] {
			t.Fatalf("行顺序/格距错误: %+v", L)
		}
		if L.CrackAfterPx != 0 {
			t.Fatalf("LOD%d after=%v", L.Level, L.CrackAfterPx)
		}
		if L.CoefficientC != 1 {
			t.Fatalf("LOD%d 主序列系数应为 1, got %v", L.Level, L.CoefficientC)
		}
		if L.MaxElevErr < 0 || math.IsNaN(L.MaxElevErr) || math.IsInf(L.MaxElevErr, 0) {
			t.Fatalf("LOD%d 高程误差非法: %v", L.Level, L.MaxElevErr)
		}
	}
	// 定律扫描所有点都应与解析公式一致。
	for _, p := range rep.LawSweep {
		if p.AbsError > 1e-9 {
			t.Fatalf("定律扫描失配: %+v", p)
		}
	}
}
