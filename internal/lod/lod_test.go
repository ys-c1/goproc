package lod

import (
	"math"
	"testing"
)

func newTestGrid(t *testing.T, n int) *Grid {
	t.Helper()
	h := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			h[y*n+x] = math.Sin(float64(x)*0.15) * math.Cos(float64(y)*0.11)
		}
	}
	g, err := NewGrid(h, n, 1, 1)
	if err != nil {
		t.Fatal(err)
	}
	return g
}

func TestEffectiveVerticesExcludesUnreferenced(t *testing.T) {
	// 4 个池顶点，但只有 0,1,2 被一个三角形引用，3 是简化后遗留的孤儿。
	indices := []uint32{0, 1, 2}
	if got := EffectiveVertices(indices); got != 3 {
		t.Fatalf("有效顶点应为 3，实际 %d", got)
	}
	if got := EffectiveVerticesSorted(indices); got != 3 {
		t.Fatalf("排序实现应为 3，实际 %d", got)
	}

	// 重复引用同一批顶点不能重复计数。
	indices = []uint32{0, 1, 2, 0, 2, 3, 9, 9, 1}
	if got := EffectiveVertices(indices); got != 5 { // 0,1,2,3,9
		t.Fatalf("有效顶点应为 5，实际 %d", got)
	}
	if EffectiveVertices(indices) != EffectiveVerticesSorted(indices) {
		t.Fatal("两种有效顶点计数实现结果不一致")
	}
}

func TestBuildLevelEffectiveCounts(t *testing.T) {
	const n = 257 // W=256
	g := newTestGrid(t, n)
	wantSide := []int{257, 129, 65, 33, 17, 9, 5, 3, 2}
	for l := 0; l <= g.MaxLevel(); l++ {
		lv, err := g.BuildLevel(l)
		if err != nil {
			t.Fatalf("level %d: %v", l, err)
		}
		if lv.SideVerts != wantSide[l] {
			t.Fatalf("level %d SideVerts=%d want %d", l, lv.SideVerts, wantSide[l])
		}
		wantEff := wantSide[l] * wantSide[l]
		if got := lv.Effective(); got != wantEff {
			t.Fatalf("level %d Effective=%d want %d", l, got, wantEff)
		}
		if EffectiveVerticesSorted(lv.Indices) != wantEff {
			t.Fatalf("level %d 排序计数与期望不符", l)
		}
		if lv.Triangles() != 2*(wantSide[l]-1)*(wantSide[l]-1) {
			t.Fatalf("level %d 三角形数错误: %d", l, lv.Triangles())
		}
		for _, id := range lv.Indices {
			if int(id) < 0 || int(id) >= n*n {
				t.Fatalf("level %d 索引越界: %d", l, id)
			}
		}
	}
}

func TestSpacingSquareLaw(t *testing.T) {
	// 相邻级：格距翻倍 -> 预判顶点比 4；跨 d 级：4^d，只依赖层级差。
	for d := 1; d <= 4; d++ {
		if got := SpacingRatio(2, 2+d); got != math.Pow(2, float64(d)) {
			t.Fatalf("d=%d spacing ratio %v", d, got)
		}
		if got := PredictedRatio(0, d); got != math.Pow(4, float64(d)) {
			t.Fatalf("d=%d predicted ratio %v", d, got)
		}
		// 与绝对层级无关
		if PredictedRatio(3, 3+d) != PredictedRatio(0, d) {
			t.Fatalf("比例不应依赖绝对层级")
		}
	}
}

func TestExactRatioMatchesActual(t *testing.T) {
	const w = 256
	g := newTestGrid(t, w+1)
	for i := 0; i <= 6; i++ {
		li, _ := g.BuildLevel(i)
		for j := i + 1; j <= g.MaxLevel(); j++ {
			lj, _ := g.BuildLevel(j)
			actual := float64(li.Effective()) / float64(lj.Effective())
			exact := ExactGridRatio(w, i, j)
			if math.Abs(actual-exact) > 1e-12 {
				t.Fatalf("pair %d->%d actual %.12f exact %.12f", i, j, actual, exact)
			}
		}
	}
}

func TestLevelZeroHasNoError(t *testing.T) {
	g := newTestGrid(t, 65)
	lv, err := g.BuildLevel(0)
	if err != nil {
		t.Fatal(err)
	}
	if lv.ErrMax != 0 || lv.ErrRMS != 0 {
		t.Fatalf("0 级误差应为零, got max=%g rms=%g", lv.ErrMax, lv.ErrRMS)
	}
}

func TestSelectLevel(t *testing.T) {
	errs := []float64{0, 0.01, 0.05, 0.2, 0.5}
	if got := SelectLevel(errs, 0.04); got != 1 {
		t.Fatalf("阈值 0.04 应选 1 级, got %d", got)
	}
	if got := SelectLevel(errs, 0.05); got != 2 {
		t.Fatalf("阈值 0.05 应选 2 级, got %d", got)
	}
	if got := SelectLevel(errs, 10); got != 4 {
		t.Fatalf("大阈值应选最粗级, got %d", got)
	}
}
