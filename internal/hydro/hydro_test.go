package hydro

import (
	"math/rand"
	"testing"
)

func floats(rows ...[]float64) ([]float64, int) {
	size := len(rows)
	out := make([]float64, 0, size*size)
	for _, r := range rows {
		if len(r) != size {
			panic("非方阵")
		}
		out = append(out, r...)
	}
	return out, size
}

func TestTinyGridPlateauMinimum(t *testing.T) {
	h, size := floats(
		[]float64{3, 1, 3},
		[]float64{2, 1, 3},
		[]float64{3, 3, 3},
	)
	an := Analyze(size, h)
	want := []struct {
		level float64
		comp  int
	}{
		{1, 1}, {2, 1}, {3, 1},
	}
	for _, c := range want {
		if got := CountComponentsBFS(size, h, c.level); got != c.comp {
			t.Fatalf("BFS w=%v = %d, want %d", c.level, got, c.comp)
		}
		if got := an.ComponentsAt(c.level); got != c.comp {
			t.Fatalf("预测 w=%v = %d, want %d", c.level, got, c.comp)
		}
	}
	if an.Minima != 1 || an.Saddles != 0 {
		t.Fatalf("事件分类错误: minima=%d saddles=%d", an.Minima, an.Saddles)
	}
	// 同高对：高度 1 的竖直对 1 个 + 高度 3 的邻接 4 个 = 5。
	if an.EqualNeighborPairs != 5 {
		t.Fatalf("同高邻对数 = %d, want 5", an.EqualNeighborPairs)
	}
	if an.Events[0].Kind != KindMinimum || len(an.Events[0].Cells) != 2 {
		t.Fatalf("高度 1 应为双格台地极小，got %+v", an.Events[0])
	}
}

func TestSaddleGrid(t *testing.T) {
	// 四角与中心为极小（5 个）。鞍点结构受边界影响：
	// 上边中四邻为两个角+中心（K=3, Δ=-2）；左、右边中各接两个角（K=2, Δ=-1）；
	// 下边中只接到中心（K=1，普通点），因为其余邻格在网格外。
	// 这正是“网格外不算水域、不环绕”的直接后果。
	h, size := floats(
		[]float64{1, 3, 1},
		[]float64{3, 1, 3},
		[]float64{1, 3, 1},
	)
	an := Analyze(size, h)
	want := map[float64]int{1: 5, 3: 1}
	for w, c := range want {
		if got := CountComponentsBFS(size, h, w); got != c {
			t.Fatalf("BFS w=%v = %d, want %d", w, got, c)
		}
		if got := an.ComponentsAt(w); got != c {
			t.Fatalf("预测 w=%v = %d, want %d", w, got, c)
		}
	}
	if an.Minima != 5 {
		t.Fatalf("minima = %d, want 5", an.Minima)
	}
	if an.Saddles != 3 {
		t.Fatalf("saddles = %d, want 3", an.Saddles)
	}
	gotK := map[[2]int]int{}
	for _, e := range an.Events {
		gotK[[2]int{e.Y, e.X}] = e.K
	}
	for p, k := range map[[2]int]int{{0, 1}: 3, {1, 0}: 2, {1, 2}: 2, {2, 1}: 1} {
		if gotK[p] != k {
			t.Fatalf("格点 %v 的 K=%d, want %d", p, gotK[p], k)
		}
	}
	// 末态单一分量；恒等式 ΣΔ = minima − Σ(K−1) = 5 − (2+1+1) = 1。
	// 恒等式：ΣΔ = minima - Σ(K-1)，末态分量数为 1。
	if an.Records[len(an.Records)-1].Components != 1 {
		t.Fatal("全域淹没后应为单一分量")
	}
}

func TestPlateauSaddle(t *testing.T) {
	// 两个极小盆地由一条双格同高鞍部连通：朴素的逐格 1-K 会重复计鞍点。
	// 下方两行做成高地，不参与上方的盆地结构。
	h, size := floats(
		[]float64{1, 2, 2, 1},
		[]float64{3, 3, 3, 3},
		[]float64{4, 4, 4, 4},
		[]float64{5, 5, 5, 5},
	)
	an := Analyze(size, h)
	if an.Minima != 2 || an.Saddles != 1 {
		t.Fatalf("minima=%d saddles=%d, want 2,1", an.Minima, an.Saddles)
	}
	var saddle *Event
	for _, e := range an.Events {
		if e.Kind == KindSaddle {
			saddle = e
		}
	}
	if saddle == nil || len(saddle.Cells) != 2 || saddle.Delta != -1 || saddle.K != 2 {
		t.Fatalf("鞍部台地异常: %+v", saddle)
	}
	for _, w := range []float64{0.5, 1, 1.5, 2, 2.5, 3} {
		got, want := an.ComponentsAt(w), CountComponentsBFS(size, h, w)
		if got != want {
			t.Fatalf("w=%v 预测=%d BFS=%d", w, got, want)
		}
	}
	if c := an.ComponentsAt(0.0); c != 0 {
		t.Fatalf("低于最低水位应无水域, got %d", c)
	}
}

func TestBoundaryRing(t *testing.T) {
	// 边界环为水、中心为陆：四邻域下环自身连通，恰一个分量，网格外不参与。
	h, size := floats(
		[]float64{0, 0, 0},
		[]float64{0, 9, 0},
		[]float64{0, 0, 0},
	)
	if c := CountComponentsBFS(size, h, 1); c != 1 {
		t.Fatalf("边界环分量数 = %d, want 1", c)
	}
	if c := Analyze(size, h).ComponentsAt(1); c != 1 {
		t.Fatalf("边界环预测分量数 = %d, want 1", c)
	}
}

func TestFuzzAgainstBFS(t *testing.T) {
	rng := rand.New(rand.NewSource(20261127))
	for trial := 0; trial < 40; trial++ {
		size := 6 + rng.Intn(9)
		h := make([]float64, size*size)
		quantized := trial%2 == 0 // 一半试验注入大量同高台地
		bins := 1 + rng.Intn(5)
		for i := range h {
			if quantized {
				h[i] = float64(rng.Intn(bins + 2))
			} else {
				h[i] = rng.Float64()
			}
		}
		an := Analyze(size, h)

		// 逐档恒等式：记录的 Δ 必须等于本档事件 Δ 之和。
		running := 0
		for _, rec := range an.Records {
			sum := 0
			for _, e := range rec.Events {
				sum += e.Delta
			}
			if sum != rec.Delta {
				t.Fatalf("trial %d 高度 %v: 事件Δ和 %d != 记录Δ %d", trial, rec.Height, sum, rec.Delta)
			}
			running += rec.Delta
			if running != rec.Components {
				t.Fatalf("trial %d: 累计 %d != 分量 %d", trial, running, rec.Components)
			}
		}

		// 每个不同高度 + 档间中点，预测与独立 BFS 逐一比对。
		levels := make([]float64, 0, len(an.Records)*2+1)
		levels = append(levels, an.Records[0].Height-0.5)
		for k, rec := range an.Records {
			levels = append(levels, rec.Height)
			if k+1 < len(an.Records) {
				levels = append(levels, (rec.Height+an.Records[k+1].Height)/2)
			}
		}
		for _, w := range levels {
			got, want := an.ComponentsAt(w), CountComponentsBFS(size, h, w)
			if got != want {
				t.Fatalf("trial %d (quantized=%v) w=%v: 预测 %d != BFS %d",
					trial, quantized, w, got, want)
			}
		}
	}
}
