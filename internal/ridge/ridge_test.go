package ridge

import (
	"math"
	"testing"
)

func quadraticFrame(n int) Frame {
	f := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			f[y*n+x] = float64(x*x) + 0.5*float64(y*y)
		}
	}
	return Frame{N: n, F: f}
}

// 引理 1：对每个格点，任务书判据与 Δf<0 等价。
func TestMaskEqualsNegativeLaplacian(t *testing.T) {
	for _, n := range []int{16, 33, 64} {
		fr := quadraticFrame(n)
		f, err := NewField(fr)
		if err != nil {
			t.Fatal(err)
		}
		mask := f.Mask()
		lap := f.Laplacian()
		for i := range mask {
			if mask[i] != (lap[i] < 0) {
				t.Fatalf("n=%d i=%d mask=%v lap=%v", n, i, mask[i], lap[i])
			}
		}
	}
}

// 特征分解自检：L1<=L2，且 H·e1 = L1·e1。
func TestEigenpair(t *testing.T) {
	n := 24
	fr := Frame{N: n, F: make([]float64, n*n)}
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			X, Y := float64(x), float64(y)
			fr.F[y*n+x] = math.Sin(X*0.3)*math.Cos(Y*0.2) + 0.01*X*Y
		}
	}
	f, err := NewField(fr)
	if err != nil {
		t.Fatal(err)
	}
	for i := range f.F {
		if f.L1[i] > f.L2[i] {
			t.Fatalf("L1>L2 at %d", i)
		}
		// H·e1 - L1·e1
		ax := f.Fxx[i]*f.E1x[i] + f.Fxy[i]*f.E1y[i] - f.L1[i]*f.E1x[i]
		ay := f.Fxy[i]*f.E1x[i] + f.Fyy[i]*f.E1y[i] - f.L1[i]*f.E1y[i]
		if math.Hypot(ax, ay) > 1e-8 {
			t.Fatalf("e1 not eigenvector at %d: (%v,%v)", i, ax, ay)
		}
		if d := math.Hypot(f.E1x[i], f.E1y[i]); math.Abs(d-1) > 1e-12 {
			t.Fatalf("e1 not unit at %d: %v", i, d)
		}
	}
}

// 上升合并树计数：二元合并时 S=P-1，弧数=2P-2。
func TestJoinTreeCounts(t *testing.T) {
	n := 32
	fr := Frame{N: n, F: make([]float64, n*n)}
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			X, Y := float64(x)*0.37+0.013, float64(y)*0.23+0.007
			fr.F[y*n+x] = math.Sin(X)*math.Sin(Y) + 0.1*X + 0.023*Y
		}
	}
	f, _ := NewField(fr)
	jt := f.BuildJoinTree()
	if jt.TieCount != 0 {
		t.Fatalf("不应有等高点对, got %d", jt.TieCount)
	}
	if got, want := len(jt.Arcs), jt.P+len(jt.Saddles)-1; got != want {
		t.Fatalf("弧数=%d want P+S-1=%d", got, want)
	}
	for ar, c := range jt.Arities {
		if ar != 2 {
			t.Fatalf("非二元合并 arity=%d 次数=%d", ar, c)
		}
	}
	if len(jt.Saddles) != jt.P-1 {
		t.Fatalf("S=%d want P-1=%d", len(jt.Saddles), jt.P-1)
	}
	if len(jt.Arcs) != 2*jt.P-2 {
		t.Fatalf("弧数=%d want 2P-2=%d", len(jt.Arcs), 2*jt.P-2)
	}
	// P 应等于严格局部极大点数。
	if gm := len(f.LocalMaxima()); jt.P != gm {
		t.Fatalf("P=%d 但极大点数=%d", jt.P, gm)
	}
}

// 离散 Morse：Mask 段数 = Births - MergeLoss，且 Births 等于 Δf<0 的局部极小点数。
func TestMorseIdentity(t *testing.T) {
	for _, n := range []int{16, 32, 48} {
		nl := n
		fr := Frame{N: nl, F: make([]float64, nl*nl)}
		for y := 0; y < nl; y++ {
			for x := 0; x < nl; x++ {
				X, Y := float64(x)*0.35, float64(y)*0.31
				fr.F[y*nl+x] = math.Sin(X*1.7)*math.Cos(Y*1.3) + 0.05*math.Sin(5*X)*math.Cos(4*Y)
			}
		}
		f, _ := NewField(fr)
		mask := f.Mask()
		mc := f.ComputeMorse(mask)
		rep := f.Analyze("t", mask, f.LocalMaxima())
		if mc.Components != rep.Actual.Total {
			t.Fatalf("n=%d morse=%d actual=%d", n, mc.Components, rep.Actual.Total)
		}
		if got := mc.Births - mc.MergeLoss; got != mc.Components {
			t.Fatalf("n=%d births-merge=%d comps=%d", n, got, mc.Components)
		}
		if got, want := mc.Births, f.NegativeLaplacianMinima(); got != want {
			t.Fatalf("n=%d births=%d negLapMin=%d", n, got, want)
		}
	}
}

// 每个 Mask 分量含 Δf 的局部极小点。
func TestEveryIslandHasLaplacianMinimum(t *testing.T) {
	n := 48
	fr := Frame{N: n, F: make([]float64, n*n)}
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			X, Y := float64(x)*0.2, float64(y)*0.27
			fr.F[y*n+x] = math.Cos(X*2)*math.Sin(Y*1.5) + 0.2*X
		}
	}
	f, _ := NewField(fr)
	mask := f.Mask()
	lap := f.Laplacian()
	for _, is := range f.Islands(mask, f.LocalMaxima()) {
		minv := math.Inf(1)
		for k := range is.CellsX {
			if v := lap[is.CellsY[k]*n+is.CellsX[k]]; v < minv {
				minv = v
			}
		}
		found := false
		for k := range is.CellsX {
			x, y := is.CellsX[k], is.CellsY[k]
			if lap[y*n+x] != minv {
				continue
			}
			loc := true
			for dy := -1; dy <= 1 && loc; dy++ {
				for dx := -1; dx <= 1; dx++ {
					if dx == 0 && dy == 0 {
						continue
					}
					xx, yy := x+dx, y+dy
					if xx < 0 || yy < 0 || xx >= n || yy >= n {
						continue
					}
					if lap[yy*n+xx] < minv {
						loc = false
						break
					}
				}
			}
			if loc {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("island id=%d size=%d 不含 Δf 局部极小", is.ID, is.Size)
		}
	}
}
