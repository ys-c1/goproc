package depress

import (
	"testing"
)

func bowl5() ([]int, int) {
	// 5x5：边界 0，一圈 7，中央 2。中央是唯一四邻域严格极小，
	// minimax 溢流高程 7，洼地深度 Δ=5。
	n := 5
	g := []int{
		0, 0, 0, 0, 0,
		0, 7, 7, 7, 0,
		0, 7, 2, 7, 0,
		0, 7, 7, 7, 0,
		0, 0, 0, 0, 0,
	}
	return g, n
}

func TestMinima4StrictAndBorder(t *testing.T) {
	g, n := bowl5()
	m := Minima4(g, n)
	if len(m) != 1 || m[0] != (Point{2, 2}) {
		t.Fatalf("碗中央应为唯一严格极小 (2,2)，得到 %v", m)
	}

	// 对角相邻的两个极小都应被计入（四邻域定义下它们不相邻）。
	d := []int{
		9, 9, 9, 9, 9,
		9, 1, 8, 9, 9,
		9, 8, 1, 9, 9,
		9, 9, 9, 9, 9,
		9, 9, 9, 9, 9,
	}
	mm := Minima4(d, 5)
	if len(mm) != 2 {
		t.Fatalf("对角双极小应计 2 个，得到 %v", mm)
	}
}

func TestSpillDepthBowl(t *testing.T) {
	g, n := bowl5()
	s := Spill(g, n)
	if got := s[2*n+2]; got != 7 {
		t.Fatalf("中央溢流高程应为 7，得到 %d", got)
	}
	if dep := Depth(g, s, Point{2, 2}, n); dep != 5 {
		t.Fatalf("中央洼地深度应为 5 灰度级，得到 %d", dep)
	}
	if r := Minima4(s, n); len(r) != 0 {
		t.Fatalf("完整填洼面不应再有严格极小，得到 %v", r)
	}
}

func TestThresholdIdentityBowl(t *testing.T) {
	g, n := bowl5()
	s := Spill(g, n)
	m := Minima4(g, n)
	for tau := 0; tau <= 8; tau++ {
		f := FillLimited(g, s, n, tau)
		got := Minima4(f, n)
		var want []Point
		for _, p := range m {
			if Depth(g, s, p, n) > tau {
				want = append(want, p)
			}
		}
		if !SetEq(got, want) {
			t.Fatalf("τ=%d: 填后极小 %v 与预判 %v 不一致", tau, got, want)
		}
	}
}

// lcgGrid 生成确定性的 0..255 网格，避免测试依赖 math/rand 全局状态。
func lcgGrid(n int, seed uint32) []int {
	g := make([]int, n*n)
	x := seed
	for i := range g {
		x = x*1664525 + 1013904223
		g[i] = int((x >> 16) & 255)
	}
	return g
}

func TestIdentityAllTausRandomGrids(t *testing.T) {
	for _, n := range []int{12, 20, 33} {
		for _, seed := range []uint32{1, 7, 42} {
			g := lcgGrid(n, seed)
			s := Spill(g, n)
			m := Minima4(g, n)
			maxD := 0
			for _, p := range m {
				if d := Depth(g, s, p, n); d > maxD {
					maxD = d
				}
			}
			// 完整填洼面无内部严格极小
			if r := Minima4(s, n); len(r) != 0 {
				t.Fatalf("n=%d seed=%d 完整填洼后仍有 %d 个极小", n, seed, len(r))
			}
			for tau := 0; tau <= maxD+1; tau++ {
				f := FillLimited(g, s, n, tau)
				got := Minima4(f, n)
				var want []Point
				for _, p := range m {
					if Depth(g, s, p, n) > tau {
						want = append(want, p)
					}
				}
				if !SetEq(got, want) {
					t.Fatalf("n=%d seed=%d τ=%d: 填后 %v != 预判 %v",
						n, seed, tau, got, want)
				}
			}
		}
	}
}

func TestCompareFlagsOverOneCell(t *testing.T) {
	// 同坐标：偏差 0，不标
	pr := Compare([]Point{{10, 10}}, []Point{{10, 10}})
	if len(pr) != 1 || pr[0].ChebCells != 0 || pr[0].OverOne {
		t.Fatalf("同坐标配对错误: %+v", pr)
	}
	// 偏移 (2,0)：d∞=2 >1，必须标
	pr = Compare([]Point{{10, 10}}, []Point{{12, 10}})
	if !pr[0].OverOne || pr[0].ChebCells != 2 {
		t.Fatalf("偏移两格应被标记: %+v", pr[0])
	}
	// 对角相邻 (1,1)：d∞=1，不标
	pr = Compare([]Point{{10, 10}}, []Point{{11, 11}})
	if pr[0].OverOne || pr[0].ChebCells != 1 {
		t.Fatalf("一格对角不应标记: %+v", pr[0])
	}
}
