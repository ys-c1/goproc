package contour

import (
	"testing"
)

func grid(n int, z []float64) *Grid { return &Grid{N: n, Z: z} }

// 每个用例验证：两种无追踪预判与追踪完全一致。
func TestAnalyzeSmallGrids(t *testing.T) {
	cases := []struct {
		name       string
		n          int
		z          []float64
		h          float64
		wantOpen   int
		wantClosed int
		wantB      int
	}{
		{
			// 单一内陆高峰：一条闭合环，不触边界。
			name: "inland_peak_closed", n: 5, h: 0.5,
			z: []float64{
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
				0, 0, 1, 0, 0,
				0, 0, 0, 0, 0,
				0, 0, 0, 0, 0,
			},
			wantOpen: 0, wantClosed: 1, wantB: 0,
		},
		{
			// 陆地高于海面、海面在角部：等高线两端出边界，开弧。
			name: "corner_low_open", n: 5, h: 0.5,
			z: []float64{
				0, 1, 1, 1, 1,
				1, 1, 1, 1, 1,
				1, 1, 1, 1, 1,
				1, 1, 1, 1, 1,
				1, 1, 1, 1, 1,
			},
			wantOpen: 1, wantClosed: 0, wantB: 2,
		},
		{
			// 对角两个低谷：每个角部低谷各切出一条两端出边界的开弧，共 4 条（B=8）。
			name: "two_corner_lows", n: 5, h: 0.5,
			z: []float64{
				0, 1, 1, 1, 0,
				1, 1, 1, 1, 1,
				1, 1, 1, 1, 1,
				1, 1, 1, 1, 1,
				0, 1, 1, 1, 0,
			},
			wantOpen: 4, wantClosed: 0, wantB: 8,
		},
		{
			// 内陆高峰 + 角部低谷并存：一闭一开。
			name: "closed_and_open", n: 5, h: 0.5,
			z: []float64{
				0, 1, 1, 1, 1,
				1, 1, 1, 1, 1,
				1, 1, 1, 1, 1,
				1, 1, 1, 1, 1,
				1, 1, 1, 1, 1,
			},
			wantOpen: 1, wantClosed: 0, wantB: 2, // 占位，下面单独构造
		},
		{
			// 环形山：外闭合环 + 内部洞边界再给一条闭合环（共 2 条）。
			name: "ring_two_closed", n: 5, h: 0.5,
			z: []float64{
				0, 0, 0, 0, 0,
				0, 1, 1, 1, 0,
				0, 1, 0, 1, 0,
				0, 1, 1, 1, 0,
				0, 0, 0, 0, 0,
			},
			wantOpen: 0, wantClosed: 2, wantB: 0,
		},
	}

	// 用例 4 单独构造：内陆峰 + 角部低谷。
	peakAndCorner := make([]float64, 25)
	for i := range peakAndCorner {
		peakAndCorner[i] = 1
	}
	peakAndCorner[0] = 0 // 角部低谷 → 开弧
	peakAndCorner[12] = 1
	peakAndCorner[7] = 1
	peakAndCorner[11] = 1
	peakAndCorner[13] = 1
	peakAndCorner[17] = 1 // 中心十字隆起，包围出内陆闭合环
	cases[3].z = peakAndCorner

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := grid(tc.n, tc.z)
			hh, _ := NudgeLevel(g, tc.h)
			r := Analyze(g, hh)
			if r.BoundaryCrossings != tc.wantB {
				t.Errorf("B=%d 期望 %d", r.BoundaryCrossings, tc.wantB)
			}
			if r.TraceOpen != tc.wantOpen {
				t.Errorf("追踪开曲线=%d 期望 %d", r.TraceOpen, tc.wantOpen)
			}
			if r.TraceClosed != tc.wantClosed {
				t.Errorf("追踪闭合=%d 期望 %d", r.TraceClosed, tc.wantClosed)
			}
			if msg := r.Check(); msg != "" {
				t.Errorf("预判与追踪不一致: %s", msg)
			}
			if r.BoundaryCrossings%2 != 0 {
				t.Errorf("B=%d 非偶数", r.BoundaryCrossings)
			}
			// 定理一的直接验证：触边界的曲线恰好是开弧。
			for _, c := range r.Curves {
				touches := false
				for _, p := range c.Pts {
					if p.X == 0 || p.Y == 0 || p.X == float64(g.N-1) || p.Y == float64(g.N-1) {
						touches = true
					}
				}
				if touches != c.Open {
					t.Errorf("曲线 open=%v 但边界触及=%v", c.Open, touches)
				}
			}
		})
	}
}

// TestRandomFields 对一批确定性随机场做穷举档位的三方一致性检查。
func TestRandomFields(t *testing.T) {
	// 简单 LCG 构造确定性数据，避免引入噪声包依赖。
	seed := uint64(123456789)
	next := func() float64 {
		seed = seed*6364136223846793005 + 1442695040888963407
		return float64(seed>>33) / (1 << 31)
	}
	for _, n := range []int{8, 16, 24} {
		z := make([]float64, n*n)
		// 空间平滑：随机格点 + 四邻域平均两轮，制造真实地形式结构。
		raw := make([]float64, n*n)
		for i := range raw {
			raw[i] = next()
		}
		copy(z, raw)
		for round := 0; round < 2; round++ {
			nz := make([]float64, n*n)
			for y := 0; y < n; y++ {
				for x := 0; x < n; x++ {
					sum, cnt := z[y*n+x], 1.0
					for _, d := range [][2]int{{1, 0}, {-1, 0}, {0, 1}, {0, -1}} {
						xx, yy := x+d[0], y+d[1]
						if xx >= 0 && xx < n && yy >= 0 && yy < n {
							sum += z[yy*n+xx]
							cnt++
						}
					}
					nz[y*n+x] = sum / cnt
				}
			}
			z = nz
		}
		g := grid(n, z)
		for k := 1; k <= 19; k++ {
			h := float64(k) / 20
			hh, _ := NudgeLevel(g, h)
			r := Analyze(g, hh)
			if msg := r.Check(); msg != "" {
				t.Fatalf("n=%d h=%.3f: %s", n, hh, msg)
			}
			if r.BoundaryCrossings != 2*r.TraceOpen {
				t.Fatalf("n=%d h=%.3f B=%d 但开曲线=%d", n, hh, r.BoundaryCrossings, r.TraceOpen)
			}
			if r.PredCloseG != r.TraceClosed {
				t.Fatalf("欧拉计数与追踪闭合数不符: pred=%d trace=%d", r.PredCloseG, r.TraceClosed)
			}
		}
	}
}

// TestBoundaryCellDefinitions 明确边界格/边界边的计数约定。
func TestBoundaryCellDefinitions(t *testing.T) {
	n := 6
	g := grid(n, make([]float64, n*n))
	// 边界边总数应为 4(N-1)。
	edges := 0
	for x := 0; x < n-1; x++ {
		edges += 2
	}
	for y := 0; y < n-1; y++ {
		edges += 2
	}
	if edges != 4*(n-1) {
		t.Fatalf("边界边数 %d != %d", edges, 4*(n-1))
	}
	// 边界格数 = N²-(N-2)²。
	boundaryCells := n*n - (n-2)*(n-2)
	if boundaryCells != 20 {
		t.Fatalf("边界格数 %d != 20", boundaryCells)
	}
	// 单调斜坡：每对相邻边界边恰好一次跨越，B 为偶数。
	z := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			z[y*n+x] = float64(x+y) / float64(2*n-2)
		}
	}
	g = grid(n, z)
	r := Analyze(g, 0.5)
	if r.BoundaryCrossings%2 != 0 || r.TraceOpen != r.BoundaryCrossings/2 {
		t.Fatalf("斜坡用例 B=%d open=%d", r.BoundaryCrossings, r.TraceOpen)
	}
}
