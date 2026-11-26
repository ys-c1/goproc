package ridge

import (
	"math"
	"testing"
)

// 反例：f₁=xy−0.4(x²+y²) 让任务书判据整片为真，却不含内部极大点；
// f₂=xy+0.4(x²+y²) 让判据整片为假。二者都否证“逐格判定得到连通的脊线段”。
func TestMaskCounterexampleHasNoRidgeLine(t *testing.T) {
	for _, n := range []int{33, 65, 129} {
		for _, tc := range []struct {
			name string
			b    float64
		}{
			{"f1=xy-0.4(x^2+y^2)", -0.4},
			{"f2=xy+0.4(x^2+y^2)", 0.4},
		} {
			fr := Frame{N: n, F: make([]float64, n*n)}
			for y := 0; y < n; y++ {
				for x := 0; x < n; x++ {
					X := -1 + 2*float64(x)/float64(n-1)
					Y := -1 + 2*float64(y)/float64(n-1)
					fr.F[y*n+x] = X*Y + tc.b*(X*X+Y*Y)
				}
			}
			f, err := NewField(fr)
			if err != nil {
				t.Fatal(err)
			}
			mask := f.Mask()
			rep := f.Analyze(tc.name, mask, f.LocalMaxima())
			interiorMax := rep.Actual.InteriorMaxima
			if interiorMax != 0 {
				t.Fatalf("n=%d %s: 反例不应有内部极大点，got %d", n, tc.name, interiorMax)
			}
			if tc.b < 0 {
				if rep.Actual.Total != 1 {
					t.Fatalf("n=%d %s: Mask 应为 1 片，got %d", n, tc.name, rep.Actual.Total)
				}
				if rep.Actual.Total > 0 && interiorMax != 0 {
					t.Fatalf("出现了不含极值点的区域")
				}
			} else {
				if rep.Actual.Total != 0 {
					t.Fatalf("n=%d %s: Mask 应为空，got %d", n, tc.name, rep.Actual.Total)
				}
			}
		}
	}
}

// 抛物线脊：f=1−x² 的脊线是 x=0 的竖直线，G=0 且 λ₁<0 应恰好给出一条直线。
func TestCrestOnParabolaRidge(t *testing.T) {
	n := 33
	fr := Frame{N: n, F: make([]float64, n*n)}
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			X := -2 + 4*float64(x)/float64(n-1)
			fr.F[y*n+x] = 1 - X*X
		}
	}
	f, _ := NewField(fr)
	r := f.Extract(ModeCrest)
	if len(r.Curves) == 0 {
		t.Fatal("抛物线脊应提取出曲线")
	}
	// 所有顶点都应位于网格中轴附近（x≈n/2）。
	cx := 0.0
	np := 0
	for _, p := range r.Points {
		cx += p[0]
		np++
	}
	cx /= float64(np)
	if math.Abs(cx-float64(n-1)/2) > 0.75 {
		t.Fatalf("脊线平均 x=%.2f，应在 %.1f 附近", cx, float64(n-1)/2)
	}
}

// 提取出的脊线图每个顶点度数 ≤2：只可能是开链或闭环。
func TestCrestDegreeAtMostTwo(t *testing.T) {
	n := 64
	fr := Frame{N: n, F: make([]float64, n*n)}
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			X, Y := float64(x)*0.25, float64(y)*0.21
			fr.F[y*n+x] = math.Sin(X)*math.Cos(Y) + 0.3*math.Sin(3*X+1)*math.Cos(2*Y+0.5)
		}
	}
	f, _ := NewField(fr)
	for _, mode := range []CompareMode{ModeCrest, ModeMasked} {
		r := f.Extract(mode)
		for v, d := range r.Degree() {
			if d > 2 {
				t.Fatalf("mode=%d 顶点 %d 度数 %d > 2", mode, v, d)
			}
		}
		for _, c := range r.Curves {
			if !c.Closed && len(c.Endpoints) != 2 {
				t.Fatalf("mode=%d 开曲线端点数 %d != 2", mode, len(c.Endpoints))
			}
		}
	}
}
