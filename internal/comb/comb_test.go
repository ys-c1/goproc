package comb

import (
	"math"
	"testing"
)

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestRationalExpansion(t *testing.T) {
	cases := []struct {
		r     float64
		wantP int
		wantQ int
	}{
		{4.0, 4, 1},
		{4.0 / 3.0, 4, 3},
	}
	for _, c := range cases {
		p, q, err := rational(c.r, 10000)
		if err > 1e-12 || p != c.wantP || q != c.wantQ {
			t.Fatalf("r=%v got %d/%d err=%v", c.r, p, q, err)
		}
	}
	if IsRational(math.Sqrt2) {
		t.Fatal("sqrt2 不应被判为有理数")
	}
	if !IsRational(4.0 / 3.0) {
		t.Fatal("4/3 应被判为有理数")
	}
}

func TestCounterexampleProducesMostTeeth(t *testing.T) {
	p := DefaultParams()
	results := map[string]*Result{}
	for _, g := range StandardGroups() {
		results[g.Name] = RunGroup(p, g)
	}
	a := results[StandardGroups()[0].Name]
	b := results[StandardGroups()[1].Name]
	c := results[StandardGroups()[2].Name]

	// 反例组梳齿必须明显多于其余两组（16 vs 0、6）
	if len(a.Teeth) <= len(b.Teeth) || len(a.Teeth) <= len(c.Teeth) {
		t.Fatalf("反例未占优: A=%d B=%d C=%d", len(a.Teeth), len(b.Teeth), len(c.Teeth))
	}
	if len(a.Teeth) < 2*len(c.Teeth) {
		t.Fatalf("反例优势不明显: A=%d C=%d", len(a.Teeth), len(c.Teeth))
	}
	// 无理数组必须为 0，且其最小残差确实超过分类容差
	if len(b.Teeth) != 0 {
		t.Fatalf("无理数组不应有梳齿， got %d", len(b.Teeth))
	}
	if b.MinResidualDeg <= p.PhiTol*180/math.Pi {
		t.Fatalf("无理数组最小残差 %v 未超过容差", b.MinResidualDeg)
	}
}

func TestPredictionsMatchDetection(t *testing.T) {
	p := DefaultParams()
	for _, g := range StandardGroups() {
		r := RunGroup(p, g)
		if r.Diff() != 0 {
			t.Fatalf("%s 实测 %d 与预测 %d 不符", g.Name, len(r.Teeth), r.Predicted)
		}
	}
}

func TestSpacingInverseToRatio(t *testing.T) {
	p := DefaultParams()
	// r=4/1 -> p/q=4/1, d=16px；r=4/3 -> p/q=4/3, d=48px
	_, q1, d1, _ := p.PredictRational(4.0)
	_, q3, d3, _ := p.PredictRational(4.0 / 3.0)
	if q1 != 1 || !approx(d1, 16) {
		t.Fatalf("r=4: q=%d d=%v", q1, d1)
	}
	if q3 != 3 || !approx(d3, 48) {
		t.Fatalf("r=4/3: q=%d d=%v", q3, d3)
	}
	// 同分子 4：间距比 d3/d1=3，与频率比 r1/r3=3 相等（间距与 r 成反比）
	if !approx(d3/d1, 3.0) {
		t.Fatalf("间距反比关系不成立: %v", d3/d1)
	}
}

func TestPhaseSweepAlignsAtPredictedAngles(t *testing.T) {
	p := DefaultParams()
	// r=4 时 Δφ(c)=2π·3·f1·c+δ，f1=1/16 -> 列步长 67.5°。
	// 只要存在整数 c 使 wrap(δ+67.5°·c)≈0 就出梳齿：δ=0°（c=0）与 δ≈67.5°（c=15）两族。
	// 分类容差 1°；整数度网格上实测仅下列角度出齿。
	predictedAngles := map[int]bool{
		0: true, 1: true,
		67: true, 68: true,
		292: true, 293: true,
		360: true,
	}
	got := map[int]int{}
	for deg := 0; deg <= 360; deg++ {
		f := Build(p, 4.0, float64(deg)*math.Pi/180)
		_, teeth := f.Detect()
		if len(teeth) > 0 {
			got[deg] = len(teeth)
		}
	}
	for deg := range got {
		if !predictedAngles[deg] {
			t.Fatalf("δ=%d° 出现未预测的梳齿 %d 个", deg, got[deg])
		}
	}
	for deg := range predictedAngles {
		if got[deg] == 0 {
			t.Fatalf("δ=%d° 应出现梳齿但未检出", deg)
		}
	}
	// δ=0 反例族必须是满齿数（16）
	if got[0] != 16 || got[360] != 16 {
		t.Fatalf("δ=0 族齿数异常: %v", got)
	}
}

func TestGeomEpsilonFormula(t *testing.T) {
	p := DefaultParams()
	// 手算：2*acos((1.70+0.10)/2)=2*acos(0.9)≈0.902 rad≈51.68deg
	want := 2 * math.Acos(0.9) * 180 / math.Pi
	got := p.GeomEpsilon() * 180 / math.Pi
	if math.Abs(got-want) > 0.01 {
		t.Fatalf("GeomEpsilon=%v want %v", got, want)
	}
}
