// Package caliper 的测试：凸包、卡壳枚举的完备性、与暴力枚举的一致性。
package caliper

import (
	"math"
	"math/rand"
	"testing"
)

// allAntipodal 按定义枚举对踵点对：对每条边取最大面积顶点，面积最大者即该边的对踵顶点。
// 在两个顶点同时取到最大面积（平行边）时两者都记入。这是独立于卡壳实现的真值定义。
func allAntipodal(hull []Point) map[Pair]struct{} {
	n := len(hull)
	out := make(map[Pair]struct{})
	if n < 2 {
		return out
	}
	if n == 2 {
		out[Pair{0, 1}] = struct{}{}
		return out
	}
	add := func(a, b int) {
		if a > b {
			a, b = b, a
		}
		out[Pair{a, b}] = struct{}{}
	}
	for i := 0; i < n; i++ {
		ni := (i + 1) % n
		var best int64
		first := true
		for j := 0; j < n; j++ {
			v := Cross(hull[i], hull[ni], hull[j])
			if first || v > best {
				best = v
				first = false
			}
		}
		for j := 0; j < n; j++ {
			if Cross(hull[i], hull[ni], hull[j]) == best {
				add(i, j)
				add(ni, j)
			}
		}
	}
	return out
}

func randomPoints(rng *rand.Rand, k, r int) []Point {
	pts := make([]Point, k)
	for i := range pts {
		pts[i] = Point{rng.Intn(2*r+1) - r, rng.Intn(2*r+1) - r}
	}
	return pts
}

func gridPoints(w, h int) []Point {
	pts := make([]Point, 0, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			pts = append(pts, Point{x, y})
		}
	}
	return pts
}

func dedup(pts []Point) []Point {
	seen := make(map[Point]struct{}, len(pts))
	out := make([]Point, 0, len(pts))
	for _, p := range pts {
		if _, ok := seen[p]; ok {
			continue
		}
		seen[p] = struct{}{}
		out = append(out, p)
	}
	return out
}

func samePairSet(a, b map[Pair]struct{}) (extra, missing int) {
	for p := range a {
		if _, ok := b[p]; !ok {
			extra++
		}
	}
	for p := range b {
		if _, ok := a[p]; !ok {
			missing++
		}
	}
	return
}

// TestCalipersMatchesDefinition 卡壳枚举集必须与“逐边取最大面积顶点”的定义集逐对相等。
func TestCalipersMatchesDefinition(t *testing.T) {
	check := func(label string, pts []Point) {
		hull := Hull(pts)
		got, _, _ := Antipodal(hull)
		gotSet := make(map[Pair]struct{}, len(got))
		for _, p := range got {
			gotSet[p] = struct{}{}
		}
		want := allAntipodal(hull)
		if len(gotSet) != len(got) {
			t.Fatalf("%s: 卡壳输出有重复", label)
		}
		if extra, missing := samePairSet(gotSet, want); extra != 0 || missing != 0 {
			t.Fatalf("%s: 卡壳与定义不符（多 %d 少 %d），hull=%v got=%v want=%v",
				label, extra, missing, hull, got, want)
		}
	}

	rng := rand.New(rand.NewSource(20261127))
	for trial := 0; trial < 500; trial++ {
		k := 1 + rng.Intn(80)
		r := []int{3, 5, 10, 25, 60}[rng.Intn(5)]
		check("随机", randomPoints(rng, k, r))
	}
	for w := 2; w <= 9; w++ {
		check("方阵", gridPoints(w, w))
	}
	for _, n := range []int{3, 4, 8, 16, 32} {
		var pts []Point
		for y := 0; y < n; y++ {
			for x := 0; x < n; x++ {
				if x*x+y*y <= n*n {
					pts = append(pts, Point{x, y})
				}
			}
		}
		check("圆盘", pts)
	}
	check("单点", []Point{{4, 4}})
	check("两点", []Point{{0, 0}, {3, 4}})
	check("共线", []Point{{0, 0}, {1, 1}, {2, 2}, {3, 3}})
	check("全部重合", []Point{{1, 1}, {1, 1}, {1, 1}})
	// 平行边：等腰梯形，上下底平行，卡壳需补记并排的两个对踵点。
	check("平行边", []Point{{0, 0}, {10, 0}, {7, 5}, {3, 5}})
}

// TestCalipersMatchesBrute 预判距离必须等于对全部点的暴力枚举距离。
func TestCalipersMatchesBrute(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	for trial := 0; trial < 400; trial++ {
		k := 2 + rng.Intn(100)
		r := 2 + rng.Intn(40)
		pts := randomPoints(rng, k, r)
		got, _, _, _, _ := Farthest(pts)
		want := Brute(pts)
		if !want.OK {
			continue
		}
		if !got.OK {
			t.Fatalf("trial %d: 预判未给出结果", trial)
		}
		if got.Dist2 != want.Dist2 {
			t.Fatalf("trial %d: 预判 d2=%d (%v,%v) 暴力 d2=%d (%v,%v)",
				trial, got.Dist2, got.A, got.B, want.Dist2, want.A, want.B)
		}
		if got.DistPx() != want.DistPx() {
			t.Fatalf("trial %d: 取整距离 %d != %d", trial, got.DistPx(), want.DistPx())
		}
		if got.A != want.A || got.B != want.B {
			t.Fatalf("trial %d: 并列裁决不一致 预判 (%v,%v) 暴力 (%v,%v)",
				trial, got.A, got.B, want.A, want.B)
		}
		// 最远点对必是对踵点对，而卡壳枚举了凸包的全部对踵点对，
		// 因此按“不同坐标对”计的并列数在两口径下应相等。随机点集可能有
		// 重复坐标，故这里对暴力口径先去重，避免把重复索引对计入并列。
		uniq := dedup(pts)
		if got.Ties != Brute(uniq).Ties {
			t.Fatalf("trial %d: 并列数 预判 %d 暴力(去重) %d", trial, got.Ties, Brute(uniq).Ties)
		}
	}
}

// TestCalipersGrid 全格子（正方形点阵）的对踵点对包含两条对角线，最远距离为 2(S-1)²。
func TestCalipersGrid(t *testing.T) {
	for _, s := range []int{2, 3, 8, 64} {
		pts := gridPoints(s, s)
		got, hull, pairs, rounds, steps := Farthest(pts)
		if len(hull) != 4 {
			t.Fatalf("S=%d: 凸包顶点 %d != 4", s, len(hull))
		}
		if rounds != len(hull) {
			t.Fatalf("S=%d: 卡壳轮数 %d != 凸包顶点数 %d", s, rounds, len(hull))
		}
		if want := int64(2 * (s - 1) * (s - 1)); got.Dist2 != want {
			t.Fatalf("S=%d: 最远 d2=%d != %d", s, got.Dist2, want)
		}
		if len(pairs) != 6 {
			t.Fatalf("S=%d: 对踵点对 %d != 6", s, len(pairs))
		}
		if steps < 0 || steps > 2*len(hull) {
			t.Fatalf("S=%d: 游标步数 %d 超出上界 2h", s, steps)
		}
	}
}

// TestCalipersStepsBound 游标总推进量不超过 2h：卡壳不重复扫描。
func TestCalipersStepsBound(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for trial := 0; trial < 200; trial++ {
		pts := randomPoints(rng, 5+rng.Intn(60), 10+rng.Intn(50))
		hull := Hull(pts)
		_, _, steps := Antipodal(hull)
		if len(hull) >= 3 && steps > 2*len(hull) {
			t.Fatalf("trial %d: h=%d 步数 %d > 2h", trial, len(hull), steps)
		}
	}
}

// TestTieBreakStable 距离并列时取字典序最小的坐标对，且并列数如实统计。
func TestTieBreakStable(t *testing.T) {
	// 正方形四个角：两条对角线都达到最远距离，并列数 2。
	pts := []Point{{0, 0}, {10, 0}, {10, 10}, {0, 10}}
	res, _, _, _, _ := Farthest(pts)
	if res.Dist2 != 200 {
		t.Fatalf("d2=%d != 200", res.Dist2)
	}
	if res.Ties != 2 {
		t.Fatalf("并列数 %d != 2", res.Ties)
	}
	if res.A != (Point{0, 0}) || res.B != (Point{10, 10}) {
		t.Fatalf("并列裁决结果为 (%v,%v)，应为 (0,0)-(10,10)", res.A, res.B)
	}
}

// TestHullCollinear 共线点集不应保留共线中间点。
func TestHullCollinear(t *testing.T) {
	h := Hull([]Point{{0, 0}, {1, 1}, {2, 2}, {3, 3}, {4, 4}})
	if len(h) != 2 || h[0] != (Point{0, 0}) || h[1] != (Point{4, 4}) {
		t.Fatalf("共线时凸包应为两端点，得到 %v", h)
	}
	h = Hull(nil)
	if h != nil {
		t.Fatalf("空输入应返回 nil，得到 %v", h)
	}
}

// TestRoundPx 距离取整必须与浮点四舍五入一致，且在半格附近无误差。
func TestRoundPx(t *testing.T) {
	for d := int64(0); d < 400000; d++ {
		want := int(math.Floor(math.Sqrt(float64(d)) + 0.5))
		if got := RoundPx(d); got != want {
			t.Fatalf("d2=%d: RoundPx=%d want %d", d, got, want)
		}
	}
	// 两个大数：S=4096 画幅的最大平方距离仍须精确。
	for _, d := range []int64{33546241, 33554432, 1<<40 + 12345} {
		want := int(math.Floor(math.Sqrt(float64(d)) + 0.5))
		if got := RoundPx(d); got != want {
			t.Fatalf("d2=%d: RoundPx=%d want %d", d, got, want)
		}
	}
}
