package divide

import "testing"

// 两谷夹一脊的一维剖面：中间列最高、向两侧降低。
// 脊西侧格向西流、东侧格向东流，V(2,y) 两侧法向分量相反，逐行构成竖直分水边。
func TestOppositeVerticalOnRidge(t *testing.T) {
	n := 5
	ridge := []float64{0.2, 0.5, 0.9, 0.5, 0.2}
	h := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			h[y*n+x] = ridge[x]
		}
	}
	dir, err := Flow(n, h)
	if err != nil {
		t.Fatal(err)
	}
	for y := 0; y < n; y++ {
		if dir[y*n+1] != W {
			t.Fatalf("脊西侧格(1,%d) 应向西，实际 %d", y, dir[y*n+1])
		}
		if dir[y*n+3] != E {
			t.Fatalf("脊东侧格(3,%d) 应向东，实际 %d", y, dir[y*n+3])
		}
	}
	oh, ov := OppositeEdges(n, dir)
	if len(oh) != 0 {
		t.Fatalf("一维剖面不应有水平反向边，得到 %d", len(oh))
	}
	got := map[int]bool{}
	for _, e := range ov {
		if e.Kind != 'V' || e.X != 2 {
			t.Fatalf("非预期竖直边 %+v", e)
		}
		got[e.Y] = true
	}
	if len(got) != n {
		t.Fatalf("竖直反向边数=%d，期望 %d", len(got), n)
	}
}

// 对称汇水盆：中心最低，所有格朝中心汇聚，相邻格绝不相背，也只属于同一个汇，
// 因此严格反向边与完整分水网都应为空。
func TestBasinNoDivide(t *testing.T) {
	n := 5
	h := make([]float64, n*n)
	cx, cy := 2, 2
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			h[y*n+x] = float64(abs(x-cx) + abs(y-cy))
		}
	}
	dir, hz, vt, oh, ov, sinks, err := Compute(n, h)
	if err != nil {
		t.Fatal(err)
	}
	if dir[cy*n+cx] != none {
		t.Fatalf("盆底汇应无去向，实际 %d", dir[cy*n+cx])
	}
	if sinks != 1 {
		t.Fatalf("单盆汇数=%d，期望 1", sinks)
	}
	if len(hz)+len(vt) != 0 {
		t.Fatalf("单盆不应有分水网，得到 H=%d V=%d", len(hz), len(vt))
	}
	if len(oh)+len(ov) != 0 {
		t.Fatalf("单盆不应有反向边，得到 %d", len(oh)+len(ov))
	}
}

// 两个被脊隔开的独立洼地应形成两个汇，其间产生分水网；且严格反向边是分水网的子集。
func TestTwoSinksProduceDivide(t *testing.T) {
	n := 7
	h := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			// 左右两个洼地（距离各自洼心越近越低），中间 x=3 隆起成脊。
			dl := abs(x-1) + abs(y-3)
			dr := abs(x-5) + abs(y-3)
			d := dl
			if dr < d {
				d = dr
			}
			h[y*n+x] = float64(d) // 离最近洼心越远越高，脊在中间最高
		}
	}
	_, hz, vt, oh, ov, sinks, err := Compute(n, h)
	if err != nil {
		t.Fatal(err)
	}
	if sinks < 2 {
		t.Fatalf("应至少有两个汇，实际 %d", sinks)
	}
	if len(hz)+len(vt) == 0 {
		t.Fatal("两洼之间应产生分水网")
	}
	// 严格反向边必须全部落在完整分水网内。
	fullH := map[[2]int]bool{}
	for _, e := range hz {
		fullH[[2]int{e.X, e.Y}] = true
	}
	fullV := map[[2]int]bool{}
	for _, e := range vt {
		fullV[[2]int{e.X, e.Y}] = true
	}
	for _, e := range oh {
		if !fullH[[2]int{e.X, e.Y}] {
			t.Fatalf("水平反向边 %+v 不在分水网内", e)
		}
	}
	for _, e := range ov {
		if !fullV[[2]int{e.X, e.Y}] {
			t.Fatalf("竖直反向边 %+v 不在分水网内", e)
		}
	}
}

// 平地：斜台地整体获得统一排水方向；闭洼保持 none 并自成一个汇。
func TestResolveFlatsSlopingAndClosed(t *testing.T) {
	n := 4
	h := []float64{
		0.5, 0.5, 0.5, 0.1,
		0.5, 0.5, 0.5, 0.1,
		0.8, 0.8, 0.6, 0.4,
		0.8, 0.8, 0.6, 0.4,
	}
	dir, err := Flow(n, h)
	if err != nil {
		t.Fatal(err)
	}
	if dir[0] != none {
		t.Fatalf("台地深处格初始应为 none，实际 %d", dir[0])
	}
	ResolveFlats(n, h, dir)
	for _, p := range [][2]int{{0, 0}, {1, 0}, {2, 0}, {0, 1}, {1, 1}, {2, 1}} {
		if got := dir[p[1]*n+p[0]]; got != E {
			t.Fatalf("台地格 %v 方向 %d，期望统一朝东 E", p, got)
		}
	}

	closed := []float64{
		0.9, 0.9, 0.9, 0.9,
		0.9, 0.2, 0.9, 0.9,
		0.9, 0.9, 0.9, 0.9,
		0.9, 0.9, 0.9, 0.9,
	}
	cdir, err := Flow(n, closed)
	if err != nil {
		t.Fatal(err)
	}
	ResolveFlats(n, closed, cdir)
	if cdir[1*n+1] != none {
		t.Fatalf("闭洼格应保持 none，实际 %d", cdir[1*n+1])
	}
	sink, cnt := Sinks(n, cdir)
	if cnt != 1 || sink[1*n+1] != sink[0] {
		t.Fatalf("闭洼地形应解析为单汇，cnt=%d", cnt)
	}
}

// Trace 把共享网格顶点的格边描成同一段，互不接触者各自成段。
func TestTraceConnectivity(t *testing.T) {
	edges := []Edge{
		{'H', 0, 1}, // 端点 (0,1)-(1,1)
		{'V', 1, 1}, // 端点 (1,1)-(1,2)，与上一条共享 (1,1)
		{'V', 3, 3}, // 孤立
	}
	segs := Trace(edges)
	if len(segs) != 2 {
		t.Fatalf("连通线段数=%d，期望 2", len(segs))
	}
	big, small := 0, 0
	for _, s := range segs {
		switch len(s.Edges) {
		case 2:
			big++
		case 1:
			small++
		}
	}
	if big != 1 || small != 1 {
		t.Fatalf("线段长度分布错误 big=%d small=%d", big, small)
	}
}

// Crest 把四条过格轴线（正交两轴 + 两对角）任一上的严格局部极大标为脊像素；
// 等高台地不产生脊。
func TestCrestStrictLocalMax(t *testing.T) {
	// 5x5 十字脊：中间行、中间列抬升为 0.9，其余为 0.1；脊格全部位于内部可判定区。
	n := 5
	h := make([]float64, n*n)
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			if x == 2 || y == 2 {
				h[y*n+x] = 0.9
			}
		}
	}
	pts := Crest(n, h)
	got := map[[2]int]bool{}
	for _, p := range pts {
		got[[2]int{p.X, p.Y}] = true
	}
	// 内部的十字脊格：中间列 x=2 的 y=1,2,3 与中间行 y=2 的 x=1,2,3，并集 5 个。
	want := map[[2]int]bool{
		{2, 1}: true, {2, 2}: true, {2, 3}: true,
		{1, 2}: true, {3, 2}: true,
	}
	if len(got) != len(want) {
		t.Fatalf("脊像素数=%d，期望 %d，实际 %v", len(got), len(want), got)
	}
	for p := range want {
		if !got[p] {
			t.Fatalf("缺少脊像素 %v", p)
		}
	}

	// 对角脊：沿主对角隆起的 5x5，中心线上各格应被对角判据捕获。
	m := 5
	g := make([]float64, m*m)
	for y := 0; y < m; y++ {
		for x := 0; x < m; x++ {
			g[y*m+x] = 0.9 - 0.2*float64(abs(x-y)) // x==y 最高，向两侧降低
		}
	}
	dpts := Crest(m, g)
	onDiag := map[[2]int]bool{}
	for _, p := range dpts {
		onDiag[[2]int{p.X, p.Y}] = true
	}
	for k := 1; k < m-1; k++ {
		if !onDiag[[2]int{k, k}] {
			t.Fatalf("主对角脊像素 (%d,%d) 未被检测到", k, k)
		}
	}

	// 等高台地不满足严格高于，不应产生任何脊像素。
	flat := make([]float64, n*n)
	for i := range flat {
		flat[i] = 0.5
	}
	if got := Crest(n, flat); len(got) != 0 {
		t.Fatalf("等高台地不应有脊像素，得到 %d", len(got))
	}
}

func abs(v int) int {
	if v < 0 {
		return -v
	}
	return v
}
