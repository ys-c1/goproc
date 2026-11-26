// Package caliper 用凸包顶点与旋转卡壳求平面整数点集的最远点对。
//
// 全部距离在整数平方距离（int64）上比较，避免浮点误差；对外距离按像素四舍五入取整。
package caliper

import (
	"sort"
)

// Point 是整数格点，坐标单位为像素。
type Point struct{ X, Y int }

// Less 按 (X, Y) 字典序比较。
func Less(a, b Point) bool {
	if a.X != b.X {
		return a.X < b.X
	}
	return a.Y < b.Y
}

// Cross 返回有向面积的两倍：(b-a) × (c-a)。
// 图像坐标 y 轴向下，符号与数学坐标相反；只在同一坐标系内做大小比较，不影响结论。
func Cross(a, b, c Point) int64 {
	return int64(b.X-a.X)*int64(c.Y-a.Y) - int64(b.Y-a.Y)*int64(c.X-a.X)
}

// Dist2 返回两点间的整数平方距离。
func Dist2(a, b Point) int64 {
	dx := int64(a.X) - int64(b.X)
	dy := int64(a.Y) - int64(b.Y)
	return dx*dx + dy*dy
}

// RoundPx 把平方距离换算成像素距离并四舍五入到整数。
//
// 用整数平方根实现，全程不经过浮点：对整数 d，sqrt(d) 恰为 n+1/2 需要
// d=n²+n+1/4，而 d 是整数，故不存在正好落在半格上的情形，四舍五入即
// 取满足 n²≤d 的最大 n，再比较 d-n² 与 (n+1)²-d 判定是否进位。
func RoundPx(dist2 int64) int {
	if dist2 <= 0 {
		return 0
	}
	n := isqrt(dist2)
	if dist2-n*n > (n+1)*(n+1)-dist2 {
		n++
	}
	return int(n)
}

// isqrt 返回 floor(sqrt(v))，v ≥ 0。
func isqrt(v int64) int64 {
	if v <= 0 {
		return 0
	}
	x := v
	y := (x + 1) / 2
	for y < x {
		x = y
		y = (x + v/x) / 2
	}
	return x
}

// Result 是一对点及它们之间的距离。
// 点数不足两个时 OK 为 false。
type Result struct {
	A, B  Point
	Dist2 int64 // 整数平方距离
	Ties  int64 // 达到该距离的无序点对数
	OK    bool
}

// DistPx 返回四舍五入到整数像素的距离。
func (r Result) DistPx() int { return RoundPx(r.Dist2) }

func normPair(a, b Point) (Point, Point) {
	if Less(b, a) {
		return b, a
	}
	return a, b
}

func pairLess(a1, b1, a2, b2 Point) bool {
	if a1 != a2 {
		return Less(a1, a2)
	}
	return Less(b1, b2)
}

// Hull 返回点集的凸包顶点。输入按 (X,Y) 升序、单调链构造，返回值有序且无三点共线。
// 重复点先合并；全部点共线时返回两个端点；单点返回该点。
func Hull(pts []Point) []Point {
	if len(pts) == 0 {
		return nil
	}
	s := append([]Point(nil), pts...)
	sort.Slice(s, func(i, j int) bool { return Less(s[i], s[j]) })
	u := s[:0]
	for _, p := range s {
		if len(u) == 0 || u[len(u)-1] != p {
			u = append(u, p)
		}
	}
	if len(u) <= 2 {
		return u
	}
	lower := make([]Point, 0, len(u))
	for _, p := range u {
		for len(lower) >= 2 && Cross(lower[len(lower)-2], lower[len(lower)-1], p) <= 0 {
			lower = lower[:len(lower)-1]
		}
		lower = append(lower, p)
	}
	upper := make([]Point, 0, len(u))
	for i := len(u) - 1; i >= 0; i-- {
		p := u[i]
		for len(upper) >= 2 && Cross(upper[len(upper)-2], upper[len(upper)-1], p) <= 0 {
			upper = upper[:len(upper)-1]
		}
		upper = append(upper, p)
	}
	h := make([]Point, 0, len(lower)+len(upper)-2)
	h = append(h, lower[:len(lower)-1]...)
	h = append(h, upper[:len(upper)-1]...)
	return h
}

// Pair 是凸包顶点的下标对，已规范化为 I < J。
type Pair struct{ I, J int }

// Antipodal 用旋转卡壳枚举凸包的全部对踵点对。
//
// rounds 是卡壳轮数（每轮处理一条凸包边，等于凸包顶点数；少于三点时为 0 或 1），
// steps 是游标沿凸包推进的总步数。
func Antipodal(hull []Point) (pairs []Pair, rounds, steps int) {
	n := len(hull)
	switch n {
	case 0, 1:
		return nil, 0, 0
	case 2:
		return []Pair{{0, 1}}, 1, 0
	}
	set := make(map[Pair]struct{}, 2*n)
	add := func(a, b int) {
		if a == b {
			return
		}
		if a > b {
			a, b = b, a
		}
		set[Pair{a, b}] = struct{}{}
	}
	j := 1
	guard := 4 * n // 理论上总推进量 <= 2n；越界即说明实现有问题，避免死循环。
	for i := 0; i < n; i++ {
		ni := (i + 1) % n
		rounds++
		for steps <= guard {
			nj := (j + 1) % n
			if Cross(hull[i], hull[ni], hull[nj]) > Cross(hull[i], hull[ni], hull[j]) {
				j = nj
				steps++
				continue
			}
			break
		}
		add(i, j)
		add(ni, j)
		nj := (j + 1) % n
		if Cross(hull[i], hull[ni], hull[nj]) == Cross(hull[i], hull[ni], hull[j]) {
			add(i, nj)
			add(ni, nj)
		}
	}
	pairs = make([]Pair, 0, len(set))
	for p := range set {
		pairs = append(pairs, p)
	}
	sort.Slice(pairs, func(a, b int) bool {
		if pairs[a].I != pairs[b].I {
			return pairs[a].I < pairs[b].I
		}
		return pairs[a].J < pairs[b].J
	})
	return pairs, rounds, steps
}

// FarthestOnHull 在给定对踵点对中取最远的一对。
// 距离并列时取坐标字典序最小的一对，保证结果可复算。
func FarthestOnHull(hull []Point, pairs []Pair) Result {
	if len(pairs) == 0 {
		return Result{}
	}
	res := Result{Dist2: -1}
	for _, p := range pairs {
		if p.I == p.J {
			continue
		}
		a, b := normPair(hull[p.I], hull[p.J])
		d := Dist2(a, b)
		switch {
		case d > res.Dist2:
			res.Dist2 = d
			res.A, res.B = a, b
			res.Ties = 1
			res.OK = true
		case d == res.Dist2:
			res.Ties++
			if pairLess(a, b, res.A, res.B) {
				res.A, res.B = a, b
			}
		}
	}
	return res
}

// Brute 对全部点两两求距，返回最远点对，作为预判结果的实测对照。
// 距离并列时与 FarthestOnHull 采用相同的字典序裁决。
func Brute(pts []Point) Result {
	n := len(pts)
	if n < 2 {
		return Result{}
	}
	res := Result{Dist2: -1}
	for i := 0; i < n; i++ {
		pi := pts[i]
		for j := i + 1; j < n; j++ {
			a, b := normPair(pi, pts[j])
			d := Dist2(a, b)
			switch {
			case d > res.Dist2:
				res.Dist2 = d
				res.A, res.B = a, b
				res.Ties = 1
				res.OK = true
			case d == res.Dist2:
				res.Ties++
				if pairLess(a, b, res.A, res.B) {
					res.A, res.B = a, b
				}
			}
		}
	}
	return res
}

// Farthest 是“凸包 + 旋转卡壳”的完整预判：返回最远点对及凸包统计。
func Farthest(pts []Point) (Result, []Point, []Pair, int, int) {
	hull := Hull(pts)
	pairs, rounds, steps := Antipodal(hull)
	return FarthestOnHull(hull, pairs), hull, pairs, rounds, steps
}
