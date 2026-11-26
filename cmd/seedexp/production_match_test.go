package main

import (
	"testing"

	"goproc/internal/noise"
)

// TestProductionMatchesNoise 锁定实验中的 productionPerm 与生产代码 noise.New
// 的可观测行为一致, 防止两份派生实现漂移后报告结论失效。
//
// 整数坐标处 fade(0)=0, 故 noise.Value(ix,iy) 恰好等于角点梯度 grad(ix,iy),
// 可在不访问私有字段的前提下比对两张置换表。
func TestProductionMatchesNoise(t *testing.T) {
	for _, seed := range []uint64{0, 1, 7, 42, 1 << 32, ^uint64(0)} {
		got := productionPerm(seed)
		s := noise.New(seed)
		for _, q := range [][2]int{{0, 0}, {17, 200}, {255, 255}, {300, 7}, {511, 511}} {
			want := s.Value(float64(q[0]), float64(q[1]))
			have := gradAtInteger(got, q[0], q[1])
			if want != have {
				t.Fatalf("seed=%d cell=%v 生产实现梯度 %v 与实验实现 %v 不一致",
					seed, q, want, have)
			}
		}
	}
}

func gradAtInteger(p [256]uint8, ix, iy int) float64 {
	var tbl [512]uint8
	for i := 0; i < 512; i++ {
		tbl[i] = p[i&255]
	}
	h := tbl[(int(tbl[ix&255])+iy)&511]
	return float64(h)/127.5 - 1
}
