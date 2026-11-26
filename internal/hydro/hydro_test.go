package hydro

import "testing"

func TestD8PlaneDirectionAndAccumulation(t *testing.T) {
	// 3x3，每一行高程从左到右 2,1,0：水全部向东流，最右列承接本行 3 格。
	h := []float64{
		2, 1, 0,
		2, 1, 0,
		2, 1, 0,
	}
	f := Build(h, 3)
	c := Idx(3, 1, 1)
	if f.Dir[c] != E {
		t.Fatalf("中心格方向应为 E(%d)，实得 %d", E, f.Dir[c])
	}
	if f.Recv[c] != Idx(3, 2, 1) {
		t.Fatalf("中心格接收格错误: %d", f.Recv[c])
	}
	if f.Slope[c] != 1.0 {
		t.Fatalf("正向相邻坡度应为 1，实得 %v", f.Slope[c])
	}
	if f.Acc[Idx(3, 2, 1)] != 3 {
		t.Fatalf("最右中行汇流应为 3，实得 %d", f.Acc[Idx(3, 2, 1)])
	}
	if f.Acc[Idx(3, 0, 0)] != 1 {
		t.Fatalf("源头格汇流应含本格=1，实得 %d", f.Acc[Idx(3, 0, 0)])
	}
}

func TestDiagonalSlopeUsesSqrt2(t *testing.T) {
	// 中心 1，东南 0：落差 1，斜距 sqrt2，坡度 1/sqrt2。
	n := 3
	h := make([]float64, n*n)
	for i := range h {
		h[i] = 2
	}
	c := Idx(n, 1, 1)
	h[c] = 1
	h[Idx(n, 2, 2)] = 0
	f := Build(h, n)
	if f.Dir[c] != SE {
		t.Fatalf("方向应为 SE，实得 %d", f.Dir[c])
	}
	want := 1.0 / 1.4142135623730951
	if d := f.Slope[c] - want; d > 1e-12 || d < -1e-12 {
		t.Fatalf("斜向坡度应为 %v，实得 %v", want, f.Slope[c])
	}
}

func TestTieKeepsLowestDirectionIndex(t *testing.T) {
	// 中心 1，正东与正南都为 0 且等坡：并列取编号更小的 E(0)。
	n := 3
	h := make([]float64, n*n)
	for i := range h {
		h[i] = 2
	}
	c := Idx(n, 1, 1)
	h[c] = 1
	h[Idx(n, 2, 1)] = 0 // E
	h[Idx(n, 1, 2)] = 0 // S
	f := Build(h, n)
	if f.Dir[c] != E {
		t.Fatalf("等坡并列应取 E(0)，实得 dir=%d", f.Dir[c])
	}
}

func TestNoDownhillIsSink(t *testing.T) {
	n := 3
	h := []float64{
		3, 2, 3,
		2, 1, 2,
		3, 2, 3,
	}
	f := Build(h, n)
	c := Idx(n, 1, 1)
	if f.Recv[c] != -1 || f.Slope[c] != 0 {
		t.Fatalf("最低中心格应为 sink，recv=%d slope=%v", f.Recv[c], f.Slope[c])
	}
}
