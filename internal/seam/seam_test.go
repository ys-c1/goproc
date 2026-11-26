package seam

import (
	"math"
	"testing"

	"goproc/api"
)

func gen(t *testing.T, cells, period int) *api.Heightmap {
	t.Helper()
	opt := api.DefaultOptions()
	opt.Size = 256
	opt.Octaves = 3
	opt.Seed = 7
	opt.TileCells = cells
	opt.Period = period
	h, err := api.Generate(opt)
	if err != nil {
		t.Fatal(err)
	}
	return h
}

// 三档周期都必须：跳变低于一档灰度，且不超内部基线。
func TestThreePeriodsAreSeamless(t *testing.T) {
	for _, c := range []struct{ cells, period int }{
		{64, 64}, {128, 128}, {256, 256},
	} {
		h := gen(t, c.cells, c.period)
		m := Measure(h)
		if !h.Options().Tileable() {
			t.Fatalf("period=%d 应可平铺", c.period)
		}
		if got := m.JumpMax(); got >= GrayStep {
			t.Errorf("period=%d 跳变 %v 未低于一档灰度 %v", c.period, got, GrayStep)
		}
		if m.Fault() {
			t.Errorf("period=%d 跳变超出内部基线 x=%v/%v y=%v/%v",
				c.period, m.JumpX, m.InteriorX, m.JumpY, m.InteriorY)
		}
		if m.StepExceeds() {
			t.Errorf("period=%d 接缝相邻差超出内部基线", c.period)
		}
	}
}

// 相位余数非零时，跳变随余数变化；余数为零时归零。
func TestRemainderDrivesJump(t *testing.T) {
	base := api.DefaultOptions()
	base.Size = 256
	base.Octaves = 3
	base.Seed = 7
	base.Period = 64

	jumpAt := func(cells int) (rem int, jump float64) {
		o := base
		o.Frequency = float64(cells) / float64(base.Size)
		h, err := api.Generate(o)
		if err != nil {
			t.Fatal(err)
		}
		rem, _ = h.Options().RemainderCells(0)
		return rem, Measure(h).JumpMax()
	}

	rem0, jump0 := jumpAt(64)
	if rem0 != 0 {
		t.Fatalf("余数应为 0，得到 %d", rem0)
	}
	if jump0 >= GrayStep {
		t.Fatalf("余数为零时跳变 %v 应归零", jump0)
	}

	seen := map[float64]bool{}
	for cells := 65; cells <= 95; cells++ {
		rem, jump := jumpAt(cells)
		if rem == 0 {
			t.Fatalf("cells=%d 余数不应为零", cells)
		}
		if jump <= GrayStep {
			t.Errorf("cells=%d 余数 %d 应留下可见跳变，得到 %v", cells, rem, jump)
		}
		seen[math.Round(jump*1e4)] = true
	}
	if len(seen) < 20 {
		t.Fatalf("跳变随余数变化不足，只有 %d 个不同取值", len(seen))
	}
}

// 逐层跳变之和（按振幅权重）应与总量同号且量级一致。
func TestOctaveJumpsCompose(t *testing.T) {
	o := api.DefaultOptions()
	o.Size = 256
	o.Octaves = 3
	o.Seed = 7
	o.Frequency = 70.0 / 256.0 // 不整除的一档
	o.Period = 64
	h, err := api.Generate(o)
	if err != nil {
		t.Fatal(err)
	}
	m := Measure(h)
	sum := 0.0
	for _, v := range m.OctaveJumpX {
		sum += v
	}
	if sum == 0 {
		t.Fatal("逐层跳变全零")
	}
	if math.Abs(sum-m.JumpX) > 1.0 {
		t.Fatalf("逐层拆开之和 %v 与总量 %v 相差过大", sum, m.JumpX)
	}
}
