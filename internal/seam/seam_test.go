package seam

import (
	"bytes"
	"testing"
)

func TestSeamDiffZeroAfterAlignment(t *testing.T) {
	cfg := DefaultConfig()
	a := Generate(cfg, cfg.SeedA, 0, 0)
	for _, dir := range []Direction{Horizontal, Vertical} {
		bx, by := EdgeOffset(cfg, dir)
		bPre := Generate(cfg, cfg.SeedB, bx, by)
		bPost := Generate(cfg, cfg.SeedB, bx, by)

		// 对齐前：两表在边界槽位上取的不是同一批条目。
		slots, diff := BoundaryEntryDiff(cfg, a, bPre, dir)
		if slots == 0 || diff == 0 {
			t.Fatalf("%s: 对齐前边界条目竟全部一致 slots=%d diff=%d", dir, slots, diff)
		}

		entries, copied := Align(cfg, a, bPost, dir)
		if copied != slots {
			t.Fatalf("%s: 对齐条目数 %d 与边界槽位数 %d 不一致", dir, copied, slots)
		}
		// 对齐条件本身：B 表在全部边界槽位上必须与 A 表取同一条目。
		if _, diffAfter := BoundaryEntryDiff(cfg, a, bPost, dir); diffAfter != 0 {
			t.Fatalf("%s: 对齐后仍有 %d 个边界槽位条目不一致", dir, diffAfter)
		}
		for j, want := range entries {
			if got := bPost.Source.Entry(j); got != want {
				t.Fatalf("%s: 槽位 %d 条目未对齐 A=%d B=%d", dir, j, want, got)
			}
		}

		scale := NewScale(a, bPre, bPost)
		mPre := Stitch(a, bPre, dir, scale).Measure()
		mPost := Stitch(a, bPost, dir, scale).Measure()

		if mPre.MaxGrayDiff == 0 {
			t.Fatalf("%s: 对齐前接缝差竟为零，种子未产生不同边界条目", dir)
		}
		if mPre.PairCount != cfg.N {
			t.Fatalf("%s: 像素对数应为 %d，实际 %d", dir, cfg.N, mPre.PairCount)
		}
		// 核心结论：对齐后原始高度差与灰度差都必须恒为零。
		if mPost.MaxRawDiff != 0 {
			t.Fatalf("%s: 对齐后原始高度差非零 %g", dir, mPost.MaxRawDiff)
		}
		if mPost.MaxGrayDiff != 0 || mPost.NonZeroPairs != 0 {
			t.Fatalf("%s: 对齐后灰度差未归零 max=%d nonzero=%d", dir, mPost.MaxGrayDiff, mPost.NonZeroPairs)
		}
		// 旋转交叉验证必须给出同一接缝差。
		if mPre.RotatedMaxGray != mPre.MaxGrayDiff || mPost.RotatedMaxGray != mPost.MaxGrayDiff {
			t.Fatalf("%s: 旋转后度量与直接度量不一致", dir)
		}
	}
}

func TestBoundaryPixelsIdenticalAfterAlignment(t *testing.T) {
	cfg := DefaultConfig()
	a := Generate(cfg, cfg.SeedA, 0, 0)
	bx, by := EdgeOffset(cfg, Horizontal)
	b := Generate(cfg, cfg.SeedB, bx, by)
	Align(cfg, a, b, Horizontal)
	// 逐像素比较共享边界：A 右边界列与 B 左边界列必须逐位相等。
	for y := 0; y < cfg.N; y++ {
		if got, want := b.Data[y*cfg.N], a.Data[y*cfg.N+cfg.N-1]; got != want {
			t.Fatalf("竖缝边界行 %d 高度不等 A=%v B=%v", y, want, got)
		}
	}

	a2 := Generate(cfg, cfg.SeedA, 0, 0)
	bx2, by2 := EdgeOffset(cfg, Vertical)
	b2 := Generate(cfg, cfg.SeedB, bx2, by2)
	Align(cfg, a2, b2, Vertical)
	for x := 0; x < cfg.N; x++ {
		if got, want := b2.Data[x], a2.Data[(cfg.N-1)*cfg.N+x]; got != want {
			t.Fatalf("横缝边界列 %d 高度不等 A=%v B=%v", x, want, got)
		}
	}
}

func TestDeterministicRuns(t *testing.T) {
	cfg := DefaultConfig()
	run := func() Metrics {
		a := Generate(cfg, cfg.SeedA, 0, 0)
		bx, by := EdgeOffset(cfg, Vertical)
		b := Generate(cfg, cfg.SeedB, bx, by)
		Align(cfg, a, b, Vertical)
		return Stitch(a, b, Vertical, NewScale(a, b)).Measure()
	}
	m1, m2 := run(), run()
	if m1.MaxGrayDiff != m2.MaxGrayDiff || m1.MeanGrayDiff != m2.MeanGrayDiff {
		t.Fatal("两次实验结果不确定")
	}
}

func TestEncodeRoundTrip(t *testing.T) {
	cfg := DefaultConfig()
	a := Generate(cfg, cfg.SeedA, 0, 0)
	bx, by := EdgeOffset(cfg, Horizontal)
	b := Generate(cfg, cfg.SeedB, bx, by)
	c := Stitch(a, b, Horizontal, NewScale(a, b))

	var pgm bytes.Buffer
	if err := c.WritePGM(&pgm); err != nil {
		t.Fatal(err)
	}
	head := "P5\n" + itoa(c.Width) + " " + itoa(c.Height) + "\n255\n"
	if got := pgm.String()[:len(head)]; got != head {
		t.Fatalf("PGM 头错误: %q", got)
	}
	if pgm.Len() != len(head)+c.Width*c.Height {
		t.Fatal("PGM 像素数不符")
	}

	var pngBuf bytes.Buffer
	if err := c.WriteMarkedPNG(&pngBuf); err != nil {
		t.Fatal(err)
	}
	if pngBuf.Len() == 0 || pngBuf.Bytes()[0] != 0x89 {
		t.Fatal("PNG 魔数错误")
	}
}
