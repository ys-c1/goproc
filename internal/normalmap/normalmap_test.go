package normalmap

import (
	"bytes"
	"image/png"
	"math"
	"testing"
)

// TestSafeBoundLemma 验证严格保守界所用的不等式
// asin(√3/M) ≤ 2/M 对所有 b≥3 成立（在很宽的位深范围内数值核验）。
func TestSafeBoundLemma(t *testing.T) {
	for b := 3; b <= 40; b++ {
		M := float64(MaxCode(b))
		got := RigorousBoundRad(b)
		if want := 2 / M; got > want+1e-15 {
			t.Fatalf("b=%d asin(sqrt3/M)=%.12f > 2/M=%.12f", b, got, want)
		}
	}
}

// TestThreeDepthsWithinBound 是核心验收：六/八/十位在标准斜坡的每一个
// 斜坡像素上，实测法线夹角误差都必须落在梯度误差界 BoundDeg 内。
func TestThreeDepthsWithinBound(t *testing.T) {
	target := BuildSlopeTarget(DefaultSlopeConfig())
	nm := target.NormalMap()
	for _, bits := range []int{6, 8, 10} {
		q := Quantize(nm, bits)
		st := MeasureErrors(nm, q, target.Mask)
		t.Logf("bits=%d pixels=%d max=%.5f° mean=%.5f° rms=%.5f° bound@maxG=%.5f° rigorous=%.5f° violations=%d steps=%d levels=%d",
			bits, st.Pixels, st.MaxDeg, st.MeanDeg, st.RMSDeg, st.BoundDeg, RigorousBoundDeg(bits), st.Violations,
			func() int { r, _ := LayerCount(target.SlopeProfile(Shade(q.Normals, target.Light, 0))); return r }(),
			func() int { _, l := LayerCount(target.SlopeProfile(Shade(q.Normals, target.Light, 0))); return l }())
		if st.Violations != 0 {
			t.Fatalf("bits=%d 有 %d 个像素超出误差界", bits, st.Violations)
		}
		if st.MaxDeg > st.BoundDeg+1e-9 {
			t.Fatalf("bits=%d max error %.5f > bound %.5f", bits, st.MaxDeg, st.BoundDeg)
		}
		// 位深加倍码级数（+2 bit ≈ ×4），最大误差应严格下降。
	}
}

// TestErrorDecreasesWithBits 位深越高误差越小。
func TestErrorDecreasesWithBits(t *testing.T) {
	target := BuildSlopeTarget(DefaultSlopeConfig())
	nm := target.NormalMap()
	var prev = math.MaxFloat64
	for _, bits := range []int{6, 8, 10} {
		st := MeasureErrors(nm, Quantize(nm, bits), target.Mask)
		if st.MaxDeg >= prev {
			t.Fatalf("bits=%d max=%.6f 未小于上一档 %.6f", bits, st.MaxDeg, prev)
		}
		prev = st.MaxDeg
	}
}

// TestMinBitsInverse 验证反解的最小性：
// MinBits 给出的 b 满足界，b-1 不满足界。
func TestMinBitsInverse(t *testing.T) {
	for _, g := range []float64{0, 0.5, 1.0, 1.732} {
		for _, eps := range []float64{2.0, 1.0, 0.5, 0.2, 0.1} {
			b := MinBits(g, eps)
			if BoundDeg(b, g) > eps+1e-12 {
				t.Fatalf("g=%.3f eps=%.2f b=%d bound=%.5f > eps", g, eps, b, BoundDeg(b, g))
			}
			if b > 1 && BoundDeg(b-1, g) <= eps+1e-12 {
				t.Fatalf("g=%.3f eps=%.2f b=%d 不是最小: b-1=%d 的界 %.5f 仍满足",
					g, eps, b, b-1, BoundDeg(b-1, g))
			}
		}
	}
}

// TestMinBitsEmpirical 反解位深在真实量化下也应达到目标误差
// （界是严格的，因此界达标必然带动实测达标）。
func TestMinBitsEmpirical(t *testing.T) {
	target := BuildSlopeTarget(DefaultSlopeConfig())
	nm := target.NormalMap()
	gmax := 0.0
	for i, ok := range target.Mask {
		if ok && nm.Gradient[i] > gmax {
			gmax = nm.Gradient[i]
		}
	}
	for _, eps := range []float64{1.0, 0.5, 0.2} {
		b := MinBits(gmax, eps)
		if b < 3 {
			continue
		}
		st := MeasureErrors(nm, Quantize(nm, b), target.Mask)
		if st.MaxDeg > eps {
			t.Fatalf("eps=%.2f 反解 b=%d 但实测 max=%.4f°", eps, b, st.MaxDeg)
		}
		t.Logf("target %.2f° at gmax=%.3f -> minbits=%d, measured max=%.5f°", eps, gmax, b, st.MaxDeg)
	}
}

// TestPlateauExcluded 平台像素不进入统计；平台自身的层数应为 1。
func TestPlateauExcluded(t *testing.T) {
	target := BuildSlopeTarget(DefaultSlopeConfig())
	nm := target.NormalMap()
	q := Quantize(nm, 8)
	shaded := Shade(q.Normals, target.Light, 0)
	// 左侧平台中间一列取整段 y，亮度应完全一致（一层）。
	x := target.Margin / 2
	first := shaded[0*target.Width+x]
	for y := 1; y < target.Height; y++ {
		if shaded[y*target.Width+x] != first {
			t.Fatalf("平台行不变性被破坏 y=%d", y)
		}
	}
	// 统计 mask 不含任何平台列。
	for y := 0; y < target.Height; y++ {
		for x := 0; x < target.Width; x++ {
			i := y*target.Width + x
			if target.Mask[i] && (x < target.X0 || x >= target.X1) {
				t.Fatal("mask 越出斜坡区间")
			}
		}
	}
}

// TestLayerCountsScaleWithBits 位深越高，斜坡亮暗层数越多（约每 +2bit ×4）。
func TestLayerCountsScaleWithBits(t *testing.T) {
	target := BuildSlopeTarget(DefaultSlopeConfig())
	nm := target.NormalMap()
	var prev int
	for _, bits := range []int{6, 8, 10} {
		prof := target.SlopeProfile(Shade(Quantize(nm, bits).Normals, target.Light, 0))
		runs, levels := LayerCount(prof)
		if runs != levels {
			t.Fatalf("bits=%d 单调扫描下 runs=%d != levels=%d", bits, runs, levels)
		}
		if runs <= prev {
			t.Fatalf("bits=%d layers=%d 未多于上一档 %d", bits, runs, prev)
		}
		prev = runs
		t.Logf("bits=%d layers=%d", bits, runs)
	}
}

// TestDeterministic 同参数两次构造结果逐字节一致。
func TestDeterministic(t *testing.T) {
	a := BuildSlopeTarget(DefaultSlopeConfig())
	b := BuildSlopeTarget(DefaultSlopeConfig())
	qa := Quantize(a.NormalMap(), 8)
	qb := Quantize(b.NormalMap(), 8)
	pa := EncodePPM(a.Width, a.Height, 8, qa.Codes)
	pb := EncodePPM(b.Width, b.Height, 8, qb.Codes)
	if string(pa) != string(pb) {
		t.Fatal("量化结果不确定")
	}
}

// TestArchivedPNGRecount 归档的 8 位着色 PNG 中间行重新解码后，
// 数出的亮暗层数必须等于报告中的 Steps8Bit，保证数字可由图片复核。
func TestArchivedPNGRecount(t *testing.T) {
	target := BuildSlopeTarget(DefaultSlopeConfig())
	nm := target.NormalMap()
	for _, bits := range []int{6, 8, 10} {
		rep := EvaluateBits(target, bits)
		shaded := Shade(Quantize(nm, bits).Normals, target.Light, 0)
		pngBytes, err := EncodeGrayPNG(target.Width, target.Height, shaded)
		if err != nil {
			t.Fatal(err)
		}
		img, err := png.Decode(bytes.NewReader(pngBytes))
		if err != nil {
			t.Fatal(err)
		}
		y := target.Height / 2
		row := make([]float64, target.X1-target.X0)
		for x := target.X0; x < target.X1; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			_ = r
			row[x-target.X0] = float64(g >> 8) // Gray 解码后高低字节相同
			_ = b
		}
		runs, _ := LayerCount(row)
		if runs != rep.Steps8Bit {
			t.Fatalf("bits=%d PNG 复核层数 %d != 报告 %d", bits, runs, rep.Steps8Bit)
		}
	}
}

// TestXYGradientDependence XY+z 重构造编码下，高梯度区间的平均角度误差
// 必须大于低梯度区间，坐实“误差随梯度增大”。
func TestXYGradientDependence(t *testing.T) {
	target := BuildSlopeTarget(DefaultSlopeConfig())
	nm := target.NormalMap()
	bins := [][2]float64{{math.Tan(5 * Deg2Rad), math.Tan(14 * Deg2Rad)}, {math.Tan(50 * Deg2Rad), math.Tan(60 * Deg2Rad)}}
	for _, bits := range []int{6, 8, 10} {
		bs := BinErrorsByGradient(nm, QuantizeXY(nm, bits), target.Mask, bins)
		if len(bs) != 2 {
			t.Fatal("bins")
		}
		if bs[1].MeanErrorDeg <= bs[0].MeanErrorDeg {
			t.Fatalf("bits=%d 高梯度误差 %.4f 未大于低梯度 %.4f", bits, bs[1].MeanErrorDeg, bs[0].MeanErrorDeg)
		}
	}
}
