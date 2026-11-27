package cliff

import (
	"math"
	"testing"
)

func TestLimitsAndCrossover(t *testing.T) {
	if got := Levels(8); got != 256 {
		t.Fatalf("八位级数=%d", got)
	}
	if got := QuantLimit(1.0, 256); got != 255 {
		t.Fatalf("八位置换上限=%v", got)
	}
	if got := QuantLimit(0.5, 256); got != 127.5 {
		t.Fatalf("半落差置换上限=%v", got)
	}
	if got := CrossoverSteepness(256); math.Abs(got-1.0/255) > 1e-12 {
		t.Fatalf("八位交点陡度=%v", got)
	}
	if got := CrossoverSteepness(65536); math.Abs(got-1.0/65535) > 1e-15 {
		t.Fatalf("十六位交点陡度=%v", got)
	}
	// 交点处两约束必须相等。
	if math.Abs(GeomLimit(1.0, CrossoverSteepness(256))-QuantLimit(1.0, 256)) > 1e-9 {
		t.Fatal("交点处两约束不等")
	}
}

func TestQuantizeBoundary(t *testing.T) {
	if got := Quantize(0.0, 256); got != 0 {
		t.Fatalf("零点量化=%d", got)
	}
	if got := Quantize(1.0, 256); got != 255 {
		t.Fatalf("满幅量化=%d", got)
	}
	// 四舍五入：0.5/255 正好落在第一级边界。
	if got := Quantize(0.5/255, 256); got != 1 {
		t.Fatalf("边界量化=%d", got)
	}
	if got := Quantize(1.0, 65536); got != 65535 {
		t.Fatalf("十六位满幅量化=%d", got)
	}
}

func TestClampedGeometryDominated(t *testing.T) {
	const H, s = 1.0, 0.01 // s 大于八位交点 1/255，几何主导
	ramp := ClampedRamp(H, s, 256)
	if len(ramp) != 100 {
		t.Fatalf("几何主导倒角宽度=%d，期望 100", len(ramp))
	}
	line := Profile1D(H, ramp, 8)
	codes := QuantizeLine(line, 256)
	top, bot, B, D := BandStats(codes)
	if top != 255 || bot != 0 {
		t.Fatalf("平台码异常 top=%d bot=%d", top, bot)
	}
	// 左边缘采样下首像素等于顶平台码被排除，实测带宽 B=n-1=99。
	if B != len(ramp)-1 {
		t.Fatalf("空间过渡像素=%d，期望 %d", B, len(ramp)-1)
	}
	// 几何主导时每像素落差超过一个灰度码，所有码互不相同：D=B。
	if D != B {
		t.Fatalf("几何主导应每像素可分辨，D=%d B=%d", D, B)
	}
}

func TestClampedQuantDominated(t *testing.T) {
	const H, s = 1.0, 0.001 // s 小于八位交点 1/255，量化主导
	ramp := ClampedRamp(H, s, 256)
	if len(ramp) != 255 {
		t.Fatalf("量化主导倒角宽度=%d，期望被压到 255", len(ramp))
	}
	line := Profile1D(H, ramp, 8)
	_, _, B, D := BandStats(QuantizeLine(line, 256))
	// 连续预测 B=255；左边缘采样首像素与顶平台同码被排除，实测 B=254，
	// 码 1..254 各出现一次，故 D=254，量化台阶全部用满。
	if B != 254 || D != 254 {
		t.Fatalf("量化主导实测 B=%d D=%d，期望 B=254 D=254", B, D)
	}
}

func TestUnclampedQuantDominatedRepeatsCodes(t *testing.T) {
	const H = 1.0
	ramp := UnclampedRamp(H, 1000) // 远大于量化上限 255
	line := Profile1D(H, ramp, 8)
	_, _, B, D := BandStats(QuantizeLine(line, 256))
	// 自然倒角保留全部几何宽度：空间带宽接近 1000（首像素塌缩进顶平台）。
	if B < 990 || B > 1000 {
		t.Fatalf("不裁剪时空间带宽=%d，期望接近 1000", B)
	}
	// 可分辨台阶被量化锁死在 254（码 1..254），与宽度无关。
	if D != 254 {
		t.Fatalf("不裁剪时可分辨台阶=%d，期望 254", D)
	}
	// 裁剪到 min 后空间带宽收缩到量化上限附近，可分辨台阶同样只有 254。
	clamped := ClampedRamp(H, 0.001, 256)
	_, _, B2, D2 := BandStats(QuantizeLine(Profile1D(H, clamped, 8), 256))
	if B2 != 254 || D2 != 254 {
		t.Fatalf("裁剪后 B=%d D=%d，期望 254/254", B2, D2)
	}
	if B2 >= B {
		t.Fatalf("裁剪未收缩带宽: %d -> %d", B, B2)
	}
}

func TestSixteenBitCrossoverRegimes(t *testing.T) {
	// 十六位交点 s*=1/65535 ≈ 1.5259e-5。
	const H = 1.0
	geom := ClampedRamp(H, 1e-4, 65536) // 几何主导：w_g=10000
	if len(geom) != 10000 {
		t.Fatalf("十六位几何主导宽度=%d", len(geom))
	}
	quant := ClampedRamp(H, 1e-6, 65536) // 量化主导：w_g=1e6 > 65535
	if len(quant) != 65535 {
		t.Fatalf("十六位量化主导宽度=%d，期望 65535", len(quant))
	}
}

func TestHalfCliff(t *testing.T) {
	// 半落差 H=0.5：量化上限降到 127.5，交点陡度不变、交点宽度减半。
	if got := CrossoverWidth(0.5, 256); got != 127.5 {
		t.Fatalf("半落差交点宽度=%v", got)
	}
	ramp := ClampedRamp(0.5, 0.001, 256)
	if len(ramp) != 128 {
		t.Fatalf("半落差量化主导宽度=%d，期望 round(127.5)=128", len(ramp))
	}
}
