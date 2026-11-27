// 端到端验证：两块定向地形在两种拼接方向下，旋转对齐后解析主方向差为零，
// 栅格重采样残差落在误差界内。
package orient

import (
	"math"
	"testing"
)

// gratingField 与 internal/field 同构的平稳定向纹理，避免 orient 反向依赖 field。
func gratingField(size int, phiDeg float64, seed uint64) []float64 {
	phi := phiDeg * math.Pi / 180
	d := make([]float64, size*size)
	state := seed
	next := func() float64 {
		state += 0x9e3779b97f4a7c15
		z := state
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb
		z ^= z >> 31
		return float64(z%4096)/2048.0 - 1
	}
	const lambda = 40.0
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			lx := float64(x)*math.Cos(phi) + float64(y)*math.Sin(phi)
			d[y*size+x] = math.Sin(2*math.Pi*lx/lambda) + 0.2*next()
		}
	}
	return d
}

func TestStitchAlignmentEndToEnd(t *testing.T) {
	const size, width, margin, inner = 256, 24, 32, 40
	const rasterBound = 2.5 // 与 cmd/stitch 报告的保守误差界一致（度）

	a := gratingField(size, 20, 7)
	b := gratingField(size, -35, 23)

	for _, sm := range []Seam{Horizontal, Vertical} {
		ax0, ay0, ax1, ay1, bx0, by0, bx1, by1 := SeamStrips(size, width, margin, sm)
		ta, nA := StructureTensor(size, a, ax0, ay0, ax1, ay1, Central)
		tb, nB := StructureTensor(size, b, bx0, by0, bx1, by1, Central)
		if nA != nB || nA == 0 {
			t.Fatalf("seam %v: pixel counts A=%d B=%d", sm, nA, nB)
		}
		thetaA, thetaB := PrincipalAngle(ta), PrincipalAngle(tb)
		phi := RotationAngle(thetaA, thetaB)

		// 解析路径必须恒为零（机器精度）。
		if r := Wrap180(Wrap180(thetaB+phi) - thetaA); math.Abs(r) > 1e-12 {
			t.Errorf("seam %v: algebraic residual %g exceeds 1e-12", sm, r)
		}
		if r := AngularDifference(thetaA, PrincipalAngle(RotateTensor(tb, phi))); math.Abs(r) > 1e-12 {
			t.Errorf("seam %v: tensor-rotate residual %g exceeds 1e-12", sm, r)
		}

		// 栅格路径残差必须落在保守误差界内（中心内区，避开旋转边界填充）。
		rotB := RotateField(size, b, phi)
		ra, _ := StructureTensor(size, a, inner, inner, size-inner, size-inner, Central)
		rb, _ := StructureTensor(size, rotB, inner, inner, size-inner, size-inner, Central)
		if r := math.Abs(AngularDifference(PrincipalAngle(ra), PrincipalAngle(rb))); r > rasterBound {
			t.Errorf("seam %v: raster residual %.4f exceeds bound %.1f", sm, r, rasterBound)
		}

		// 无边界填充的对齐：把 B 的注入方向整体平移 phi（等价于过程化纹理
		// 直接在旋转坐标系重采样），字面接缝条带的角差同样必须落在界内。
		b2 := gratingField(size, Wrap180(-35+phi), 23)
		_, _, _, _, fx0, fy0, fx1, fy1 := SeamStrips(size, width, margin, sm)
		tbF, _ := StructureTensor(size, b2, fx0, fy0, fx1, fy1, Central)
		taSeam, _ := StructureTensor(size, a, ax0, ay0, ax1, ay1, Central)
		if r := math.Abs(AngularDifference(PrincipalAngle(taSeam), PrincipalAngle(tbF))); r > rasterBound {
			t.Errorf("seam %v: field-level seam residual %.4f exceeds bound %.1f", sm, r, rasterBound)
		}
	}
}
