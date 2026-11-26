package calibrate

import (
	"math"
	"testing"

	"goproc/internal/water"
)

func frameFromMask(size int, mask []bool) *water.Frame {
	f := &water.Frame{Size: size, White: append([]bool(nil), mask...)}
	for _, w := range mask {
		if w {
			f.WhiteCount++
		}
	}
	return f
}

func TestFitQuadraticRecoversCurve(t *testing.T) {
	// 用已知二次曲线生成三个标定点，拟合必须原样恢复。
	want := Curve{A: 0.002, B: -0.05, C: 1.2}
	pts := [3]water.CalibrationPoint{
		{LightElevDeg: 10, Threshold: want.Threshold(10)},
		{LightElevDeg: 30, Threshold: want.Threshold(30)},
		{LightElevDeg: 55, Threshold: want.Threshold(55)},
	}
	got, err := FitQuadratic(pts)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got.A-want.A) > 1e-12 || math.Abs(got.B-want.B) > 1e-12 || math.Abs(got.C-want.C) > 1e-12 {
		t.Fatalf("拟合偏差: got %+v want %+v", got, want)
	}
}

func TestFitQuadraticRejectsDuplicateAngles(t *testing.T) {
	pts := [3]water.CalibrationPoint{
		{LightElevDeg: 10, Threshold: 1},
		{LightElevDeg: 10, Threshold: 2},
		{LightElevDeg: 30, Threshold: 3},
	}
	if _, err := FitQuadratic(pts); err == nil {
		t.Fatal("重复光照角应报错")
	}
}

func TestCriticalAngleRoundTrip(t *testing.T) {
	sh := water.DefaultShader()
	pts := [3]water.CalibrationPoint{
		{LightElevDeg: 20, Threshold: 0.999},
		{LightElevDeg: 40, Threshold: 0.9985},
		{LightElevDeg: 60, Threshold: 0.9981},
	}
	c, err := FitQuadratic(pts)
	if err != nil {
		t.Fatal(err)
	}
	threshold := c.Threshold(35)
	gamma, err := CriticalAngle(c, sh, 35)
	if err != nil {
		t.Fatal(err)
	}
	// 反解根代回模型，反射系数必须回到阈值。
	back := sh.ReflectanceAt(math.Cos(gamma))
	if math.Abs(back-threshold) > 1e-9 {
		t.Fatalf("反解不闭合: rho=%v threshold=%v", back, threshold)
	}
	if gamma <= 0 || gamma >= math.Pi/2 {
		t.Fatalf("临界夹角越界: %v", gamma)
	}
}

func TestCriticalAngleEmptyWhenThresholdAbovePeak(t *testing.T) {
	sh := water.DefaultShader()
	c := Curve{A: 0, B: 0, C: sh.K + 1} // 阈值高于正入射峰值 K
	if _, err := CriticalAngle(c, sh, 30); err == nil {
		t.Fatal("阈值高于峰值应报错")
	}
}

func TestCompareMarksOverTwoCells(t *testing.T) {
	size := 8
	rendered := make([]bool, size*size)
	pred := make([]bool, size*size)
	// 渲染白像素在 (0,0)；预判斑块放在远处。
	rendered[0] = true
	pred[4*size+4] = true // (4,4)：到 (0,0) 格距 4
	pred[0*size+1] = true // (1,0)：到 (0,0) 格距 1
	d := Compare(pred, frameFromMask(size, rendered), size)
	if d.PredictedCount != 2 || d.RenderedWhiteCount != 1 {
		t.Fatalf("计数错误: %+v", d)
	}
	if d.OnlyPredicted != 2 || d.OnlyRendered != 1 {
		t.Fatalf("集合差异错误: %+v", d)
	}
	if d.OverTwoCells != 1 || !d.OverMask[4*size+4] {
		t.Fatalf("超两格标注错误: over=%d mask=%v", d.OverTwoCells, d.OverMask)
	}
	if d.OverMask[0*size+1] {
		t.Fatal("一格偏差不应标注为超两格")
	}
	if math.Abs(d.MaxOffsetCells-4*math.Sqrt2) > 1e-9 {
		t.Fatalf("最大偏差应为 4 个对角步（4√2 格），实际 %v", d.MaxOffsetCells)
	}
}

func TestCompareIdenticalMasks(t *testing.T) {
	size := 4
	mask := []bool{true, false, true, false, false, true, false, false, true, false, false, true, false, true, false, false}
	d := Compare(mask, frameFromMask(size, mask), size)
	if d.OverTwoCells != 0 || d.MaxOffsetCells != 0 ||
		d.OnlyPredicted != 0 || d.OnlyRendered != 0 {
		t.Fatalf("一致掩码不应有偏差: %+v", d)
	}
}

func TestPredictIsGeometricOnly(t *testing.T) {
	// 平面高度场：法线全部朝天，预判集合应只取决于半向量夹角，
	// 且临界夹角与曲线反解一致。
	size := 5
	flat := make([]float64, size*size)
	sh := water.DefaultShader()
	nf := water.NewNormalField(size, flat, sh.GridStep)
	c := Curve{A: 0, B: 0, C: 0.999}
	gamma, err := CriticalAngle(c, sh, 60)
	if err != nil {
		t.Fatal(err)
	}
	pred, err := Predict(c, sh, nf, 60)
	if err != nil {
		t.Fatal(err)
	}
	// 60 度光照时半向量朝天，与法线夹角为零；若零角在临界角内则全中。
	wantAll := 0.0 <= gamma
	for i, m := range pred.Mask {
		if m != wantAll {
			t.Fatalf("像素 %d 预判 %v，期望全%v", i, m, wantAll)
		}
	}
	if pred.Count != size*size && wantAll {
		t.Fatalf("预判计数 %d 期望 %d", pred.Count, size*size)
	}
}
