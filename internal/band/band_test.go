package band

import "testing"

func TestStepAndPredict(t *testing.T) {
	if got := Step(8); got != 1.0/255.0 {
		t.Fatalf("Step(8) = %v, want 1/255", got)
	}
	if got := Predict(1.0, 8); got != 256 {
		t.Fatalf("Predict(1, 8) = %d, want 256", got)
	}
	if got := Predict(1.0, 16); got != 65536 {
		t.Fatalf("Predict(1, 16) = %d, want 65536", got)
	}
	if got := Predict(0.5, 8); got != 128 {
		t.Fatalf("Predict(0.5, 8) = %d, want 128", got)
	}
}

func TestQuantize(t *testing.T) {
	got := Quantize([]float64{0, 0.5, 1, -1, 2}, 8)
	want := []uint32{0, 128, 255, 0, 255}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Quantize[%d] = %d, want %d", i, got[i], want[i])
		}
	}
}

func TestScanCountsDisconnectedSegments(t *testing.T) {
	// 同码 1 在四个角上互不相连，四连通下应计为四段；码 2 连成十字为一段。
	grid := []uint32{
		1, 2, 1,
		2, 2, 2,
		1, 2, 1,
	}
	total, segs := Scan(grid, 3)
	if total != 5 {
		t.Fatalf("total = %d, want 5", total)
	}
	ones := 0
	for _, s := range segs {
		if s.Code == 1 {
			ones++
			if s.Pixels != 1 {
				t.Fatalf("code 1 segment pixels = %d, want 1", s.Pixels)
			}
		}
	}
	if ones != 4 {
		t.Fatalf("code 1 segments = %d, want 4", ones)
	}
}
