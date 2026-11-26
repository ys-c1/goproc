package noise

import "testing"

func TestSeedDerivation(t *testing.T) {
	a := New(7)
	b := New(7)
	if a.Value(1.5, 2.5) != b.Value(1.5, 2.5) {
		t.Fatal("同种子不同值")
	}
	c := New(8)
	if a.Value(1.5, 2.5) == c.Value(1.5, 2.5) {
		t.Fatal("不同种子同值")
	}
}

func TestBounded(t *testing.T) {
	s := New(3)
	for i := 0; i < 1000; i++ {
		v := s.Value(float64(i)*0.37, float64(i)*0.11)
		if v < -1.5 || v > 1.5 {
			t.Fatalf("越界: %v", v)
		}
	}
}
