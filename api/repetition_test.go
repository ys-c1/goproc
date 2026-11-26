package api

import (
	"testing"
)

// 默认（零值 SharedPerm=false）必须走按层子表；显式 SharedPerm=true 走旧的
// 共用置换表路径。两条路径同种子应都确定，但产物应当不同（修复确实改变了结果）。
func TestDefaultPerLayerVsSharedPerm(t *testing.T) {
	opt := DefaultOptions()
	a, err := Generate(opt)
	if err != nil {
		t.Fatal(err)
	}
	b, err := Generate(opt)
	if err != nil {
		t.Fatal(err)
	}
	for i := range a.Data {
		if a.Data[i] != b.Data[i] {
			t.Fatal("默认参数不确定")
		}
	}

	old := opt
	old.SharedPerm = true
	s, err := Generate(old)
	if err != nil {
		t.Fatal(err)
	}
	diff := 0
	for i := range a.Data {
		if a.Data[i] != s.Data[i] {
			diff++
		}
	}
	if diff == 0 {
		t.Fatal("按层子表与共用置换表的归一化结果完全相同，修复未生效")
	}
	if diff < len(a.Data)/2 {
		t.Fatalf("两条路径差异像素过少: %d/%d", diff, len(a.Data))
	}
}
