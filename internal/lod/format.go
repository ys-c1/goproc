package lod

import "strconv"

// strconvFormat 以最少小数位格式化浮点。
func strconvFormat(v float64) string {
	return strconv.FormatFloat(v, 'f', 3, 64)
}

// strconvFormatPrec 以指定小数位格式化。
func strconvFormatPrec(v float64, prec int) string {
	return strconv.FormatFloat(v, 'f', prec, 64)
}

// itoa 是 strconv.Itoa 的短别名，供渲染标注使用。
func itoa(i int) string { return strconv.Itoa(i) }
