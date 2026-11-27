package hmap

import (
	"errors"
	"fmt"
)

// DecodePGM 解析八位二进制 PGM（P5），返回边长与零到一的浮点数据。
// 仅接受正方形图，足以读取本仓库导出的高度图。
func DecodePGM(b []byte) (int, []float64, error) {
	pos := 0
	token := func() (string, error) {
		for pos < len(b) && isSpace(b[pos]) {
			pos++
		}
		start := pos
		for pos < len(b) && !isSpace(b[pos]) {
			pos++
		}
		if start == pos {
			return "", errors.New("pgm: 头部截断")
		}
		return string(b[start:pos]), nil
	}
	magic, err := token()
	if err != nil {
		return 0, nil, err
	}
	if magic != "P5" {
		return 0, nil, fmt.Errorf("pgm: 不支持的魔数 %q，仅支持 P5", magic)
	}
	// 头部允许 # 注释，出现在空白之后。
	nextToken := func() (string, error) {
		for {
			for pos < len(b) && isSpace(b[pos]) {
				pos++
			}
			if pos < len(b) && b[pos] == '#' {
				for pos < len(b) && b[pos] != '\n' {
					pos++
				}
				continue
			}
			return token()
		}
	}
	wStr, err := nextToken()
	if err != nil {
		return 0, nil, err
	}
	hStr, err := nextToken()
	if err != nil {
		return 0, nil, err
	}
	maxStr, err := nextToken()
	if err != nil {
		return 0, nil, err
	}
	w, err := atoi(wStr)
	if err != nil {
		return 0, nil, fmt.Errorf("pgm: 宽度非法: %w", err)
	}
	h, err := atoi(hStr)
	if err != nil {
		return 0, nil, err
	}
	maxv, err := atoi(maxStr)
	if err != nil {
		return 0, nil, err
	}
	if w <= 0 || h <= 0 || w != h {
		return 0, nil, fmt.Errorf("pgm: 仅支持正方形图，得到 %dx%d", w, h)
	}
	if maxv != 255 {
		return 0, nil, fmt.Errorf("pgm: 仅支持 maxval=255，得到 %d", maxv)
	}
	// maxval 后恰好一个空白符，之后是原始像素。
	if pos >= len(b) || !isSpace(b[pos]) {
		return 0, nil, errors.New("pgm: 像素数据缺少分隔符")
	}
	pos++
	n := w * h
	if len(b)-pos < n {
		return 0, nil, fmt.Errorf("pgm: 像素数据不足，需要 %d 字节实际 %d", n, len(b)-pos)
	}
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = float64(b[pos+i]) / 255.0
	}
	return w, out, nil
}

func isSpace(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'
}

func atoi(s string) (int, error) {
	if s == "" {
		return 0, errors.New("空数字")
	}
	v := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0, fmt.Errorf("非法数字 %q", s)
		}
		v = v*10 + int(c-'0')
	}
	return v, nil
}
