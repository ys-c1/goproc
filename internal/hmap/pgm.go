package hmap

import (
	"bytes"
	"errors"
	"fmt"
)

// pgmReader 顺序消费 PGM 头标记，标记间以空白或注释分隔。
type pgmReader struct {
	b   []byte
	pos int
}

func (r *pgmReader) skipSpace() error {
	for r.pos < len(r.b) {
		c := r.b[r.pos]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f' {
			r.pos++
			continue
		}
		if c == '#' {
			for r.pos < len(r.b) && r.b[r.pos] != '\n' {
				r.pos++
			}
			continue
		}
		return nil
	}
	return errors.New("pgm: 头部意外结束")
}

func (r *pgmReader) token() ([]byte, error) {
	if err := r.skipSpace(); err != nil {
		return nil, err
	}
	start := r.pos
	for r.pos < len(r.b) {
		c := r.b[r.pos]
		if c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f' {
			break
		}
		r.pos++
	}
	return r.b[start:r.pos], nil
}

func (r *pgmReader) intToken() (int, error) {
	tok, err := r.token()
	if err != nil {
		return 0, err
	}
	n := 0
	for _, c := range tok {
		if c < '0' || c > '9' {
			return 0, fmt.Errorf("pgm: 非法整数 %q", string(tok))
		}
		n = n*10 + int(c-'0')
	}
	return n, nil
}

// DecodePGM 读取 P5 八位灰度 PGM，返回边长与归一化到零到一的数据。
func DecodePGM(b []byte) (int, []float64, error) {
	r := &pgmReader{b: b}
	magic, err := r.token()
	if err != nil {
		return 0, nil, err
	}
	if !bytes.Equal(magic, []byte("P5")) {
		return 0, nil, fmt.Errorf("pgm: 仅支持 P5，收到 %q", string(magic))
	}
	w, err := r.intToken()
	if err != nil {
		return 0, nil, err
	}
	h, err := r.intToken()
	if err != nil {
		return 0, nil, err
	}
	maxv, err := r.intToken()
	if err != nil {
		return 0, nil, err
	}
	if w <= 0 || h <= 0 || w != h || maxv != 255 {
		return 0, nil, fmt.Errorf("pgm: 不支持的参数 w=%d h=%d max=%d", w, h, maxv)
	}
	// maxval 之后恰好一个空白字符，随后是原始像素。
	if r.pos >= len(b) {
		return 0, nil, errors.New("pgm: 缺少像素数据")
	}
	r.pos++
	size := w
	if len(b)-r.pos < size*size {
		return 0, nil, errors.New("pgm: 像素数据不足")
	}
	out := make([]float64, size*size)
	for i := range out {
		out[i] = float64(b[r.pos+i]) / 255
	}
	return size, out, nil
}
