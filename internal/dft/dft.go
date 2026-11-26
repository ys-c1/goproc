// Package dft 提供仅依赖标准库的基二复数 FFT。
// 约定：输入长度必须是 2 的幂；正变换不归一化，逆变换带 1/N。
package dft

import "math"

// FFT 原地基二 Cooley–Tukey 迭代 FFT，a 的长度必须是 2 的幂。
// inverse=false 为正变换，true 为带 1/N 归一化的逆变换。
func FFT(a []complex128, inverse bool) {
	n := len(a)
	if n&(n-1) != 0 {
		panic("dft: 输入长度不是 2 的幂")
	}
	// 位逆序置换。
	for i, j := 1, 0; i < n; i++ {
		bit := n >> 1
		for ; j&bit != 0; bit >>= 1 {
			j ^= bit
		}
		j ^= bit
		if i < j {
			a[i], a[j] = a[j], a[i]
		}
	}
	sign := 1.0
	if inverse {
		sign = -1.0
	}
	for length := 2; length <= n; length <<= 1 {
		half := length >> 1
		ang := sign * 2 * math.Pi / float64(length)
		wlen := complex(math.Cos(ang), math.Sin(ang))
		for i := 0; i < n; i += length {
			w := complex(1, 0)
			for j := 0; j < half; j++ {
				u := a[i+j]
				v := a[i+j+half] * w
				a[i+j] = u + v
				a[i+j+half] = u - v
				w *= wlen
			}
		}
	}
	if inverse {
		inv := 1 / float64(n)
		for i := range a {
			a[i] *= complex(inv, 0)
		}
	}
}

// FFT2 对 n×n 的行优先二维实数场做二维正变换，返回 n×n 复数谱。
func FFT2(real []float64, n int) []complex128 {
	if len(real) != n*n {
		panic("dft: 二维输入尺寸不匹配")
	}
	buf := make([]complex128, n*n)
	for i, v := range real {
		buf[i] = complex(v, 0)
	}
	row := make([]complex128, n)
	for y := 0; y < n; y++ {
		copy(row, buf[y*n:(y+1)*n])
		FFT(row, false)
		copy(buf[y*n:(y+1)*n], row)
	}
	col := make([]complex128, n)
	for x := 0; x < n; x++ {
		for y := 0; y < n; y++ {
			col[y] = buf[y*n+x]
		}
		FFT(col, false)
		for y := 0; y < n; y++ {
			buf[y*n+x] = col[y]
		}
	}
	return buf
}

// Freq 返回长度为 n 的 DFT 角频率（单位：弧度/采样间距），
// 即 2πk/n，k 按 0,1,...,n/2-1,-n/2,...,-1 排列。
func Freq(n int) []float64 {
	w := make([]float64, n)
	for k := 0; k < n; k++ {
		kk := k
		if kk > n/2 {
			kk -= n
		}
		w[k] = 2 * math.Pi * float64(kk) / float64(n)
	}
	return w
}
