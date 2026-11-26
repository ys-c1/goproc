package main

import (
	"fmt"
	"strings"

	"goproc/internal/ridge"
)

// renderReport 把两个画幅的预判与实测按行渲染成报告。
func renderReport(frames []ridge.Frame, opt options) string {
	var out []string
	out = append(out, reportTable(frames, opt.seed, opt.octaves, opt.gain, opt.freq))
	for _, f := range frames {
		out = append(out, frameDetail(f)...)
	}
	out = append(out, notes()...)
	return strings.Join(out, "\n") + "\n"
}

// reportTable 每个画幅占一行：预判总数、实测总数、偏差。
func reportTable(frames []ridge.Frame, seed uint64, octaves int, gain, freq float64) string {
	var b strings.Builder
	fmt.Fprintf(&b, "脊线像素总数：预判 vs 逐像素实测（seed=%d octaves=%d gain=%g freq=%g）\n", seed, octaves, gain, freq)
	fmt.Fprintf(&b, "两种判定都化成一维差分序列的左右符号翻转：曲率脊 D_i=κ(i+1)−κ(i)；分水脊 Δh_i=h(i+1)−h(i)。\n\n")
	fmt.Fprintf(&b, "%-10s %10s %10s %10s %10s %10s %10s %10s\n",
		"画幅", "候选像素", "曲率预判", "曲率实测", "曲率偏差", "分水预判", "分水实测", "分水偏差")
	for _, f := range frames {
		fmt.Fprintf(&b, "%-10s %10d %10.1f %10d %10s %10.1f %10d %10s\n",
			fmt.Sprintf("%dx%d", f.Size, f.Size), f.Candidates,
			f.KMin.Count, f.KMin.Measured, pctCell(f.KMin, f.KMin.Measured),
			f.Crest.Count, f.Crest.Measured, pctCell(f.Crest, f.Crest.Measured))
	}
	b.WriteString("\n")

	var over []string
	for _, f := range frames {
		over = append(over, overDeviations(f)...)
	}
	if len(over) == 0 {
		fmt.Fprintf(&b, "偏差超过一成的画幅：无。所有画幅、两种判定公式的预判偏差均在一成以内。\n")
	} else {
		fmt.Fprintf(&b, "偏差超过一成的画幅：%s。\n", strings.Join(over, "；"))
		for _, f := range frames {
			if reasons := invalidPremises(f); len(reasons) > 0 {
				fmt.Fprintf(&b, "  %dx%d 前提失效：%s。此处不是预判不准，而是该判定在该参数下退化（见说明第 4 条）。\n",
					f.Size, f.Size, strings.Join(reasons, "，"))
			}
		}
	}
	return b.String()
}

// pctCell 把模型偏差渲染成定宽单元格；实测为零时写作「—」。
func pctCell(m ridge.Model, measured int) string {
	if measured == 0 {
		return "—"
	}
	return fmt.Sprintf("%+.2f%%", 100*m.Deviation())
}

// overDeviations 列出偏差超过一成的项；实测为零时无法定义相对偏差，单独标注。
func overDeviations(f ridge.Frame) []string {
	var out []string
	name := fmt.Sprintf("%dx%d", f.Size, f.Size)
	for _, m := range []struct {
		label    string
		model    ridge.Model
		measured int
	}{
		{"曲率脊", f.KMin, f.KMin.Measured},
		{"分水脊", f.Crest, f.Crest.Measured},
	} {
		if m.measured == 0 {
			if m.model.Count >= 1 {
				out = append(out, fmt.Sprintf("%s（%s 实测为 0，预判 %.1f）", name, m.label, m.model.Count))
			}
			continue
		}
		if dev := m.model.Deviation(); dev > 0.10 || dev < -0.10 {
			out = append(out, fmt.Sprintf("%s（%s %+.1f%%）", name, m.label, 100*dev))
		}
	}
	return out
}

// invalidPremises 诊断使联合高斯近似失效的图像特征。
// 判据：差分相关系数接近 1（曲率沿横截面近乎常数或单调）、
// 超额峰度过大、偏度过大、或差分离散度落入 8 位量化噪声量级。
func invalidPremises(f ridge.Frame) []string {
	var out []string
	pr := f.KMin.Pairs
	if pr.Rho > 0.85 {
		out = append(out, fmt.Sprintf("κ 差分 ρ=%.3f 接近 1，曲率沿横截面近乎光滑", pr.Rho))
	}
	if pr.Kurt > 2 {
		out = append(out, fmt.Sprintf("κ 差分超额峰度 %.1f 远离高斯", pr.Kurt))
	}
	if pr.Skew > 1 || pr.Skew < -1 {
		out = append(out, fmt.Sprintf("κ 差分偏度 %.1f 偏离对称", pr.Skew))
	}
	if f.KMin.Measured == 0 {
		out = append(out, "实测曲率脊为零，判定在该参数下退化为空集")
	}
	return out
}

func frameDetail(f ridge.Frame) []string {
	var L []string
	L = append(L, fmt.Sprintf("── %dx%d ──", f.Size, f.Size))
	L = append(L, fmt.Sprintf("  候选像素 %d = 边长² − 4×边长（两端各两列无完整 κ 邻域，不判定）", f.Candidates))
	L = append(L, fmt.Sprintf("  曲率差分：均值 %.3e  σ %.3e  偏度 %.3f  超额峰度 %.3f", f.KMin.Pairs.Mean, f.KMin.Pairs.Sigma, f.KMin.Pairs.Skew, f.KMin.Pairs.Kurt))
	L = append(L, fmt.Sprintf("            左方差 %.3e  右方差 %.3e  相关系数 ρ=%.5f  |D|≤1e-3σ 占比 %.2e",
		f.KMin.Pairs.VarLeft, f.KMin.Pairs.VarRight, f.KMin.Pairs.Rho, f.KMin.Pairs.BandShare))
	L = append(L, fmt.Sprintf("  曲率脊：P = 1/4 − arcsin(ρ)/(2π) = %.6f    预判 %.1f    实测 %d    偏差 %+.3f%%",
		f.KMin.P, f.KMin.Count, f.KMin.Measured, 100*f.KMin.Deviation()))
	L = append(L, fmt.Sprintf("  独立同分布对照：只用曲率边际分布得 P=1/3，预判 %.0f，为实测的 %.2f 倍 —— 相邻曲率强正相关，边际假设不成立", f.IIDCount, f.IIDCount/float64(f.KMin.Measured)))
	L = append(L, fmt.Sprintf("  分水脊（高度极大）：ρ=%.5f  P=%.6f  预判 %.1f  实测 %d  偏差 %+.3f%%",
		f.Crest.Pairs.Rho, f.Crest.P, f.Crest.Count, f.Crest.Measured, 100*f.Crest.Deviation()))
	L = append(L, fmt.Sprintf("  两类像素重合 %d：P(分水|曲率)=%.1f%%，P(曲率|分水)=%.1f%%",
		f.Both, pct(f.Both, f.KMin.Measured), pct(f.Both, f.Crest.Measured)))
	L = append(L, fmt.Sprintf("  并列差分（dL 或 dR 恰为 0）%d 处，不计入严格极小", f.Ties))
	h := ridge.ComponentHistogram(f.Comps)
	L = append(L, fmt.Sprintf("  曲率脊连通段：共 %d 段；按像素数 1 / 2–3 / 4–10 / 11–50 / >50 分别为 %d / %d / %d / %d / %d 段",
		len(f.Comps), h[0], h[1], h[2], h[3], h[4]))
	top := ridge.TopComponents(f.Comps, 5)
	var ts []string
	for _, c := range top {
		ts = append(ts, fmt.Sprintf("%d", c.Pixels))
	}
	if len(ts) > 0 {
		L = append(L, fmt.Sprintf("  最大五段像素数：%s", strings.Join(ts, "、")))
	}
	length, edges := ridge.TotalLength(f.Mask)
	L = append(L, fmt.Sprintf("  脊线总长（连通段内最小生成树边长）%.1f 像素，共 %d 条边，长/像素 = %.3f", length, edges, length/float64(f.KMin.Measured)))
	L = append(L, "")
	return L
}

func pct(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return 100 * float64(a) / float64(b)
}

func notes() []string {
	return []string{
		"说明",
		"  1. 「脊线」有两种定义，都在候选列上判定，都化成一维差分序列的左右符号翻转：",
		"     · 曲率脊：横截面二阶中心差分 κ(x)=h(x−1)−2h(x)+h(x+1) 的严格局部极小；",
		"     · 分水脊：高度 h(x) 沿横截面的严格局部极大。两者不是同一集合，见第 3 条。",
		"  2. 预判公式。记相邻位置差分为 D_i，左右差分对 (D_i, D_{i+1}) 的相关系数为 ρ。",
		"     在零均值、等方差的联合高斯近似下，异号且左负右正的概率为 1/4 − arcsin(ρ)/(2π)，",
		"     预判总数 = 该概率 × 候选像素数。ρ 由全图差分统计一次算得，不需要逐条追踪脊线；",
		"     实测总数由逐像素符号翻转判定独立得到，二者按画幅比对。",
		"  3. 「脊线像素必落在曲率取极小的位置」作为几何命题不成立。",
		"     反例：h(x)=exp(−x²/2)+0.35·exp(−(x−1.5)²/0.4)，高度极大在 x≈0.01，",
		"     而曲率极小在 x≈1.49。曲率小刻画的是「平坦或陡峭」而不是「高或低」，",
		"     曲率极小可以是山脊、谷底、肩部或斜坡的拐折。实测两集合的重合率仅约一成，",
		"     所以曲率公式只能预判「曲率脊」的数目，不能预判山脊的数目；山脊数目要另用分水脊公式预判。",
		"  4. 预判公式的前提是差分对近似联合高斯：零均值、等方差。默认参数下超额峰度约 0.4，",
		"     偏度约 0，左右方差基本相等，前提成立。低层数、低频率（如 octaves=1、freq≤0.01）时",
		"     地形近乎一张平滑曲面，曲率沿横截面近乎常数，差分 ρ→1、峰度骤增，符号翻转退化成",
		"     空集或极少点，偏差可达数百个百分点。报告对这类画幅在「前提失效」下点名，",
		"     它反映的是判定在该参数下无定义，不是预判精度问题。",
		"     |D| 落在 1e-3σ 数值歧义带内的比例约 1e-3，符号判定的并列与浮点噪声可以忽略。",
		"  5. 预判只用每行的水平差分，不做跨行配对；若地形存在横向条带结构，",
		"     水平与竖直两个方向的预判会有差别，应分别报告。",
		"  6. 总长取连通段内相邻像素的最小生成树边长（直步 1、对角 √2），是折线长度的下界；",
		"     单像素段计 0。它把「逐条量」的代价降到一次连通标记加一次排序。",
		"  7. 独立复核：python3 -I tools/verify_ridge.py artifacts/ridge <画幅>，不引用 Go 代码，",
		"     从 height_*.f64 重算并与 ridge_mask_*.pgm 逐像素比对。必须用浮点高度：",
		"     8 位 PGM 的量化步长（约 3.9e-3）大于曲率均值（约 3.5e-3），",
		"     从 PGM 复算会把曲率压成十几个取值、制造上千个并列，实测从 2818 跳到 4442。",
	}
}
