// Command tilecheck 校验平铺接缝，并把逐行相位差与跳变写到留档文件。
//
// 用法见 artifacts/verify.sh。输出：
//   - tile_metrics.tsv：三种周期 × 每一层的相位、余数、整除结论与逐层跳变
//   - tile_summary.tsv：每档一张图的接缝跳变、接缝相邻差、内部相邻差
//   - tile_phase_sweep.tsv：固定周期、相位递增时，跳变随余数变化
//   - tile_stats.txt：汇总与阈值结论
//   - tile_<name>_seed<S>.pgm/.png 与 _2x2：单图与拼接图
package main

import (
	"flag"
	"fmt"
	"image"
	"image/png"
	"os"
	"path/filepath"
	"strings"

	"goproc/api"
	"goproc/internal/seam"
)

type config struct {
	name   string
	cells  int // 第零层横跨的格点数，等于相位
	period int // 置换表周期（格点）
}

func main() {
	out := flag.String("out", "artifacts", "输出目录")
	size := flag.Int("size", 512, "边长像素数")
	seed := flag.Uint64("seed", 7, "种子")
	octaves := flag.Int("octaves", 3, "层数")
	reps := flag.Int("reps", 2, "拼接重数")
	flag.Parse()

	// 三档周期：64、128、256 个格点。第零层相位分别取与周期相等的格点数，
	// 频率由 TileCells/Size 推出，因此第零层恰好横跨置换表的一个整周期。
	configs := []config{
		{name: "P64", cells: 64, period: 64},
		{name: "P128", cells: 128, period: 128},
		{name: "P256", cells: 256, period: 256},
	}

	if err := os.MkdirAll(*out, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	var tsv strings.Builder
	tsv.WriteString("tile\tperiod\tseed\tsize\toctave\tphase_cells\tphase_remainder\tdividible\t" +
		"octave_jump_x\toctave_jump_y\n")

	var summary strings.Builder
	summary.WriteString("tile\tperiod\tseed\tsize\tjump_x\tjump_y\tstep_x\tstep_y\t" +
		"interior_x\tinterior_y\tbelow_one_gray\tfault\n")

	var stats strings.Builder
	fmt.Fprintf(&stats, "三档周期接缝校验 seed=%d size=%d octaves=%d lacunarity=2 gain=0.5\n", *seed, *size, *octaves)
	fmt.Fprintf(&stats, "一档灰度 = 1/255 = %.10f\n\n", seam.GrayStep)

	allPass := true
	for _, c := range configs {
		opt := api.DefaultOptions()
		opt.Size = *size
		opt.Seed = *seed
		opt.Octaves = *octaves
		opt.TileCells = c.cells
		opt.Period = c.period
		h, err := api.Generate(opt)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: 生成失败: %v\n", c.name, err)
			os.Exit(1)
		}
		m := seam.Measure(h)
		phases, rems, divs := h.Options().TileReport()

		for i := range phases {
			fmt.Fprintf(&tsv, "%s\t%d\t%d\t%d\t%d\t%g\t%g\t%v\t%.6e\t%.6e\n",
				c.name, c.period, *seed, h.Size, i, phases[i], rems[i], divs[i],
				m.OctaveJumpX[i], m.OctaveJumpY[i])
		}

		belowGray := m.BelowOneGray()
		fmt.Fprintf(&summary, "%s\t%d\t%d\t%d\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%v\t%v\n",
			c.name, c.period, *seed, h.Size,
			m.JumpX, m.JumpY, m.StepX, m.StepY, m.InteriorX, m.InteriorY,
			belowGray, m.Fault())
		fault := m.Fault()
		stepOK := !m.StepExceeds()
		if !belowGray || fault {
			allPass = false
		}

		fmt.Fprintf(&stats, "[%s] 置换表周期 %d 格点，第零层相位 %g 格点\n", c.name, c.period, phases[0])
		for i := range phases {
			fmt.Fprintf(&stats, "  第 %d 层  相位 %6g 格点  余数 %6g  整除 %v  跳变 x=%.3e y=%.3e\n",
				i, phases[i], rems[i], divs[i], m.OctaveJumpX[i], m.OctaveJumpY[i])
		}
		fmt.Fprintf(&stats, "  相位跳变   |h(Size)-h(0)| x=%.3e y=%.3e  (上限 1 档灰度=%.3e)  %s\n",
			m.JumpX, m.JumpY, seam.GrayStep, passWord(belowGray))
		fmt.Fprintf(&stats, "  接缝相邻差 x=%.3e y=%.3e  内部相邻差最大值 x=%.3e y=%.3e  %s\n",
			m.StepX, m.StepY, m.InteriorX, m.InteriorY,
			map[bool]string{true: "未超出内部基线", false: "超出内部基线"}[stepOK])
		fmt.Fprintf(&stats, "  结论: %s\n\n",
			map[bool]string{true: "无缝", false: "存在断层"}[!fault && belowGray])

		base := filepath.Join(*out, fmt.Sprintf("tile_%s_seed%d", c.name, *seed))
		writePGM(base+".pgm", h.ToPGM())
		writePNG(base+".png", h.Size, h.Data)
		writePGM(base+"_2x2.pgm", h.ToPGMRepeat(*reps))
		writePNG(base+"_2x2.png", h.Size*(*reps), h.Repeat(*reps))
	}

	// 余数扫描：固定周期 64，让第零层相位从 64 增到 95，即余数依次为 0..31。
	// Generate 仍按 Frequency 生成（不走 TileCells），因此不整除的参数也能出图，
	// 正好用来验证式 (2.3)：跳变随余数变化，且余数为零时归零。
	sweep := &strings.Builder{}
	sweep.WriteString("period\tcells\tremainder\tjump\tstep\tinterior\n")
	type row struct {
		cells, rem int
		jump, step float64
		interior   float64
	}
	var rows []row
	for cells := 64; cells <= 95; cells++ {
		opt := api.DefaultOptions()
		opt.Size = *size
		opt.Seed = *seed
		opt.Octaves = *octaves
		opt.Frequency = float64(cells) / float64(*size)
		opt.Period = 64
		h, err := api.Generate(opt)
		if err != nil {
			fmt.Fprintf(os.Stderr, "sweep cells=%d: %v\n", cells, err)
			os.Exit(1)
		}
		m := seam.Measure(h)
		rem, _ := h.Options().RemainderCells(0)
		rows = append(rows, row{cells, rem, m.JumpMax(), m.StepMax(), m.InteriorMax()})
		fmt.Fprintf(sweep, "64\t%d\t%d\t%.6e\t%.6e\t%.6e\n", cells, rem,
			m.JumpMax(), m.StepMax(), m.InteriorMax())
	}

	fmt.Fprintf(&stats, "相位余数扫描（周期 64，第零层相位 64..95；余数 0 时跳变为零，非零时跳变随余数变化）\n")
	fmt.Fprintf(&stats, "  cells  rem   jump        step        interior\n")
	for _, r := range rows {
		fmt.Fprintf(&stats, "  %5d  %3d   %.6e  %.6e  %.6e\n", r.cells, r.rem, r.jump, r.step, r.interior)
	}
	fmt.Fprintf(&stats, "\n总判定: %s\n",
		map[bool]string{true: "通过——三档周期接缝跳变均低于一档灰度且无断层", false: "未通过"}[allPass])

	write := func(name, s string) {
		if err := os.WriteFile(filepath.Join(*out, name), []byte(s), 0o644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
	}
	write("tile_metrics.tsv", tsv.String())
	write("tile_summary.tsv", summary.String())
	write("tile_phase_sweep.tsv", sweep.String())
	write("tile_stats.txt", stats.String())

	fmt.Print(stats.String())
	if !allPass {
		os.Exit(1)
	}
}

func passWord(ok bool) string {
	if ok {
		return "低于一档灰度"
	}
	return "未低于一档灰度"
}

func writePGM(name string, b []byte) {
	if err := os.WriteFile(name, b, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func writePNG(name string, size int, v []float64) {
	img := image.NewGray(image.Rect(0, 0, size, size))
	for i, x := range v {
		g := int(x*255 + 0.5)
		if g < 0 {
			g = 0
		}
		if g > 255 {
			g = 255
		}
		img.Pix[i] = byte(g)
	}
	f, err := os.Create(name)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()
	if err := png.Encode(f, img); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
