package main

import (
	"flag"
)

// flagSet 包住命令行参数解析，便于测试。
// 参数在构造时一次性注册，避免解析前漏注册。
type flagSet struct {
	fs     *flag.FlagSet
	sizes  *string
	seed   *uint64
	octave *int
	gain   *float64
	freq   *float64
	outDir *string
}

func newFlagSet() *flagSet {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	return &flagSet{
		fs:     fs,
		sizes:  fs.String("sizes", "128,256", "画幅边长，逗号分隔"),
		seed:   fs.Uint64("seed", 1, "种子"),
		octave: fs.Int("octaves", 4, "分形层数"),
		gain:   fs.Float64("gain", 0.5, "振幅衰减"),
		freq:   fs.Float64("freq", 0.02, "基础频率"),
		outDir: fs.String("out", "artifacts/ridge", "输出目录"),
	}
}

// parseArgs 构造参数集、解析并取值为 options。
func parseArgs(args []string) (options, error) {
	f := newFlagSet()
	if err := f.fs.Parse(args); err != nil {
		return options{}, err
	}
	return f.options()
}

func (f *flagSet) options() (options, error) {
	ns, err := parseSizes(*f.sizes)
	if err != nil {
		return options{}, err
	}
	return options{
		sizes:   ns,
		seed:    *f.seed,
		octaves: *f.octave,
		gain:    *f.gain,
		freq:    *f.freq,
		outDir:  *f.outDir,
	}, nil
}
