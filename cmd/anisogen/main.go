// 命令行 anisogen 按预设生成各向异性地形，导出原始 float64、PGM 与参数 JSON。
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"goproc/internal/aniso"
	"goproc/internal/hmap"
)

type specComp struct {
	Seed      uint64  `json:"seed"`
	StrikeDeg float64 `json:"strike_deg"`
	Ratio     float64 `json:"ratio"`
	Frequency float64 `json:"frequency"`
	Weight    float64 `json:"weight"`
}

type spec struct {
	Name       string     `json:"name"`
	Compose    string     `json:"compose"` // linear | rms
	Size       int        `json:"size"`
	Octaves    int        `json:"octaves"`
	Lacunarity float64    `json:"lacunarity"`
	Gain       float64    `json:"gain"`
	Components []specComp `json:"components"`
}

func toComps(cs []specComp) []aniso.Component {
	out := make([]aniso.Component, len(cs))
	for i, c := range cs {
		out[i] = aniso.Component{
			Seed:      c.Seed,
			Strike:    c.StrikeDeg * math.Pi / 180,
			Ratio:     c.Ratio,
			Frequency: c.Frequency,
			Weight:    c.Weight,
		}
	}
	return out
}

// 两档各向异性与一个各向同性标定场，均用非相干能量合成（FieldRMS），
// 模拟不同期次、独立成因的脊系。
// A：强各向异性，主分量大轴比；弱旁瓣在谱上可见但低于 HOG 高度阈值。
// B：弱各向异性，两个方向簇；第三分量与第二簇间距小于间距阈值，HOG 中合并。
var presets = map[string]spec{
	"A": {
		Name: "terrain_A", Compose: "rms", Size: 512, Octaves: 2, Lacunarity: 2.0, Gain: 0.5,
		Components: []specComp{
			{Seed: 101, StrikeDeg: 30, Ratio: 4.5, Frequency: 0.03, Weight: 1.0},
			{Seed: 102, StrikeDeg: 110, Ratio: 3.0, Frequency: 0.06, Weight: 0.6},
		},
	},
	"B": {
		Name: "terrain_B", Compose: "rms", Size: 512, Octaves: 2, Lacunarity: 2.0, Gain: 0.5,
		Components: []specComp{
			{Seed: 201, StrikeDeg: 60, Ratio: 3.0, Frequency: 0.03, Weight: 1.0},
			{Seed: 202, StrikeDeg: 150, Ratio: 3.0, Frequency: 0.03, Weight: 1.0},
			{Seed: 203, StrikeDeg: 158, Ratio: 3.0, Frequency: 0.03, Weight: 0.3},
		},
	},
	"iso": {
		Name: "terrain_iso", Compose: "linear", Size: 512, Octaves: 2, Lacunarity: 2.0, Gain: 0.5,
		Components: []specComp{
			{Seed: 301, StrikeDeg: 0, Ratio: 1.0, Frequency: 0.03, Weight: 1.0},
		},
	},
}

func main() {
	preset := flag.String("preset", "A", "预设：A | B | iso")
	outdir := flag.String("outdir", "artifacts/terrains", "输出目录")
	flag.Parse()
	sp, ok := presets[*preset]
	if !ok {
		fmt.Fprintf(os.Stderr, "unknown preset %q\n", *preset)
		os.Exit(2)
	}
	var field []float64
	switch sp.Compose {
	case "rms":
		field = aniso.FieldRMS(sp.Size, sp.Octaves, sp.Lacunarity, sp.Gain, toComps(sp.Components))
	default:
		field = aniso.Field(sp.Size, sp.Octaves, sp.Lacunarity, sp.Gain, toComps(sp.Components))
	}

	if err := os.MkdirAll(*outdir, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir:", err)
		os.Exit(1)
	}
	base := filepath.Join(*outdir, sp.Name)

	raw := make([]byte, len(field)*8)
	for i, v := range field {
		binary.LittleEndian.PutUint64(raw[i*8:], math.Float64bits(v))
	}
	if err := os.WriteFile(base+".raw", raw, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write raw:", err)
		os.Exit(1)
	}
	if err := os.WriteFile(base+".pgm", hmap.EncodePGM(sp.Size, field), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write pgm:", err)
		os.Exit(1)
	}
	j, _ := json.MarshalIndent(sp, "", "  ")
	if err := os.WriteFile(base+".json", append(j, '\n'), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "write json:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s.{raw,pgm,json} size=%d comps=%d\n", base, sp.Size, len(sp.Components))
}
