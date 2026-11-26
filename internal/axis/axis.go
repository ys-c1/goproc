// Package axis constructs phase-locked finite-layer terrains and compares
// phase-derived symmetry axes with exhaustive column-mirror correlation.
package axis

import (
	"math"

	"goproc/internal/hmap"
	"goproc/internal/noise"
)

// LayerSpec records one finite noise layer's horizontal phase lock.
//
// The carrier is cos(omega*x + PhiUnwrapped). Order is the integer m in
// omega*axis + PhiUnwrapped = m*pi. PhiWrapped is the same phase reduced to
// (-pi, pi].
type LayerSpec struct {
	Index        int     `json:"index"`
	Cycles       float64 `json:"cycles"`
	Amplitude    float64 `json:"amplitude"`
	Omega        float64 `json:"omega"`
	Order        int     `json:"order"`
	PhiUnwrapped float64 `json:"phi_unwrapped"`
	PhiWrapped   float64 `json:"phi_wrapped"`
	PhaseDrift   float64 `json:"phase_drift,omitempty"`
}

// Case is one generated image and its known zero-based symmetry column.
type Case struct {
	Index  int         `json:"index"`
	Name   string      `json:"name"`
	Width  int         `json:"width"`
	Height int         `json:"height"`
	Seed   uint64      `json:"seed"`
	Axis   int         `json:"axis"`
	Layers []LayerSpec `json:"layers"`
	Data   []float64   `json:"-"`
}

// Score is the normalized mirror correlation for one candidate column.
type Score struct {
	Column int     `json:"column"`
	R      float64 `json:"r"`
	Pairs  int     `json:"pairs"`
}

// PixelCheck is an exact pixelwise confirmation at one candidate column.
type PixelCheck struct {
	Axis                int     `json:"axis"`
	Pairs               int     `json:"pairs"`
	SelfPairs           int     `json:"self_pairs"`
	NonSelfObservations int     `json:"non_self_observations"`
	Viable              bool    `json:"viable"`
	ExactEqual          int     `json:"exact_equal"`
	Mismatches          int     `json:"mismatches"`
	MaxAbsDiff          float64 `json:"max_abs_diff"`
	QuantizedBad        int     `json:"quantized_mismatches"`
}

// Prediction is the O(K) phase-sum axis estimate.
type Prediction struct {
	SumOmega      float64   `json:"sum_omega"`
	SumPhi        float64   `json:"sum_phi_unwrapped"`
	SumOrder      int       `json:"sum_order"`
	AxisFloat     float64   `json:"axis_float"`
	RoundedAxis   int       `json:"rounded_axis"`
	LayerAxes     []float64 `json:"layer_axes"`
	MeanLayerAxis float64   `json:"mean_layer_axis"`
}

// DefaultCycles are deliberately non-integer and mutually incommensurate over
// either demonstration width, preventing W/2 periodic alias axes from tying
// the true axis.
var DefaultCycles = []float64{2.31, 4.17, 7.63, 13.29}

// Generate creates a normalized finite-layer terrain exactly mirror symmetric
// about axis. The continuous carriers satisfy the phase-lock equation; each
// discrete layer is then pair-averaged and assigned one shared value so the
// pixel equality is exact rather than a floating-point near miss.
func Generate(name string, width, height, axis, octaves int, seed uint64) *Case {
	return GenerateWithDrift(name, width, height, axis, octaves, seed, -1, 0)
}

// GenerateWithDrift intentionally breaks one layer's phase lock. It is used as
// a negative control for the audit and must not be treated as a valid proof
// case for phase-first axis prediction.
func GenerateWithDrift(name string, width, height, axis, octaves int, seed uint64, driftLayer int, driftRadians float64) *Case {
	if octaves <= 0 {
		octaves = len(DefaultCycles)
	}
	if octaves > len(DefaultCycles) {
		octaves = len(DefaultCycles)
	}

	c := &Case{
		Index: 0,
		Name:  name,
		Width: width, Height: height, Seed: seed, Axis: axis,
		Layers: make([]LayerSpec, 0, octaves),
		Data:   make([]float64, width*height),
	}

	for k := 0; k < octaves; k++ {
		cycles := DefaultCycles[k]
		omega := 2 * math.Pi * cycles / float64(width)

		// Deterministic small integer phase orders make the alias bookkeeping
		// visible without making all raw phases identical.
		orderSrc := noise.New(seed ^ uint64(0x9e3779b97f4a7c15+uint64(k)*7919))
		order := int(math.Round(orderSrc.Value(0.37, 11.7+float64(k)))) * 2
		if order < -4 {
			order = -4
		}
		if order > 4 {
			order = 4
		}

		phaseDrift := 0.0
		if k == driftLayer {
			phaseDrift = driftRadians
		}
		phi := math.Pi*float64(order) - omega*float64(axis) + phaseDrift
		spec := LayerSpec{
			Index:        k,
			Cycles:       cycles,
			Amplitude:    math.Pow(0.5, float64(k)),
			Omega:        omega,
			Order:        order,
			PhiUnwrapped: phi,
			PhiWrapped:   math.Atan2(math.Sin(phi), math.Cos(phi)),
			PhaseDrift:   phaseDrift,
		}
		c.Layers = append(c.Layers, spec)

		envSrc := noise.New(seed + uint64(k+1)*104729)
		layer := make([]float64, width*height)
		for y := 0; y < height; y++ {
			for x := 0; x < width; x++ {
				// A seeded 2-D value-noise envelope gives the finite carrier
				// terrain texture. Pair projection below makes its discrete
				// mirror equality exact at the known phase axis.
				envelope := 0.72 + 0.28*envSrc.Value(
					float64(x)*0.055+float64(k)*13.17,
					float64(y)*0.075+float64(k)*3.11,
				)
				carrier := math.Cos(omega*float64(x) + phi)
				layer[y*width+x] = spec.Amplitude * envelope * carrier
			}
			if phaseDrift == 0 {
				symmetrizePairs(layer[y*width:y*width+width], axis)
			}
		}

		for i, v := range layer {
			c.Data[i] += v
		}
	}

	hmap.Normalize(c.Data)
	return c
}

func symmetrizePairs(row []float64, axis int) {
	width := len(row)
	lo := maxInt(0, 2*axis-(width-1))
	for x := lo; x < axis; x++ {
		j := 2*axis - x
		if j < 0 || j >= width {
			continue
		}
		shared := (row[x] + row[j]) * 0.5
		row[x] = shared
		row[j] = shared
	}
}

// PredictAxis evaluates sum_k (omega_k*a + phi_k) = pi*sum m_k.
// Phi must be the unwrapped phase recorded at generation time.
func PredictAxis(layers []LayerSpec) Prediction {
	p := Prediction{LayerAxes: make([]float64, len(layers))}
	for i, l := range layers {
		p.SumOmega += l.Omega
		p.SumPhi += l.PhiUnwrapped
		p.SumOrder += l.Order
		a := (math.Pi*float64(l.Order) - l.PhiUnwrapped) / l.Omega
		p.LayerAxes[i] = a
		p.MeanLayerAxis += a
	}
	if len(layers) > 0 {
		p.MeanLayerAxis /= float64(len(layers))
	}
	p.AxisFloat = (math.Pi*float64(p.SumOrder) - p.SumPhi) / p.SumOmega
	p.RoundedAxis = int(math.Round(p.AxisFloat))
	return p
}

// overlapColumns returns the inclusive source-x interval whose mirror column
// 2*c-x lies inside [0,width-1].
func overlapColumns(width, c int) (lo, hi int) {
	lo = maxInt(0, 2*c-(width-1))
	hi = minInt(width-1, 2*c)
	return lo, hi
}

// MirrorCorrelation computes Pearson correlation between paired pixels h(x,y)
// and h(2c-x,y) over the valid overlap.
func MirrorCorrelation(data []float64, width, height, c int) Score {
	lo, hi := overlapColumns(width, c)
	if hi < lo {
		return Score{Column: c}
	}

	var n, su, sv, suu, svv, suv float64
	for y := 0; y < height; y++ {
		base := y * width
		for x := lo; x <= hi; x++ {
			j := 2*c - x
			u := data[base+x]
			v := data[base+j]
			n++
			su += u
			sv += v
			suu += u * u
			svv += v * v
			suv += u * v
		}
	}

	score := Score{Column: c, Pairs: int(n)}
	du := n*suu - su*su
	dv := n*svv - sv*sv
	if du <= 0 || dv <= 0 {
		score.R = math.NaN()
		return score
	}
	score.R = (n*suv - su*sv) / math.Sqrt(du*dv)
	return score
}

// ScanColumns evaluates every integer column with at least minOverlapColumns
// mirrored source columns. An overlap of one is the degenerate image-edge
// self-pair c=0 or c=W-1; callers normally pass 3 to include c=1..W-2 while
// excluding those trivial rho=1 self matches. This audit reference costs
// O(H W^2), not the O(K+HW) phase-first path.
func ScanColumns(c *Case, minOverlapColumns int) []Score {
	if minOverlapColumns < 3 {
		minOverlapColumns = 3
	}
	out := make([]Score, 0, c.Width)
	for col := 0; col < c.Width; col++ {
		lo, hi := overlapColumns(c.Width, col)
		if hi-lo+1 < minOverlapColumns {
			continue
		}
		out = append(out, MirrorCorrelation(c.Data, c.Width, c.Height, col))
	}
	return out
}

// BestColumn returns the smallest column within eps of the largest score.
// The explicit tie rule keeps periodic or degenerate aliases reproducible.
func BestColumn(scores []Score, eps float64) (Score, []int) {
	bestIdx := -1
	bestR := -math.MaxFloat64
	for i, s := range scores {
		if math.IsNaN(s.R) {
			continue
		}
		if s.R > bestR {
			bestR = s.R
			bestIdx = i
		}
	}
	if bestIdx < 0 {
		return Score{}, nil
	}
	best := scores[bestIdx]
	tied := []int{}
	for _, s := range scores {
		if !math.IsNaN(s.R) && s.R >= bestR-eps {
			tied = append(tied, s.Column)
		}
	}
	if len(tied) > 0 {
		for _, s := range scores {
			if s.Column == tied[0] {
				best = s
				break
			}
		}
	}
	return best, tied
}

// ConfirmAxis performs the requested pixel-by-pixel check at one column.
func ConfirmAxis(c *Case, column int) PixelCheck {
	lo, hi := overlapColumns(c.Width, column)
	check := PixelCheck{Axis: column}
	if hi < lo {
		return check
	}
	for y := 0; y < c.Height; y++ {
		base := y * c.Width
		for x := lo; x <= hi; x++ {
			j := 2*column - x
			u := c.Data[base+x]
			v := c.Data[base+j]
			if x == column {
				check.SelfPairs++
			}
			check.Pairs++
			d := math.Abs(u - v)
			if d == 0 {
				check.ExactEqual++
			} else {
				check.Mismatches++
			}
			if d > check.MaxAbsDiff {
				check.MaxAbsDiff = d
			}
			if hmap.QuantizeGray(u) != hmap.QuantizeGray(v) {
				check.QuantizedBad++
			}
		}
	}
	// The center column maps to itself once per row. Every other scanned
	// observation belongs to a real left/right mirror pair (and both members
	// are observed). An image-edge candidate has only these self observations.
	check.NonSelfObservations = check.Pairs - check.SelfPairs
	check.Viable = check.NonSelfObservations > 0
	return check
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
