package brain

import "math"

type ActivationFunc struct {
	Derivate func(float32) float32
	Activate func(float32) float32
}

func relu(x float32) float32 {
	return float32(math.Max(0, float64(x)))

}
func sigmoid(x float32) float32 {

	return float32(1 / (1 + math.Exp(float64(x)*(-1))))

}

func tanh(x float32) float32 {
	return float32(math.Tanh(float64(x)))
}

func devSigmoid(x float32) float32 {
	return x * (1 - x)
}
func devRelu(x float32) float32 {
	out := 0.0
	if x > 0 {
		out = 1
	}
	return float32(out)
}
func devTanh(x float32) float32 {
	return 1 - (x * x)

}

var (
	MathFuncs = map[string]ActivationFunc{
		"sigmoid": {
			Derivate: devSigmoid,
			Activate: sigmoid,
		},
		"relu": {
			Derivate: devRelu,
			Activate: relu,
		},
		"tanh": {
			Derivate: devTanh,
			Activate: tanh,
		},
	}
)
